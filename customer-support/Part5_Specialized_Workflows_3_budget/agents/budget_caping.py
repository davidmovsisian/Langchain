"""
What is covered:
- NODES EXECUTED IN SEQUENTUAL ORDER, NO PARALLEL NODES EXECUTION IS ALLOWED. 
- Model downgrade when a node exceeds its slice
- Peer adjustment so node_remaining stays consistent with global_remaining
- Hard stop via BudgetExceededError before overspending
- Conservative fallback when usage metadata is missing
- Startup warnings for misconfigured fractions and incompatible max_tokens
- Estimation accuracy report - tells you if primary_max_tokens values are miscalibrated
- Per-node call counter + early conservative switch - identifies the trend of node usage and enables early switch to fallback model
"""

from dataclasses import dataclass, field
from contextvars import ContextVar
from langchain.chat_models import init_chat_model
import litellm
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
import logging
from typing import Any,Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Budget domain types
# ---------------------------------------------------------------------------
class BudgetExceededError(RuntimeError):
    """Raised when neither the primary nor the fallback model fits the budget."""

    def __init__(
        self,
        node_name: str,
        primary_expected: float,
        fallback_expected: float,
        remaining: float,
    ) -> None:
        self.node_name = node_name
        self.primary_expected = primary_expected
        self.fallback_expected = fallback_expected
        self.remaining = remaining
        super().__init__(
            f"Budget exceeded for '{node_name}': "
            f"primary=${primary_expected:.6f}, "
            f"fallback=${fallback_expected:.6f}, "
            f"global remaining=${remaining:.6f}"
        )


@dataclass(frozen=True)
class BudgetPolicy:
    """
    Per-agent budget policy declared at agent-definition time.

    budget_fraction is the node's preferred share of total_budget (soft
    allocation). Fractions across all nodes need not sum to 1 — the leftover
    becomes the unallocated_pool, a shared spare pool visible in report().
    Must be in (0, 1].

    primary_max_tokens / fallback_max_tokens are the max output token ceilings
    passed to each model. They are used for worst-case cost estimation before
    each call. A compatibility check in GraphBudget.register_policy() warns at
    startup if the output cost alone exceeds the node's allocation.
    """

    budget_fraction: float
    primary_model: str
    primary_max_tokens: int
    fallback_model: str
    fallback_max_tokens: int
    trend_cntr: int #if number of node calls > trend_cntr, early swith to fallback model to preserve the node's budget

    def __post_init__(self) -> None:
        if not (0.0 < self.budget_fraction <= 1.0):
            raise ValueError(
                f"budget_fraction must be in (0, 1], got {self.budget_fraction}"
            )
        if self.primary_max_tokens <= 0 or self.fallback_max_tokens <= 0:
            raise ValueError("max_tokens values must be positive integers")


@dataclass
class LLMInvocationCost:
    """One record written to the ledger after each model call."""

    node_name: str
    model: str                          # model that was actually used
    estimated_input_tokens: int         # token count computed BEFORE the call
    estimated_cost: float               # worst-case cost computed BEFORE the call
    max_tokens: int                     # output ceiling passed to the model
    actual_cost: float | None = None    # cost computed AFTER the call
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None


# ---------------------------------------------------------------------------
# GraphBudget — central budget ledger
# ---------------------------------------------------------------------------

@dataclass
class GraphBudget:
    """
    Tracks global spend and per-node spend for one graph execution.

    Sequential execution is assumed, so no locking is used. If parallel
    branches are introduced later, wrap all mutations in a threading.Lock.

    Weighted budget partitioning + dynamic reallocation
    ---------------------------------------------------
    Each node declares a budget_fraction (its preferred share of total_budget).
    This divides the budget into:

        node_allocation  = total_budget * fraction        (soft per-node target)
        unallocated_pool = total_budget - sum(allocations) (spare pool, diagnostic only)

    On every model call, budget_middleware computes the effective node_remaining:

        node_remaining = node_allocation
                         - node_spend        (what this node has spent)
                         - node_adjustment   (its share of what other nodes borrowed)

    The node_adjustment term is the reallocation mechanism. When a node spends
    beyond its own slice (excess_spend > 0), that excess is deducted from all
    other nodes proportionally to their fractions via _deduct_excess_from_peers().
    This keeps node_remaining values consistent with global_remaining at all times.

    Model selection on each call follows four branches:

        primary_expected <= node_remaining
            → use primary (within own slice)

        primary_expected > node_remaining AND primary_expected <= global_remaining
            → use fallback (node exhausted its slice; borrowing conservatively)

        primary_expected > global_remaining AND fallback_expected <= global_remaining
            → use fallback (last affordable option)

        fallback_expected > global_remaining
            → BudgetExceededError

    Naming guide
    ------------
    unallocated_pool   budget left unallocated by design (fractions sum < 1.0).
                       Diagnostic only — never used in decision logic.
    excess_spend       the portion of one node's spend that exceeded its own slice.
    _node_adjustment   per-node cumulative deduction from peers' excess spend.
    _deduct_excess_from_peers()  propagates excess_spend to other nodes.
    """

    total_budget: float
    actual_spend: float = 0.0
    _node_spend: dict[str, float] = field(default_factory=dict, repr=False)
    _node_adjustment: dict[str, float] = field(default_factory=dict, repr=False)
    _policies: dict[str, BudgetPolicy] = field(default_factory=dict, repr=False)
    _node_call_count: dict[str, int] = field(default_factory=dict, repr=False)
    records: list[LLMInvocationCost] = field(default_factory=list)

    # --------------- derived properties ------------------------------------

    @property
    def remaining(self) -> float:
        """Global unspent budget."""
        return max(0.0, self.total_budget - self.actual_spend)

    @property
    def unallocated_pool(self) -> float:
        """
        Budget not claimed by any node's soft allocation.
        Diagnostic only — confirms that fractions were configured as intended.
        Not used in any decision logic inside budget_middleware.
        """
        claimed = sum(
            self.total_budget * p.budget_fraction
            for p in self._policies.values()
        )
        return max(0.0, self.total_budget - claimed)

    # --------------- registration ------------------------------------------

    def register_policy(self, node_name: str, policy: BudgetPolicy) -> None:
        """
        Register a node's BudgetPolicy into the ledger.

        Called automatically by _flush_pending_registrations() on the first
        model invocation after set_graph_budget() is called in main.py.
        Idempotent: registering the same node_name twice is safe.

        Two startup warnings are emitted:

        1. Fraction overflow — if the sum of all registered fractions exceeds
           1.0, the unallocated_pool is negative (over-subscribed). Nodes will
           start competing for shared budget sooner than expected.

        2. max_tokens compatibility — if a model's output cost alone (with zero
           input tokens, the cheapest possible call) already exceeds the node's
           entire allocation, the node can never use that model within its own
           slice. Catches mismatches between total_budget, budget_fraction, and
           max_tokens at startup rather than mid-conversation.
        """
        if node_name in self._policies:
            return  # idempotent

        # ── Warning 1: fraction over-subscription ───────────────────────────
        total_fraction = (
            sum(p.budget_fraction for p in self._policies.values())
            + policy.budget_fraction
        )
        if total_fraction > 1.0 + 1e-9:
            logger.warning(
                "register_policy: total budget_fraction across all nodes is "
                "%.2f > 1.0 after adding '%s' (fraction=%.2f). "
                "The unallocated_pool is over-subscribed; nodes will compete "
                "for shared budget sooner than expected.",
                total_fraction,
                node_name,
                policy.budget_fraction,
            )

        self._policies[node_name] = policy

        # ── Warning 2: max_tokens vs node allocation compatibility ──────────
        # Use input_tokens=0 as a lower bound: if output cost alone already
        # exceeds the allocation, no real call will ever fit the node's slice.
        node_allocation = self.total_budget * policy.budget_fraction

        min_primary_cost = _cost_for_tokens(
            _litellm_model_name(policy.primary_model),
            input_tokens=0,
            output_tokens=policy.primary_max_tokens,
        )
        if min_primary_cost > node_allocation:
            logger.warning(
                "register_policy: '%s' primary model output cost alone "
                "(model=%s, max_tokens=%d) is $%.6f, which already exceeds "
                "its entire node allocation of $%.6f (%.0f%% of $%.2f budget). "
                "This node will always downgrade to fallback on its first call. "
                "Consider reducing primary_max_tokens or increasing budget_fraction.",
                node_name,
                policy.primary_model,
                policy.primary_max_tokens,
                min_primary_cost,
                node_allocation,
                policy.budget_fraction * 100,
                self.total_budget,
            )

        min_fallback_cost = _cost_for_tokens(
            _litellm_model_name(policy.fallback_model),
            input_tokens=0,
            output_tokens=policy.fallback_max_tokens,
        )
        if min_fallback_cost > node_allocation:
            logger.warning(
                "register_policy: '%s' fallback model output cost alone "
                "(model=%s, max_tokens=%d) is $%.6f, which already exceeds "
                "its entire node allocation of $%.6f. "
                "This node will raise BudgetExceededError on its first call "
                "unless other nodes leave sufficient global budget. "
                "Consider reducing fallback_max_tokens or increasing budget_fraction.",
                node_name,
                policy.fallback_model,
                policy.fallback_max_tokens,
                min_fallback_cost,
                node_allocation,
            )

    # --------------- per-node budget queries -------------------------------

    def node_allocation(self, node_name: str) -> float:
        """Soft budget allocated to this node (total_budget * fraction)."""
        policy = self._policies.get(node_name)
        if policy is None:
            return 0.0
        return self.total_budget * policy.budget_fraction

    def node_spend(self, node_name: str) -> float:
        """Total amount charged to this node so far."""
        return self._node_spend.get(node_name, 0.0)

    def node_remaining(self, node_name: str) -> float:
        """
        Effective unspent allocation for this node.

        Three terms:
          node_allocation  — the original soft slice (total_budget * fraction)
          node_spend       — what this node has spent so far
          node_adjustment  — this node's proportional share of excess spend
                             by other nodes that borrowed from the global pool

        The adjustment term is what keeps node_remaining values consistent
        with global_remaining after any node exceeds its own slice.
        """
        allocation = self.node_allocation(node_name)
        spend      = self._node_spend.get(node_name, 0.0)
        adjustment = self._node_adjustment.get(node_name, 0.0)
        return max(0.0, allocation - spend - adjustment)

    def node_call_count(self, node_name: str) -> int:
        """Number of model calls made by this node so far."""
        return self._node_call_count.get(node_name, 0)

    # --------------- spend recording ---------------------------------------

    def record_spend(self, record: LLMInvocationCost) -> None:
        """
        Commit a completed invocation to the ledger.

        Steps:
        1. Resolve the cost (fall back to estimate if actual is missing).
        2. Determine how much of the cost fell within the node's own slice
           and how much exceeded it (excess_spend).
        3. Update global and per-node ledgers.
        4. If excess_spend > 0, propagate deductions to peer nodes so their
           node_remaining values stay consistent with global_remaining.

        Using estimated_cost when actual_cost is None keeps the ledger
        conservative — it may slightly over-count but will never under-count
        global spend, which is the safety-critical direction.
        """
        node_name = record.node_name

        cost = record.actual_cost
        if cost is None:
            cost = record.estimated_cost
            logger.warning(
                "budget: could not read actual usage metadata for '%s'; "
                "charging estimated cost $%.6f as a conservative fallback.",
                node_name,
                cost,
            )

        # How much of this spend falls within the node's own slice,
        # and how much exceeds it and draws from the shared global pool.
        spent_so_far       = self._node_spend.get(node_name, 0.0)
        remaining_in_slice = max(0.0, self.node_allocation(node_name) - spent_so_far)
        excess_spend       = max(0.0, cost - remaining_in_slice)

        # Commit to global and per-node ledgers.
        self.actual_spend += cost
        self._node_spend[node_name] = spent_so_far + cost

        # increment the counter of node calls. 
        self._node_call_count[node_name] = self._node_call_count.get(node_name, 0) + 1

        # Propagate excess spend to peers so their node_remaining stays accurate.
        if excess_spend > 0.0:
            self._deduct_excess_from_peers(node_name, excess_spend)

        self.records.append(record)

    def _deduct_excess_from_peers(
        self, spending_node: str, excess_spend: float
    ) -> None:
        """
        Distribute excess_spend as proportional deductions across all peer nodes.

        When a node spends beyond its own slice, it draws from the shared global
        pool. That draw effectively reduces what is available to other nodes.
        This method makes that reduction explicit by adjusting each peer's
        node_remaining downward in proportion to its budget_fraction.

        Only peers that still have positive node_remaining absorb the deduction
        — a node already at zero has nothing left to give. If the deduction
        for a peer would push it below zero, it is clamped to its current
        node_remaining (the unclaimed portion is absorbed by global_remaining,
        which already accounts for the spend via actual_spend).

        Parameters
        ----------
        spending_node : the node that exceeded its slice
        excess_spend  : the amount by which it exceeded its slice
        """
        candidates = {
            name: policy
            for name, policy in self._policies.items()
            if name != spending_node
            and self.node_remaining(name) > 0.0
        }

        if not candidates:
            # All peers are already exhausted. The excess is absorbed by
            # global_remaining only. Log so the state is visible.
            logger.warning(
                "budget: '%s' drew $%.6f from the global pool but all peer "
                "nodes are already exhausted. node_remaining for peers is "
                "already $0.00; global_remaining still reflects the spend.",
                spending_node,
                excess_spend,
            )
            return

        total_candidate_fraction = sum(
            p.budget_fraction for p in candidates.values()
        )

        for name, policy in candidates.items():
            share = policy.budget_fraction / total_candidate_fraction
            deduction = excess_spend * share

            # Clamp so node_remaining cannot go below zero.
            max_deductible = self.node_remaining(name)
            actual_deduction = min(deduction, max_deductible)

            self._node_adjustment[name] = (
                self._node_adjustment.get(name, 0.0) + actual_deduction
            )

            logger.debug(
                "budget: adjusting '%s' by -$%.6f (share=%.1f%%) "
                "due to excess spend by '%s'.",
                name,
                actual_deduction,
                share * 100,
                spending_node,
            )

    # --------------- reporting --------------------------------------------

    def report(self) -> None:
        print("\n── LLM budget report " + "─" * 50)
        for i, r in enumerate(self.records, 1):
            actual = (
                "N/A" if r.actual_cost is None else f"${r.actual_cost:.6f}"
            )
            print(
                f"  {i:>2}. [{r.node_name}] model={r.model} | "
                f"est=${r.estimated_cost:.6f} actual={actual} | "
                f"in={r.actual_input_tokens} out={r.actual_output_tokens}"
            )
        print("─" * 70)
        print(f"  Budget:          ${self.total_budget:.6f}")
        print(f"  Spent:           ${self.actual_spend:.6f}")
        print(f"  Remaining:       ${self.remaining:.6f}")
        print(f"  Unallocated pool:${self.unallocated_pool:.6f}")
        if self._policies:
            print("\n  Per-node breakdown:")
            print(
                f"  {'Node':<22} {'Alloc':>10} {'Spent':>10} "
                f"{'Adjusted':>10} {'Remaining':>10}"
            )
            print("  " + "─" * 66)
            for name in self._policies:
                alloc      = self.node_allocation(name)
                spent      = self.node_spend(name)
                adjustment = self._node_adjustment.get(name, 0.0)
                remaining  = self.node_remaining(name)
                print(
                    f"  {name:<22} ${alloc:>9.6f} ${spent:>9.6f} "
                    f"${adjustment:>9.6f} ${remaining:>9.6f}"
                )
        print("─" * 70)

    def estimation_accuracy_report(self) -> None:
        """
        Show how pessimistic the pre-call estimates were vs actual spend,
        per node.

        pessimism_factor = total_estimated / total_actual

        A factor of 3.0x means estimates were 3x too high — the node was
        consuming node_remaining 3x faster than necessary and may have been
        downgraded to fallback prematurely. The fix is to lower
        primary_max_tokens in the policy to better reflect real usage.

        A factor close to 1.0x means max_tokens is well-calibrated to the
        actual output the model produces for this node's tasks.
        """
        print("\n── Estimation accuracy report " + "─" * 41)
        print(
            f"  {'Node':<22} {'Calls':>5} {'Estimated':>12} "
            f"{'Actual':>12} {'Factor':>8}"
        )
        print("  " + "─" * 63)
        for name in self._policies:
            node_records = [r for r in self.records if r.node_name == name]
            if not node_records:
                print(f"  {name:<22} {'—':>5}")
                continue
            total_estimated = sum(r.estimated_cost for r in node_records)
            total_actual    = sum(
                r.actual_cost if r.actual_cost is not None else r.estimated_cost
                for r in node_records
            )
            factor = total_estimated / total_actual if total_actual > 0 else 0.0
            calibration = (
                "well-calibrated" if factor < 1.5
                else "slightly pessimistic" if factor < 2.5
                else "too pessimistic — consider lowering max_tokens"
            )
            print(
                f"  {name:<22} {len(node_records):>5} "
                f"${total_estimated:>11.6f} ${total_actual:>11.6f} "
                f"{factor:>7.2f}x  ← {calibration}"
            )
        print("─" * 70)

# ---------------------------------------------------------------------------
# ContextVar — one GraphBudget per execution context
# ---------------------------------------------------------------------------

_current_graph_budget: ContextVar[GraphBudget | None] = ContextVar(
    "current_graph_budget", default=None
)


def set_graph_budget(budget: GraphBudget):
    return _current_graph_budget.set(budget)


def reset_graph_budget(token) -> None:
    _current_graph_budget.reset(token)


# ---------------------------------------------------------------------------
# LiteLLM helpers
# ---------------------------------------------------------------------------

def _litellm_model_name(model: str) -> str:
    """LangChain uses 'openai:gpt-4o'; LiteLLM pricing uses 'openai/gpt-4o'."""
    return model.replace(":", "/", 1) if ":" in model else model


def _model_name(model: Any) -> str:
    name = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "_model_name", None)
    )
    if not name:
        raise ValueError(
            f"Cannot determine model name from {type(model).__name__}"
        )
    return _litellm_model_name(str(name))


def _token_count(
    model_name: str, messages: list[Any], tools: list[Any] | None
) -> int:
    """Count input tokens using LiteLLM's tokenizer (includes tool schemas)."""

    kwargs: dict[str, Any] = {"model": model_name, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    return int(litellm.token_counter(**kwargs))


def _cost_for_tokens(
    model_name: str, input_tokens: int, output_tokens: int
) -> float:
    """Compute cost for a given input+output token pair using LiteLLM pricing."""
    import litellm

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model_name,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
    )
    return float(prompt_cost + completion_cost)


def _usage_from_response(
    response: ModelResponse,
) -> tuple[int | None, int | None]:
    """Extract actual input/output token counts from the model response."""
    message = getattr(response, "result", None) or getattr(
        response, "response", None
    )
    usage = getattr(message, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(message, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}

    input_tokens  = usage.get("input_tokens",  usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    return (
        int(input_tokens)  if input_tokens  is not None else None,
        int(output_tokens) if output_tokens is not None else None,
    )


# ---------------------------------------------------------------------------
# Deferred policy registration
# ---------------------------------------------------------------------------
# budget_middleware() is called at import time when agents are instantiated,
# but set_graph_budget() is called in main.py just before the graph runs.
# Policies collected here are flushed into the live GraphBudget on the first
# model invocation.

_PENDING_REGISTRATIONS: list[tuple[str, BudgetPolicy]] = []


def _flush_pending_registrations(budget: GraphBudget) -> None:
    while _PENDING_REGISTRATIONS:
        node_name, policy = _PENDING_REGISTRATIONS.pop()
        budget.register_policy(node_name, policy)


# ---------------------------------------------------------------------------
# Budget middleware
# ---------------------------------------------------------------------------

def budget_middleware(policy: BudgetPolicy, node_name: str):
    """
    Wrap every model call inside a node with budget-aware model selection.

    Runs AFTER format_prompt_middleware so the system prompt is already set
    on the request and is included in the input token count.

    Decision logic:
        1. Count actual input tokens for this invocation (system prompt +
           conversation history + tool schemas).
        2. Compute worst-case cost: actual input tokens + max_tokens as the
           output ceiling (pessimistic — actual output is almost always less).
        3. Compare against node_remaining and global_remaining:
              primary_expected <= node_remaining    → use primary
              primary_expected <= global_remaining  → use fallback (borrowing)
              fallback_expected <= global_remaining → use fallback (last resort)
              otherwise                             → BudgetExceededError
        4. Execute the model call with the selected model.
        5. Record actual spend; propagate excess deductions to peer nodes.
    """

    primary = init_chat_model(model=policy.primary_model, temperature=0).bind(
        parallel_tool_calls=False,
        max_tokens=policy.primary_max_tokens,
    )
    fallback = init_chat_model(model=policy.fallback_model, temperature=0).bind(
        parallel_tool_calls=False,
        max_tokens=policy.fallback_max_tokens,
    )

    # Queue this policy for registration into the GraphBudget on first run.
    _PENDING_REGISTRATIONS.append((node_name, policy))

    @wrap_model_call
    def _(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        budget = _current_graph_budget.get()
        if budget is None:
            raise RuntimeError(
                "budget_middleware requires a GraphBudget. "
                "Call set_graph_budget() before invoking the graph."
            )

        # On the very first call, flush all policies collected at import time
        # into the live budget. This also triggers the compatibility warnings.
        _flush_pending_registrations(budget)

        # ── Assemble the exact message list the model will see ───────────────
        # format_prompt_middleware has already set request.system_prompt, so
        # prepend it here to include it in the token count.
        messages: list[Any] = list(request.messages)
        system_prompt = getattr(request, "system_prompt", None)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        tools = getattr(request, "tools", None)

        # ── Pre-call cost estimation ─────────────────────────────────────────
        def estimate(model_obj: Any, max_tokens: int) -> tuple[str, int, float]:
            name = _model_name(model_obj)
            input_tokens = _token_count(name, messages, tools)
            # Worst-case: actual input tokens + max_tokens as output ceiling.
            cost = _cost_for_tokens(name, input_tokens, max_tokens)
            return name, input_tokens, cost

        primary_name, primary_input_tokens, primary_expected = estimate(
            primary, policy.primary_max_tokens
        )

        # ── Model selection ──────────────────────────────────────────────────
        node_rem   = budget.node_remaining(node_name) # node remined budget
        global_rem = budget.remaining # global remined budget

        # counter of how many time the node is already called at this moment
        call_count = budget.node_call_count(node_name)

        # Early conservative switch: if this node has already made 2+ calls and
        # its remaining slice is below its fair share of global_remaining, switch
        # to fallback proactively rather than waiting until the slice is exhausted.
        # This preserves budget for later calls within the same turn (e.g. a node
        # that follows a search → reason → confirm → book pattern needs headroom
        # for all four calls, not just the first two).
        #
        # Fair share = global_remaining * this node's fraction, which is what the
        # node would receive if the remaining global budget were redistributed now.
        # If node_rem has dropped below that fair share, the node is running ahead
        # of pace and should conserve.

        _fair_share = global_rem * policy.budget_fraction
        _early_switch = call_count >= policy.trend_cntr and node_rem < _fair_share

        if primary_expected <= node_rem and not _early_switch:
            # Happy path: within the node's own weighted slice.
            selected              = primary
            selected_name         = primary_name
            selected_expected     = primary_expected
            selected_input_tokens = primary_input_tokens
            selected_max_tokens   = policy.primary_max_tokens

        else:
            # Either the node has exceeded its slice, or the early conservative
            # switch has triggered. Log which case applies.

            # Compute estimated fallback cost.
            fallback_name, fallback_input_tokens, fallback_expected = estimate(
                fallback, policy.fallback_max_tokens
            )

            if _early_switch and primary_expected <= node_rem:
                logger.info(
                    "budget: '%s' early conservative switch on call #%d "
                    "(node_remaining=$%.6f < fair_share=$%.6f); "
                    "using fallback to preserve headroom for later calls.",
                    node_name, call_count + 1, node_rem, _fair_share,
                )
            
            if primary_expected <= global_rem:
                # Primary fits globally but not within the node's own slice.
                # Downgrade to fallback — the node is borrowing from the shared
                # pool and should spend conservatively.
                logger.info(
                    "budget: '%s' exceeded its slice ($%.6f remaining); "
                    "downgrading to fallback (global remaining $%.6f).",
                    node_name, node_rem, global_rem,
                )

            elif fallback_expected <= global_rem:
                # Primary doesn't fit globally either; fallback is the last
                # affordable option.
                logger.warning(
                    "budget: '%s' can only afford fallback model "
                    "(primary $%.6f exceeds global remaining $%.6f).",
                    node_name, primary_expected, global_rem,
                )
                
            else:
                raise BudgetExceededError(
                    node_name, primary_expected, fallback_expected, global_rem
                )

            selected              = fallback
            selected_name         = fallback_name
            selected_expected     = fallback_expected
            selected_input_tokens = fallback_input_tokens
            selected_max_tokens   = policy.fallback_max_tokens            

        # ── Execute ──────────────────────────────────────────────────────────
        request_for_call = request.override(model=selected)
        response         = handler(request_for_call)

        # ── Post-call: record actual spend ───────────────────────────────────
        actual_input, actual_output = _usage_from_response(response)
        actual_cost: float | None = None
        if actual_input is not None and actual_output is not None:
            actual_cost = _cost_for_tokens(
                selected_name, actual_input, actual_output
            )

        record = LLMInvocationCost(
            node_name             = node_name,
            model                 = selected_name,
            estimated_input_tokens= selected_input_tokens,
            estimated_cost        = selected_expected,
            max_tokens            = selected_max_tokens,
            actual_cost           = actual_cost,
            actual_input_tokens   = actual_input,
            actual_output_tokens  = actual_output,
        )
        # record_spend commits the cost and propagates any excess deductions
        # to peer nodes via _deduct_excess_from_peers().
        budget.record_spend(record)

        return response

    return _