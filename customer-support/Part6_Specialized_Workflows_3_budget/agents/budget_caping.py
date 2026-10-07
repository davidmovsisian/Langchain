"""
What is covered:
- NODES EXECUTED IN PARALLEL — FULL THREAD-SAFETY VIA LOCK + RESERVE/COMMIT/RELEASE. 
  WORKS BOTH FPR SEQUENTIAL AND PARALLEL NODES
- Model downgrade when a node exceeds its slice (primary → fallback)
- Early conservative switch to fallback when a node has made >= trend_cntr calls
  and its remaining slice is below its fair share of global_remaining, preserving
  headroom for later calls within the same agent turn
- Peer adjustment (_node_adjustment) so node_remaining stays consistent with
  global_remaining after any node borrows from the shared pool —
  excess spend is distributed proportionally across peer nodes
- Overflow buffer (overflow_fraction) smooths the BudgetExceededError cliff edge:
  reserve() allows a reservation to succeed if it fits within
  global_remaining + overflow_buffer, preventing hard stops caused by small
  estimation errors or slightly longer-than-usual model responses.
  Model selection logic (primary → fallback decisions) still uses strict
  global_remaining so the system remains conservative during normal operation;
  the overflow buffer is only consulted at the hard-stop boundary.
- Hard stop via BudgetExceededError when spend exceeds
  total_budget * (1 + overflow_fraction)
- Conservative fallback when usage metadata is missing — estimated_cost is
  charged instead of silently recording zero spend
- Startup warnings in register_policy():
    1. Fraction over-subscription — fractions sum > 1.0, nodes compete for
       shared budget sooner than expected
    2. max_tokens incompatibility — output cost alone (input_tokens=0) exceeds
       the node's allocation; fires separately for primary and fallback
- Estimation accuracy report — per-node pessimism_factor showing how much
  pre-call estimates overstated actual spend; flags nodes where
  primary_max_tokens should be reduced to prevent premature fallback
- Per-node call counter tracked in _node_call_count, incremented in commit()
- Reserve/commit/release escrow protocol eliminates TOCTOU race under parallel
  execution; release() on exception unblocks parallel nodes immediately
- Public/unsafe method split prevents deadlock under the shared _lock
- ALL_POLICIES dict replaces _PENDING_REGISTRATIONS list — safe under
  concurrent requests
- report() snapshots all fields under a single lock acquisition; shows
  overflow buffer, effective ceiling, and per-node Reserved column

Parallel-safety design
----------------------
Three problems exist when nodes run concurrently, each requiring a
different fix:

Problem 1 — Check-then-act race in model selection
    Two nodes read the same global_remaining, both decide the primary
    model fits, both call it, together they overspend.
    Fix: reserve/commit/release escrow. Budget is reserved atomically
    before the model call (I/O), committed with actual cost after,
    released on error. All three operations hold _lock for their
    entirety. The model call itself runs outside the lock.

Problem 2 — Mutation races in GraphBudget fields
    Concurrent writes to actual_spend, _node_spend, etc. produce
    corrupted totals.
    Fix: every method that reads-then-writes holds _lock for the
    complete operation. Public read methods (node_remaining, remaining)
    also acquire _lock. Internal methods called from within a held lock
    use _unsafe variants that skip re-acquisition to avoid deadlock.

Problem 3 — _PENDING_REGISTRATIONS global list drained by first thread
    Under concurrency the first thread drains the list; subsequent
    threads find it empty and get no policies registered.
    Fix: replaced with ALL_POLICIES dict (written once at import time,
    read-only thereafter). GraphBudget.register_policy() is called from
    the request handler before the graph runs, not lazily on first call.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from contextvars import ContextVar
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
import litellm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget domain types
# ---------------------------------------------------------------------------

class BudgetExceededError(RuntimeError):
    """
    Raised when neither the primary nor the fallback model fits within
    total_budget * (1 + overflow_fraction).
    """

    def __init__(
        self,
        node_name: str,
        primary_expected: float,
        fallback_expected: float,
        remaining: float,
        overflow_buffer: float,
    ) -> None:
        self.node_name = node_name
        self.primary_expected = primary_expected
        self.fallback_expected = fallback_expected
        self.remaining = remaining
        self.overflow_buffer = overflow_buffer
        super().__init__(
            f"Budget exceeded for '{node_name}': "
            f"primary=${primary_expected:.6f}, "
            f"fallback=${fallback_expected:.6f}, "
            f"global remaining=${remaining:.6f}, "
            f"overflow buffer=${overflow_buffer:.6f}"
        )


@dataclass(frozen=True)
class BudgetPolicy:
    """
    Per-agent budget policy declared at agent-definition time.

    budget_fraction  preferred share of total_budget (soft allocation).
                     Fractions need not sum to 1 — the leftover is
                     accessible to all nodes via the fallback path.
                     Must be in (0, 1].

    primary_max_tokens / fallback_max_tokens
                     Max output token ceilings. Used for worst-case cost
                     estimation before each call. register_policy() warns
                     at startup if output cost alone exceeds the allocation.

    trend_cntr       If the node has already made >= trend_cntr calls AND
                     its remaining slice is below its fair share of
                     global_remaining, switch to fallback early to preserve
                     headroom for later calls within the same agent turn.
    """

    budget_fraction: float
    primary_model: str
    primary_max_tokens: int
    fallback_model: str
    fallback_max_tokens: int
    trend_cntr: int

    def __post_init__(self) -> None:
        if not (0.0 < self.budget_fraction <= 1.0):
            raise ValueError(
                f"budget_fraction must be in (0, 1], got {self.budget_fraction}"
            )
        if self.primary_max_tokens <= 0 or self.fallback_max_tokens <= 0:
            raise ValueError("max_tokens values must be positive integers")
        if self.trend_cntr < 1:
            raise ValueError("trend_cntr must be >= 1")


@dataclass
class LLMInvocationCost:
    """One record written to the ledger after each model call."""

    node_name: str
    model: str
    estimated_input_tokens: int
    estimated_cost: float
    max_tokens: int
    actual_cost: float | None = None
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None


# ---------------------------------------------------------------------------
# GraphBudget — central budget ledger (thread-safe)
# ---------------------------------------------------------------------------

@dataclass
class GraphBudget:
    """
    Tracks global spend and per-node spend for one graph execution.

    Constructor parameters
    ----------------------
    total_budget      Hard nominal budget ceiling in dollars.
    overflow_fraction Fraction of total_budget allowed as an overflow
                      buffer beyond the nominal ceiling, to smooth spikes
                      that would otherwise cause a premature
                      BudgetExceededError due to estimation pessimism.

                      overflow_fraction=0.10 means the system tolerates
                      up to total_budget * 1.10 in actual spend before
                      raising BudgetExceededError.

                      The overflow buffer is ONLY consulted at the hard-
                      stop boundary inside reserve(). Model selection
                      (primary → fallback decisions) always uses strict
                      global_remaining so the system stays conservative
                      during normal operation.

                      Must be in [0, 1). Use 0.0 for strict enforcement.

    Example:
        GraphBudget(total_budget=0.10, overflow_fraction=0.10)
        # nominal ceiling:   $0.10
        # overflow buffer:   $0.01
        # effective ceiling: $0.11  — BudgetExceededError fires above this

    Thread-safety
    -------------
    A single _lock guards all mutable fields. The protocol is:

        reserve()  — acquire lock, check budget (with overflow buffer),
                     escrow expected cost, release lock.
                     Called BEFORE the model I/O call.
        commit()   — acquire lock, release reservation, record actual
                     spend, propagate peer adjustments, release lock.
                     Called AFTER the model I/O call succeeds.
        release()  — acquire lock, release reservation without spending.
                     Called when the model I/O call raises an exception.

    Public read methods (node_remaining, remaining, …) acquire _lock.
    Internal _unsafe variants are called from within a held lock and
    must NOT re-acquire it (would deadlock).

    Weighted budget partitioning + dynamic reallocation
    ---------------------------------------------------
    node_allocation = total_budget * fraction        (soft per-node target)

    Effective node_remaining on each call:

        node_remaining = node_allocation
                         − node_spend        (committed spend by this node)
                         − node_reserved     (escrowed by this node in flight)
                         − node_adjustment   (peer excess propagated to this node)

    When a node spends beyond its slice (excess_spend > 0), the excess is
    distributed to peers via _deduct_excess_from_peers_unsafe(), keeping
    node_remaining values consistent with global_remaining at all times.

    Naming guide
    ------------
    overflow_fraction            fraction of total_budget allowed as buffer.
    overflow_buffer              total_budget * overflow_fraction (dollar amount).
    effective_ceiling            total_budget * (1 + overflow_fraction).
    excess_spend                 portion of a spend that exceeded the node's slice.
    _node_adjustment             cumulative peer deductions received by each node.
    _deduct_excess_from_peers_unsafe()  propagates excess_spend to other nodes.
    _reserved / _node_reserved   global and per-node escrowed amounts.
    """

    total_budget: float
    overflow_fraction: float
    actual_spend: float = field(init=False, default=0.0)
    _reserved: float = field(init=False, default=0.0)
    _node_spend: dict[str, float] = field(init=False, default_factory=dict, repr=False)
    _node_reserved: dict[str, float] = field(init=False, default_factory=dict, repr=False)
    _node_adjustment: dict[str, float] = field(init=False, default_factory=dict, repr=False)
    _node_call_count: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _policies: dict[str, BudgetPolicy] = field(init=False, default_factory=dict, repr=False)
    records: list[LLMInvocationCost] = field(init=False, default_factory=list)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.total_budget <= 0.0:
            raise ValueError(
                f"total_budget must be positive, got {self.total_budget}"
            )
        if not (0.0 <= self.overflow_fraction < 1.0):
            raise ValueError(
                f"overflow_fraction must be in [0, 1), got {self.overflow_fraction}"
            )

    # --------------- overflow buffer properties -----------------------------

    @property
    def overflow_buffer(self) -> float:
        """Dollar amount of the overflow allowance (total_budget * overflow_fraction)."""
        return self.total_budget * self.overflow_fraction

    @property
    def effective_ceiling(self) -> float:
        """Maximum spend before BudgetExceededError (total_budget * (1 + overflow_fraction))."""
        return self.total_budget * (1.0 + self.overflow_fraction)

    # --------------- public derived properties (acquire lock) ---------------

    @property
    def remaining(self) -> float:
        """
        Global unspent and unreserved budget relative to the nominal
        total_budget (strict — does not include the overflow buffer).
        Used for model selection decisions to stay conservative.
        """
        with self._lock:
            return self._remaining_unsafe()

    def node_allocation(self, node_name: str) -> float:
        """Soft budget allocated to this node (total_budget * fraction)."""
        with self._lock:
            return self._node_allocation_unsafe(node_name)

    def node_spend(self, node_name: str) -> float:
        """Total committed spend for this node."""
        with self._lock:
            return self._node_spend.get(node_name, 0.0)

    def node_remaining(self, node_name: str) -> float:
        """
        Effective unspent allocation for this node.

        Accounts for committed spend, in-flight reservations, and peer
        adjustments from nodes that borrowed from the global pool.
        """
        with self._lock:
            return self._node_remaining_unsafe(node_name)

    def node_call_count(self, node_name: str) -> int:
        """Number of completed model calls by this node."""
        with self._lock:
            return self._node_call_count.get(node_name, 0)

    # --------------- internal _unsafe variants (no lock — caller holds it) --

    def _remaining_unsafe(self) -> float:
        """
        Strict remaining budget relative to total_budget.
        Does NOT include the overflow buffer — used for model selection
        so decisions remain conservative during normal operation.
        """
        return max(0.0, self.total_budget - self.actual_spend - self._reserved)

    def _node_allocation_unsafe(self, node_name: str) -> float:
        policy = self._policies.get(node_name)
        if policy is None:
            return 0.0
        return self.total_budget * policy.budget_fraction

    def _node_remaining_unsafe(self, node_name: str) -> float:
        """node_remaining without acquiring the lock."""
        allocation = self._node_allocation_unsafe(node_name)
        spend      = self._node_spend.get(node_name, 0.0)
        reserved   = self._node_reserved.get(node_name, 0.0)
        adjustment = self._node_adjustment.get(node_name, 0.0)
        return max(0.0, allocation - spend - reserved - adjustment)

    # --------------- registration ------------------------------------------

    def register_policy(self, node_name: str, policy: BudgetPolicy) -> None:
        """
        Register a node's BudgetPolicy into the ledger.

        Called from the request handler before the graph runs. Idempotent.

        Two startup warnings:
        1. Fraction over-subscription — fractions sum > 1.0.
        2. max_tokens incompatibility — output cost alone exceeds the
           node's allocation; fires separately for primary and fallback.
           The overflow buffer is not considered here because it is an
           emergency allowance, not a normal allocation source.
        """
        with self._lock:
            if node_name in self._policies:
                return

            total_fraction = (
                sum(p.budget_fraction for p in self._policies.values())
                + policy.budget_fraction
            )
            if total_fraction > 1.0 + 1e-9:
                logger.warning(
                    "register_policy: total budget_fraction is %.2f > 1.0 "
                    "after adding '%s' (fraction=%.2f). "
                    "Nodes will compete for shared budget sooner than expected.",
                    total_fraction, node_name, policy.budget_fraction,
                )

            self._policies[node_name] = policy
            node_allocation = self.total_budget * policy.budget_fraction

            min_primary_cost = _cost_for_tokens(
                _litellm_model_name(policy.primary_model), 0, policy.primary_max_tokens
            )
            if min_primary_cost > node_allocation:
                logger.warning(
                    "register_policy: '%s' primary output cost alone ($%.6f) "
                    "exceeds its allocation ($%.6f, %.0f%% of $%.2f). "
                    "Node will always downgrade to fallback on its first call. "
                    "Consider reducing primary_max_tokens or increasing budget_fraction.",
                    node_name, min_primary_cost, node_allocation,
                    policy.budget_fraction * 100, self.total_budget,
                )

            min_fallback_cost = _cost_for_tokens(
                _litellm_model_name(policy.fallback_model), 0, policy.fallback_max_tokens
            )
            if min_fallback_cost > node_allocation:
                logger.warning(
                    "register_policy: '%s' fallback output cost alone ($%.6f) "
                    "exceeds its allocation ($%.6f). "
                    "Node will raise BudgetExceededError unless peers leave "
                    "sufficient global budget (overflow buffer=$%.6f may help). "
                    "Consider reducing fallback_max_tokens or increasing budget_fraction.",
                    node_name, min_fallback_cost, node_allocation,
                    self.overflow_buffer,
                )

    # --------------- reserve / commit / release (parallel-safe escrow) ------

    def reserve(self, node_name: str, amount: float) -> bool:
        """
        Atomically check and escrow budget before a model call.

        Acquires _lock for the full check-then-escrow so no two parallel
        nodes can claim the same budget window (eliminates TOCTOU race).

        The affordability check uses global_remaining + overflow_buffer.
        This smooths the hard-stop cliff edge: a call that exceeds
        global_remaining by a small amount due to estimation pessimism
        is still allowed, up to the overflow ceiling.

        Model selection (primary → fallback decisions) in budget_middleware
        uses strict global_remaining without the buffer, so the system
        stays conservative during normal operation. The buffer is only
        consulted here, at the last-resort boundary.

        Returns True  — reservation succeeded; caller may proceed.
        Returns False — amount exceeds global_remaining + overflow_buffer;
                        caller should raise BudgetExceededError.
        """
        with self._lock:
            available = self._remaining_unsafe() + self.overflow_buffer
            if amount > available:
                return False
            self._reserved += amount
            self._node_reserved[node_name] = (
                self._node_reserved.get(node_name, 0.0) + amount
            )
            return True

    def commit(self, node_name: str, reserved_amount: float,
               record: LLMInvocationCost) -> None:
        """
        Release the reservation and record actual spend after a successful call.

        Steps (all under _lock):
        1. Resolve cost — use actual_cost; fall back to estimated_cost if missing.
        2. Release the reservation from _reserved and _node_reserved.
        3. Compute excess_spend — how much exceeded the node's own slice.
        4. Update actual_spend, _node_spend, _node_call_count, records.
        5. If excess_spend > 0, propagate peer adjustments.
        """
        with self._lock:
            cost = record.actual_cost
            if cost is None:
                cost = record.estimated_cost
                logger.warning(
                    "budget: no usage metadata for '%s'; "
                    "charging estimated $%.6f as conservative fallback.",
                    node_name, cost,
                )

            # Release reservation.
            self._reserved = max(0.0, self._reserved - reserved_amount)
            self._node_reserved[node_name] = max(
                0.0, self._node_reserved.get(node_name, 0.0) - reserved_amount
            )

            # Compute excess before updating _node_spend.
            spent_so_far       = self._node_spend.get(node_name, 0.0)
            remaining_in_slice = max(
                0.0, self._node_allocation_unsafe(node_name) - spent_so_far
            )
            excess_spend = max(0.0, cost - remaining_in_slice)

            # Commit actual spend.
            self.actual_spend += cost
            self._node_spend[node_name] = spent_so_far + cost
            self._node_call_count[node_name] = (
                self._node_call_count.get(node_name, 0) + 1
            )
            self.records.append(record)

            # Propagate peer adjustments if this node borrowed from the pool.
            if excess_spend > 0.0:
                self._deduct_excess_from_peers_unsafe(node_name, excess_spend)

    def release(self, node_name: str, reserved_amount: float) -> None:
        """
        Release a reservation without spending — called when the model
        call raises an exception.

        Prevents budget from being permanently escrowed after an error,
        which would block other parallel nodes unnecessarily.
        """
        with self._lock:
            self._reserved = max(0.0, self._reserved - reserved_amount)
            self._node_reserved[node_name] = max(
                0.0, self._node_reserved.get(node_name, 0.0) - reserved_amount
            )

    # --------------- peer adjustment (called under _lock) -------------------

    def _deduct_excess_from_peers_unsafe(
        self, spending_node: str, excess_spend: float
    ) -> None:
        """
        Distribute excess_spend as proportional deductions to peer nodes.

        Called from commit() which already holds _lock. Must NOT re-acquire.

        Only peers with positive _node_remaining_unsafe absorb the deduction.
        Each peer's share is proportional to its budget_fraction relative to
        the total fraction of all absorbing peers. Deductions are clamped so
        a peer's node_remaining cannot go below zero.
        """
        candidates = {
            name: policy
            for name, policy in self._policies.items()
            if name != spending_node
            and self._node_remaining_unsafe(name) > 0.0
        }

        if not candidates:
            logger.warning(
                "budget: '%s' drew $%.6f from the global pool but all peers "
                "are exhausted. global_remaining still reflects the spend.",
                spending_node, excess_spend,
            )
            return

        total_candidate_fraction = sum(
            p.budget_fraction for p in candidates.values()
        )

        for name, policy in candidates.items():
            share            = policy.budget_fraction / total_candidate_fraction
            deduction        = excess_spend * share
            max_deductible   = self._node_remaining_unsafe(name)
            actual_deduction = min(deduction, max_deductible)

            self._node_adjustment[name] = (
                self._node_adjustment.get(name, 0.0) + actual_deduction
            )
            logger.debug(
                "budget: adjusting '%s' by -$%.6f (share=%.1f%%) "
                "due to excess spend by '%s'.",
                name, actual_deduction, share * 100, spending_node,
            )

    # --------------- reporting ----------------------------------------------

    def report(self) -> None:
        # Snapshot under lock for a consistent view.
        with self._lock:
            records           = list(self.records)
            total             = self.total_budget
            spent             = self.actual_spend
            reserved          = self._reserved
            remaining         = self._remaining_unsafe()
            overflow_buf      = self.overflow_buffer
            effective_ceil    = self.effective_ceiling
            node_rows         = [
                (
                    name,
                    self._node_allocation_unsafe(name),
                    self._node_spend.get(name, 0.0),
                    self._node_reserved.get(name, 0.0),
                    self._node_adjustment.get(name, 0.0),
                    self._node_remaining_unsafe(name),
                )
                for name in self._policies
            ]

        print("\n── LLM budget report " + "─" * 50)
        for i, r in enumerate(records, 1):
            actual = "N/A" if r.actual_cost is None else f"${r.actual_cost:.6f}"
            print(
                f"  {i:>2}. [{r.node_name}] model={r.model} | "
                f"est=${r.estimated_cost:.6f} actual={actual} | "
                f"in={r.actual_input_tokens} out={r.actual_output_tokens}"
            )
        print("─" * 70)
        print(f"  Budget (nominal):  ${total:.6f}")
        print(f"  Overflow buffer:   ${overflow_buf:.6f}  "
              f"({self.overflow_fraction * 100:.0f}% of budget)")
        print(f"  Effective ceiling: ${effective_ceil:.6f}")
        print(f"  Spent:             ${spent:.6f}")
        print(f"  Reserved (inflight):${reserved:.6f}")
        print(f"  Remaining (strict):${remaining:.6f}")
        if node_rows:
            print("\n  Per-node breakdown:")
            print(
                f"  {'Node':<22} {'Alloc':>10} {'Spent':>10} "
                f"{'Reserved':>10} {'Adjusted':>10} {'Remaining':>10}"
            )
            print("  " + "─" * 76)
            for name, alloc, sp, res, adj, rem in node_rows:
                print(
                    f"  {name:<22} ${alloc:>9.6f} ${sp:>9.6f} "
                    f"${res:>9.6f} ${adj:>9.6f} ${rem:>9.6f}"
                )
        print("─" * 70)

    def estimation_accuracy_report(self) -> None:
        """
        Show how pessimistic pre-call estimates were vs actual spend per node.

        pessimism_factor = total_estimated / total_actual

        > 2.5x  too pessimistic — nodes downgrade to fallback prematurely.
                Lower primary_max_tokens to better match real output length.
        1.0–1.5x well-calibrated.
        """
        with self._lock:
            records  = list(self.records)
            policies = dict(self._policies)

        print("\n── Estimation accuracy report " + "─" * 41)
        print(
            f"  {'Node':<22} {'Calls':>5} {'Estimated':>12} "
            f"{'Actual':>12} {'Factor':>8}"
        )
        print("  " + "─" * 63)
        for name in policies:
            node_records = [r for r in records if r.node_name == name]
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
                "well-calibrated"                                if factor < 1.5 else
                "slightly pessimistic"                           if factor < 2.5 else
                "too pessimistic — consider lowering max_tokens"
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


def set_graph_budget(budget: GraphBudget) -> object:
    return _current_graph_budget.set(budget)


def reset_graph_budget(token: object) -> None:
    _current_graph_budget.reset(token)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ALL_POLICIES — written once at import time, read-only thereafter
# ---------------------------------------------------------------------------
# Replaces the mutable _PENDING_REGISTRATIONS list which was unsafe under
# concurrency (first thread drains the list; subsequent threads get nothing).
#
# Usage in the request handler (or main.py for single-user mode):
#
#     graph_budget = GraphBudget(total_budget=0.10, overflow_fraction=0.10)
#     for node_name, policy in ALL_POLICIES.items():
#         graph_budget.register_policy(node_name, policy)
#     token = set_graph_budget(graph_budget)

ALL_POLICIES: dict[str, BudgetPolicy] = {}


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
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_core.messages import convert_to_openai_messages

    openai_messages = convert_to_openai_messages(messages)

    kwargs: dict[str, Any] = {"model": model_name, "messages": openai_messages}
    if tools:
        kwargs["tools"] = [convert_to_openai_tool(tool) for tool in tools]

    return int(litellm.token_counter(**kwargs))


def _cost_for_tokens(
    model_name: str, input_tokens: int, output_tokens: int
) -> float:
    """Compute cost for a given input+output token pair using LiteLLM pricing."""
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
    result = getattr(response, "result", None)
    message = result[-1] if isinstance(result, list) and result else (result or response)

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
# Budget middleware
# ---------------------------------------------------------------------------

def budget_middleware(policy: BudgetPolicy, node_name: str, parallel_tool_calls: bool = False):
    """
    Wrap every model call inside a node with budget-aware model selection.

    Parallel-safe flow
    ------------------
    1. Assemble messages (outside lock — pure computation).
    2. Estimate cost for primary model (outside lock — pure computation).
    3. Select model + reserve budget atomically (inside lock via reserve()).
       reserve() uses global_remaining + overflow_buffer for the affordability
       check, preventing hard stops from small estimation errors.
    4. Execute the model call (outside lock — I/O, can be long).
    5. On success: commit actual spend (inside lock via commit()).
    6. On exception: release reservation (inside lock via release()).

    Model selection logic
    ---------------------
    Uses strict global_remaining (no overflow buffer) to stay conservative:

    primary_expected <= node_remaining (own slice)  → use primary
    primary_expected <= global_remaining             → use fallback (borrowing)
    fallback_expected <= global_remaining            → use fallback (last resort)
    otherwise → attempt reserve() with overflow buffer:
        reserve() succeeds  → use fallback (overflow absorbed the gap)
        reserve() fails     → BudgetExceededError (above effective ceiling)

    Early conservative switch: if call_count >= trend_cntr AND node_remaining
    is below fair_share (= global_remaining * fraction), switch to fallback
    proactively to preserve headroom for later calls in the same turn.
    """
    primary = init_chat_model(model=policy.primary_model, temperature=0).bind(
        parallel_tool_calls=parallel_tool_calls,
        max_tokens=policy.primary_max_tokens,
    )
    fallback = init_chat_model(model=policy.fallback_model, temperature=0).bind(
        parallel_tool_calls=parallel_tool_calls,
        max_tokens=policy.fallback_max_tokens,
    )

    # Register into ALL_POLICIES at import time (idempotent).
    ALL_POLICIES[node_name] = policy

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

        # ── 1. Assemble message list ─────────────────────────────────────────
        messages: list[Any] = list(request.messages)
        system_prompt = getattr(request, "system_prompt", None) or getattr(request, "system_message", None)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        tools = getattr(request, "tools", None)

        # ── 2. Pre-call cost estimation (outside lock) ───────────────────────
        def estimate(model_obj: Any, max_tokens: int) -> tuple[str, int, float]:
            name         = _model_name(model_obj)
            input_tokens = _token_count(name, messages, tools)
            cost         = _cost_for_tokens(name, input_tokens, max_tokens)
            return name, input_tokens, cost

        primary_name, primary_input_tokens, primary_expected = estimate(
            primary, policy.primary_max_tokens
        )

        # ── 3. Model selection + atomic reservation ──────────────────────────
        # Snapshots for selection decisions — uses strict remaining (no buffer)
        # so the system stays conservative during normal operation.
        node_rem   = budget.node_remaining(node_name)
        global_rem = budget.remaining
        call_count = budget.node_call_count(node_name)

        _fair_share   = global_rem * policy.budget_fraction
        _early_switch = call_count >= policy.trend_cntr and node_rem < _fair_share

        if primary_expected <= node_rem and not _early_switch:
            # Happy path: within the node's own weighted slice.
            # reserve() confirms atomically that budget still exists.
            if budget.reserve(node_name, primary_expected):
                selected              = primary
                selected_name         = primary_name
                selected_expected     = primary_expected
                selected_input_tokens = primary_input_tokens
                selected_max_tokens   = policy.primary_max_tokens
            else:
                # Budget moved between snapshot and reserve — fall through.
                primary_expected = float("inf")
                selected         = None
        else:
            selected = None

        if selected is None:
            # Fallback path: slice exhausted, early switch, or reserve() raced.
            fallback_name, fallback_input_tokens, fallback_expected = estimate(
                fallback, policy.fallback_max_tokens
            )

            if _early_switch and primary_expected <= node_rem:
                logger.info(
                    "budget: '%s' early conservative switch on call #%d "
                    "(node_remaining=$%.6f < fair_share=$%.6f).",
                    node_name, call_count + 1, node_rem, _fair_share,
                )
            elif primary_expected <= global_rem:
                logger.info(
                    "budget: '%s' exceeded its slice ($%.6f remaining); "
                    "downgrading to fallback (global remaining $%.6f).",
                    node_name, node_rem, global_rem,
                )
            elif fallback_expected <= global_rem:
                logger.warning(
                    "budget: '%s' can only afford fallback "
                    "(primary $%.6f exceeds global remaining $%.6f).",
                    node_name, primary_expected, global_rem,
                )
            else:
                # Neither model fits within strict global_remaining.
                # reserve() still tries with the overflow buffer — a small
                # estimation overshoot may be absorbed without a hard stop.
                logger.warning(
                    "budget: '%s' fallback $%.6f exceeds strict remaining "
                    "$%.6f; attempting overflow buffer $%.6f.",
                    node_name, fallback_expected, global_rem,
                    budget.overflow_buffer,
                )

            # Atomically reserve — overflow buffer applied inside reserve().
            if not budget.reserve(node_name, fallback_expected):
                raise BudgetExceededError(
                    node_name,
                    primary_expected,
                    fallback_expected,
                    budget.remaining,
                    budget.overflow_buffer,
                )

            selected              = fallback
            selected_name         = fallback_name
            selected_expected     = fallback_expected
            selected_input_tokens = fallback_input_tokens
            selected_max_tokens   = policy.fallback_max_tokens

        # ── 4. Model call (outside lock — I/O) ──────────────────────────────
        request_for_call = request.override(model=selected)
        try:
            response = handler(request_for_call)
        except Exception:
            budget.release(node_name, selected_expected)
            raise

        # ── 5. Commit actual spend ───────────────────────────────────────────
        actual_input, actual_output = _usage_from_response(response)
        actual_cost: float | None = None
        if actual_input is not None and actual_output is not None:
            actual_cost = _cost_for_tokens(
                selected_name, actual_input, actual_output
            )

        record = LLMInvocationCost(
            node_name              = node_name,
            model                  = selected_name,
            estimated_input_tokens = selected_input_tokens,
            estimated_cost         = selected_expected,
            max_tokens             = selected_max_tokens,
            actual_cost            = actual_cost,
            actual_input_tokens    = actual_input,
            actual_output_tokens   = actual_output,
        )
        budget.commit(node_name, selected_expected, record)

        return response

    return _