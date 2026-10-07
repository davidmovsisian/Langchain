# Multi-Agent Travel Assistant with Thread-Safe Budget Capping

A multi-agent travel booking application built with **LangGraph**, **LangChain**, and **LiteLLM**, with a **FastAPI** server and a small **React** web UI. A centralized, thread-safe budget ledger (`GraphBudget`) estimates and caps LLM spend per conversation, downgrades to cheaper models when budget tightens, and a human-in-the-loop layer requires approval for sensitive actions such as bookings and cancellations.

---

## Key Architecture & Features

* **Multi-agent graph**
  * A **primary agent** answers general questions (flight lookup, company policy, web search) and hands off to four **specialists**: flights, hotels, car rental, and excursions.
  * Specialists are created by one factory, `build_specialist(...)`, so they share the same middleware stack, state schema, and budget wiring.
  * Handoff details (location, dates, request) travel in the shared `TravelState.handoff_data` and are injected into each specialist's prompt, together with the passenger's flight information (`user_info`). `handoff_data` is cleared when a specialist returns control.
* **Centralized thread-safe ledger (`GraphBudget`)**
  * Atomic **reserve / commit / release** protocol under a single lock. The model call itself runs outside the lock, and `release()` frees the reservation if a call fails.
  * Global and per-node spend are tracked in real time, using the actual token usage reported by each response (falling back to the estimate if usage is missing).
* **Dynamic budget partitioning**
  * Each node gets a soft slice via `budget_fraction`; the fractions across all agents should sum to 1.0.
  * When a node overspends its slice, the excess is deducted proportionally from its peers (`_node_adjustment`) so per-node remainders stay consistent with the global pool.
* **Model downgrading and early switching**
  * Falls back from the primary model (e.g. `openai:gpt-4o`) to a cheaper one (`openai:gpt-4o-mini`) when a call no longer fits the node's slice.
  * `trend_cntr` triggers an early, proactive switch after that many calls in a node when its remaining slice is below its fair share.
  * Every call is also capped by `primary_max_tokens` / `fallback_max_tokens`.
* **Overflow buffer and hard stop**
  * `overflow_fraction` adds an elastic allowance above the nominal budget to absorb small estimation errors.
  * `BudgetExceededError` is raised only when even the cheaper model cannot fit under the effective ceiling.
* **Human-in-the-loop approvals**
  * Sensitive tools (book / update / cancel) pause the graph with `interrupt()`. Several approvals can be pending at once, and each is answered by its interrupt id. The user approves, or rejects with feedback the agent takes into account.
* **Context management**
  * Pruning of stale read-only tool traffic, LLM summarization of long histories (primary agent), and request-time clearing of old search results (specialists).
* **Web API and UI**
  * A FastAPI server with per-session budgets, plus a single-file React UI with a chat, approval cards, and a live budget panel.

---

## Project Structure

```text
├── agents/
│   ├── __init__.py
│   ├── budget_caping.py          # Ledger, escrow protocol, cost estimation, budget middleware
│   ├── common.py                 # TravelState, prompt/approval middleware, context management, complete_or_escalate
│   ├── specialist.py             # build_specialist(...) factory for the four sub-agents
│   ├── primary_assistant.py      # Primary agent, handoff tools, context-management setup
│   ├── flight_assistant.py       # Flight specialist (tools, prompt, budget policy)
│   ├── hotel_assistant.py        # Hotel specialist
│   ├── car_rental_assistant.py   # Car rental specialist
│   ├── excursion_assistant.py    # Excursion / trip recommendation specialist
│   └── graph.py                  # StateGraph compilation (InMemorySaver checkpointer)
├── tools/                        # Domain tool definitions (flights, hotels, cars, excursions, policies)
├── utils/                        # Utilities and event printer
├── ui/
│   └── index.html                # Single-file React UI, served by the API at "/"
├── api.py                        # FastAPI server (budget setup, chat, approvals, static UI)
└── main.py                       # CLI entry point running the scripted tutorial conversation
```

All modules use relative imports inside the project package, so run them as modules (`python -m ...`, `uvicorn package.module:app`) from the directory that **contains** the package, not by executing the files directly.

---

## Setup & Installation

1. Use Python 3.10+ (the code uses `X | None` type syntax).
2. Install dependencies:
   ```bash
   pip install langchain langgraph litellm langchain-core langchain-community langchain-openai fastapi uvicorn
   ```
3. Set your API keys:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export TAVILY_API_KEY="your-tavily-api-key"
   ```

---

## Running the Application

Replace `<package>` below with the name of the project folder (for example `Part6_Specialized_Workflows_3_budget`), and run from its parent directory.

### Option 1: Web app (API + UI)

```bash
uvicorn <package>.api:app --workers 1
```

Open <http://localhost:8000/> for the UI, or <http://localhost:8000/docs> for the interactive API docs.

> **Use exactly one worker.** Sessions, budgets and the `InMemorySaver` checkpointer live inside the server process, so multiple workers would not see each other's sessions. Sessions are lost on restart, and idle ones expire after one hour (at most 1000 are kept).

### Option 2: CLI tutorial script

```bash
python -m <package>.main
```

The script steps through a scripted conversation (flight info, hotel, car, excursions). Whenever a sensitive action is attempted it pauses and asks for approval (`y` to approve, or type feedback to reject). When it finishes it prints the budget ledger report and an estimation-accuracy breakdown. In a non-interactive run (no stdin) approvals default to *approve*, so don't run it unattended against real data.

---

## API Reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/budget` | Set up budget capping and create a session |
| `GET` | `/budget/{session_id}` | Current spend / remaining budget, overall and per agent |
| `POST` | `/chat/{session_id}` | Send a message, or answer pending approvals |
| `DELETE` | `/sessions/{session_id}` | Discard a session |

### 1. Create a session with a budget

```http
POST /budget
{"total_budget": 0.10, "overflow_fraction": 0.10, "passenger_id": "3442 587242"}
```

`total_budget` is in USD (default 0.10, must be > 0). `overflow_fraction` is in `[0, 1)` (default 0.10; use 0 for a strict cap). The response contains the `session_id` (also used as the LangGraph `thread_id`) and the initial per-agent budget status.

### 2. Chat

```http
POST /chat/{session_id}
{"message": "Hi there, what time is my flight?"}
```

```json
{"status": "completed", "reply": "...", "pending_approvals": [], "budget": {"...": "..."}}
```

If a sensitive tool needs approval, the response has `"status": "needs_approval"` and a list of `pending_approvals` (`id`, `action`, `args`). Answer **all** of them in one request:

```http
POST /chat/{session_id}
{"approvals": {"<id>": {"approved": true}, "<id2>": {"approved": false, "reason": "pick a cheaper one"}}}
```

Send exactly one of `message` or `approvals` per request.

### Status codes

| Code | Meaning |
|---|---|
| `402` | Budget exhausted. The body includes the node and the final budget status. |
| `404` | Unknown or expired session. |
| `409` | Another request for the session is still running, a new message was sent while approvals are pending, or approvals were sent with nothing pending. |
| `422` | Invalid body, or approval ids that don't match the pending ids exactly. |
| `503` | Too many active sessions. |

---

## Configuring Budgets

Each agent has a `BudgetPolicy` in its module:

| Field | Meaning |
|---|---|
| `budget_fraction` | Share of the total budget (fractions across all agents should sum to 1.0). Defaults: primary 0.40, each specialist 0.15. |
| `primary_model` / `fallback_model` | Preferred and cheaper model (`provider:model`). |
| `primary_max_tokens` / `fallback_max_tokens` | Output-token cap per call; it also drives the worst-case cost estimate. |
| `trend_cntr` | Number of calls after which the node switches early to the fallback model if it is below its fair share (must be >= 1). |

Tuning tip: if the primary agent runs out of budget first, lower its `primary_max_tokens` or `trend_cntr` before raising `total_budget`.

---

## Known Limitations

* The `SummarizationMiddleware` makes its own LLM call, which is **not yet counted** against the budget.
* Sessions and checkpoints are in memory only (single process, lost on restart). A durable checkpointer plus a persisted budget ledger would be needed to scale out.
* The API has no authentication; anyone with a `session_id` can use that session.
* Replies are returned when a turn completes; there is no token streaming.
