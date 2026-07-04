# Dashboard Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface `Amount`/`Time` in the live dashboard table, highlight flagged transactions in red, and show a side-by-side precision/recall/AUC comparison panel for both models.

**Architecture:** Thread `amount`/`time` through the existing `ensemble.combine()` call path (already shared by the scorer service and the API's manual-score endpoint) so every scored-transaction payload carries them. Extend the Streamlit dashboard with a pandas `Styler` for row coloring and a one-time `GET /models` fetch for the comparison panel — both as small, independently unit-testable pure functions with thin Streamlit wiring.

**Tech Stack:** Same as the existing project — Python, pandas, Streamlit, `requests` (newly added), pytest + fakeredis.

## Global Constraints

- No changes to the scoring models, thresholds, or ensemble flagging logic — this only threads existing data through and adds display.
- No persistence of comparison metrics history — the panel reflects whatever `models/metrics.json` currently contains, fetched once per dashboard session.
- No live-updating comparison panel — models don't change mid-session, so a single fetch at startup is sufficient.
- `ensemble.combine()`'s signature change is breaking (two new required params) — every existing call site and test referencing its signature or output shape is updated in the same task, not left inconsistent.

---

### Task 1: Carry `amount`/`time` through `combine()` and its callers

**Files:**
- Modify: `src/fraud_detection/ensemble.py`
- Modify: `src/fraud_detection/scorer_service.py`
- Modify: `src/fraud_detection/api.py`
- Test: `tests/test_ensemble.py`
- Test: `tests/test_scorer_service.py`
- Test: `tests/test_pipeline_integration.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `combine(transaction_id: str, if_score: float, if_flagged: bool, ae_score: float, ae_flagged: bool, amount: float, time: float) -> dict` from `fraud_detection.ensemble`, returning `{"transaction_id": str, "amount": float, "time": float, "isolation_forest": {"score": float, "flagged": bool}, "autoencoder": {"score": float, "flagged": bool}, "ensemble_flagged": bool}`.

- [ ] **Step 1: Update the failing/changed tests first**

Replace `tests/test_ensemble.py` with:
```python
from fraud_detection.ensemble import combine


def test_ensemble_flagged_when_either_model_flags():
    result = combine("tx-1", 0.9, True, 0.1, False, 149.62, 100.0)
    assert result["ensemble_flagged"] is True
    assert result["isolation_forest"] == {"score": 0.9, "flagged": True}
    assert result["autoencoder"] == {"score": 0.1, "flagged": False}
    assert result["amount"] == 149.62
    assert result["time"] == 100.0


def test_ensemble_not_flagged_when_neither_flags():
    result = combine("tx-2", 0.1, False, 0.2, False, 10.0, 200.0)
    assert result["ensemble_flagged"] is False


def test_ensemble_flagged_when_both_flag():
    result = combine("tx-3", 0.9, True, 0.95, True, 5.0, 300.0)
    assert result["ensemble_flagged"] is True
    assert result["transaction_id"] == "tx-3"
```

In `tests/test_scorer_service.py`, update the assertions at the end of `test_read_and_handle_batch_publishes_combined_result` (the `_sample_raw_transaction()` fixture already sets `Amount=10.0, Time=100.0`):
```python
    assert processed == 1
    message = pubsub.get_message(timeout=1)
    scored = json.loads(message["data"])
    assert scored["transaction_id"] == "tx-1"
    assert scored["ensemble_flagged"] is True
    assert scored["amount"] == 10.0
    assert scored["time"] == 100.0
```

In `tests/test_pipeline_integration.py`, update the key-set assertion inside the `for _ in range(3):` loop:
```python
        assert set(scored.keys()) == {
            "transaction_id", "amount", "time", "isolation_forest", "autoencoder", "ensemble_flagged",
        }
```

In `tests/test_api.py`, update `test_score_endpoint_returns_combined_result` (the `_sample_transaction()` fixture already sets `Amount=10.0, Time=100.0`):
```python
def test_score_endpoint_returns_combined_result():
    with TestClient(app) as client:
        app.state.scaler = _StubScaler()
        app.state.if_model = _StubModel(0.9, True)
        app.state.ae_model = _StubModel(0.2, False)

        response = client.post("/score", json=_sample_transaction())

        assert response.status_code == 200
        body = response.json()
        assert body["ensemble_flagged"] is True
        assert body["isolation_forest"]["score"] == 0.9
        assert body["autoencoder"]["score"] == 0.2
        assert body["amount"] == 10.0
        assert body["time"] == 100.0
```

- [ ] **Step 2: Run the updated tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_ensemble.py tests/test_scorer_service.py tests/test_pipeline_integration.py tests/test_api.py -v`
Expected: FAIL — `test_ensemble.py`'s tests fail with a `TypeError` (too many positional arguments to `combine()`); the other three fail on the new `amount`/`time` assertions (`KeyError` or assertion mismatch).

- [ ] **Step 3: Update `combine()`**

In `src/fraud_detection/ensemble.py`, replace the whole file with:
```python
def combine(
    transaction_id: str,
    if_score: float,
    if_flagged: bool,
    ae_score: float,
    ae_flagged: bool,
    amount: float,
    time: float,
) -> dict:
    return {
        "transaction_id": transaction_id,
        "amount": amount,
        "time": time,
        "isolation_forest": {"score": if_score, "flagged": if_flagged},
        "autoencoder": {"score": ae_score, "flagged": ae_flagged},
        "ensemble_flagged": if_flagged or ae_flagged,
    }
```

- [ ] **Step 4: Update `scorer_service.handle_message` to pass `amount`/`time`**

In `src/fraud_detection/scorer_service.py`, replace the `handle_message` function with:
```python
def handle_message(message_id, fields, redis_client, scaler, if_model, ae_model,
                    stream_name, group_name, channel_name):
    try:
        transaction_id = fields["transaction_id"]
        raw_data = json.loads(fields["data"])
        result = score_transaction(raw_data, scaler, if_model, ae_model)
        scored = combine(
            transaction_id,
            result["if_score"], result["if_flagged"],
            result["ae_score"], result["ae_flagged"],
            raw_data["Amount"], raw_data["Time"],
        )
        redis_client.publish(channel_name, json.dumps(scored))
    except Exception:
        logger.exception("Failed to process message %s, skipping", message_id)
    finally:
        redis_client.xack(stream_name, group_name, message_id)
```

- [ ] **Step 5: Update `api.score_endpoint` to pass `amount`/`time`**

In `src/fraud_detection/api.py`, replace the `score_endpoint` function with:
```python
@app.post("/score")
def score_endpoint(transaction: TransactionIn):
    if app.state.scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    raw_data = transaction.model_dump(exclude={"transaction_id"})
    result = score_transaction(raw_data, app.state.scaler, app.state.if_model, app.state.ae_model)
    return combine(
        transaction.transaction_id or "manual",
        result["if_score"], result["if_flagged"],
        result["ae_score"], result["ae_flagged"],
        transaction.Amount, transaction.Time,
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_ensemble.py tests/test_scorer_service.py tests/test_pipeline_integration.py tests/test_api.py -v`
Expected: PASS (10 tests total across the four files)

- [ ] **Step 7: Run the full suite to confirm no other regressions**

Run: `./.venv/bin/pytest tests -v`
Expected: All tests PASS (dashboard's own tests are untouched by this task and still reflect the old `combine()` shape — Task 2 updates them next)

- [ ] **Step 8: Commit**

```bash
git add src/fraud_detection/ensemble.py src/fraud_detection/scorer_service.py \
  src/fraud_detection/api.py tests/test_ensemble.py tests/test_scorer_service.py \
  tests/test_pipeline_integration.py tests/test_api.py
git commit -m "Carry amount/time through combine() and its callers"
```

---

### Task 2: Surface `amount`/`time` in the dashboard's `format_row`

**Files:**
- Modify: `src/fraud_detection/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `combine()`'s output shape from Task 1 — specifically that the scored-transaction dict now has top-level `"amount": float` and `"time": float` keys.
- Produces: `format_row(scored: dict) -> dict` now additionally includes `"amount": float` (rounded to 2 decimals) and `"time": float` (unrounded) in its returned dict, alongside the existing `transaction_id`/`if_score`/`if_flagged`/`ae_score`/`ae_flagged`/`ensemble_flagged` keys.

- [ ] **Step 1: Write the failing test**

Replace `tests/test_dashboard.py`'s existing test with:
```python
from fraud_detection.dashboard import format_row


def test_format_row_flattens_scored_transaction():
    scored = {
        "transaction_id": "tx-1",
        "amount": 149.619,
        "time": 100.0,
        "isolation_forest": {"score": 0.8765, "flagged": True},
        "autoencoder": {"score": 0.1234, "flagged": False},
        "ensemble_flagged": True,
    }
    row = format_row(scored)
    assert row == {
        "transaction_id": "tx-1",
        "amount": 149.62,
        "time": 100.0,
        "if_score": 0.876,
        "if_flagged": True,
        "ae_score": 0.123,
        "ae_flagged": False,
        "ensemble_flagged": True,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dashboard.py -v`
Expected: FAIL — `KeyError: 'amount'` (the input `scored` dict from the test now has `amount`/`time` keys that the old `format_row` ignores, but the *expected* output dict includes them, so the equality assertion fails)

- [ ] **Step 3: Update `format_row`**

In `src/fraud_detection/dashboard.py`, replace the `format_row` function with:
```python
def format_row(scored: dict) -> dict:
    return {
        "transaction_id": scored["transaction_id"],
        "amount": round(scored["amount"], 2),
        "time": scored["time"],
        "if_score": round(scored["isolation_forest"]["score"], 3),
        "if_flagged": scored["isolation_forest"]["flagged"],
        "ae_score": round(scored["autoencoder"]["score"], 3),
        "ae_flagged": scored["autoencoder"]["flagged"],
        "ensemble_flagged": scored["ensemble_flagged"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_dashboard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fraud_detection/dashboard.py tests/test_dashboard.py
git commit -m "Surface amount/time in the dashboard's format_row"
```

---

### Task 3: Red-highlight flagged rows in the live table

**Files:**
- Modify: `src/fraud_detection/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: nothing from other tasks — operates purely on a `pd.Series` representing one rendered row (a dict shaped like `format_row`'s output, with at least an `"ensemble_flagged": bool` key).
- Produces: `highlight_flagged_row(row: pd.Series) -> list[str]` from `fraud_detection.dashboard` — returns `["background-color: #ffcccc"] * len(row)` when `row["ensemble_flagged"]` is `True`, else `[""] * len(row)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_dashboard.py` currently starts with a single line,
`from fraud_detection.dashboard import format_row`, followed by the
`test_format_row_flattens_scored_transaction` test from Task 2. Replace just
that top import line with:
```python
import pandas as pd

from fraud_detection.dashboard import format_row, highlight_flagged_row
```
Leave the existing `test_format_row_flattens_scored_transaction` test
untouched, and append these two new tests at the end of the file:
```python
def test_highlight_flagged_row_returns_red_background_when_flagged():
    row = pd.Series({"transaction_id": "tx-1", "amount": 10.0, "ensemble_flagged": True})
    styles = highlight_flagged_row(row)
    assert styles == ["background-color: #ffcccc"] * len(row)


def test_highlight_flagged_row_returns_empty_when_not_flagged():
    row = pd.Series({"transaction_id": "tx-1", "amount": 10.0, "ensemble_flagged": False})
    styles = highlight_flagged_row(row)
    assert styles == [""] * len(row)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_dashboard.py -v`
Expected: FAIL — `ImportError: cannot import name 'highlight_flagged_row'`

- [ ] **Step 3: Implement `highlight_flagged_row` and wire it into `main()`**

In `src/fraud_detection/dashboard.py`, add `import pandas as pd` to the top imports, add the new function after `format_row`:
```python
def highlight_flagged_row(row: pd.Series) -> list:
    if row["ensemble_flagged"]:
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)
```

Then in `main()`, replace:
```python
            table_placeholder.dataframe(rows)
```
with:
```python
            table_placeholder.dataframe(pd.DataFrame(rows).style.apply(highlight_flagged_row, axis=1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_dashboard.py -v`
Expected: PASS (3 tests: `test_format_row_flattens_scored_transaction`, `test_highlight_flagged_row_returns_red_background_when_flagged`, `test_highlight_flagged_row_returns_empty_when_not_flagged`)

- [ ] **Step 5: Commit**

```bash
git add src/fraud_detection/dashboard.py tests/test_dashboard.py
git commit -m "Highlight flagged transactions in red in the live dashboard table"
```

---

### Task 4: Model-comparison panel

**Files:**
- Modify: `src/fraud_detection/config.py`
- Modify: `requirements.txt`
- Modify: `src/fraud_detection/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: the `GET /models` endpoint's response shape from `fraud_detection.api` (already built) — a dict of the form `{"isolation_forest": {"precision": float, "recall": float, "auc": float, "avg_latency_ms": float}, "autoencoder": {...}}` (the `metrics.json` contents, unchanged by this plan).
- Produces: `API_BASE_URL = "http://localhost:8000"` from `fraud_detection.config`. `build_comparison_panel(metrics: dict) -> dict` from `fraud_detection.dashboard`, returning `{"isolation_forest": {"precision": float, "recall": float, "auc": float}, "autoencoder": {"precision": float, "recall": float, "auc": float}}` with each value rounded to 3 decimals.

- [ ] **Step 1: Add `API_BASE_URL` to config and `requests` to requirements**

In `src/fraud_detection/config.py`, change the last line from:
```python
API_WS_URL = "ws://localhost:8000/stream"
```
to:
```python
API_WS_URL = "ws://localhost:8000/stream"
API_BASE_URL = "http://localhost:8000"
```

In `requirements.txt`, add a new line `requests` (anywhere in the file; alphabetical grouping isn't enforced elsewhere in this file).

- [ ] **Step 2: Install the new dependency**

Run: `./.venv/bin/pip install -r requirements.txt`
Expected: `requests` installs successfully (no errors)

- [ ] **Step 3: Write the failing test for `build_comparison_panel`**

After Task 3, `tests/test_dashboard.py` starts with:
```python
import pandas as pd

from fraud_detection.dashboard import format_row, highlight_flagged_row
```
Replace that second import line with:
```python
from fraud_detection.dashboard import build_comparison_panel, format_row, highlight_flagged_row
```
Leave both existing tests untouched, and append this new test at the end of
the file:
```python
def test_build_comparison_panel_rounds_and_selects_fields():
    metrics = {
        "isolation_forest": {
            "precision": 0.05410628019323672,
            "recall": 0.5185185185185185,
            "auc": 0.9393065239611206,
            "avg_latency_ms": 2.09,
        },
        "autoencoder": {
            "precision": 0.05915178571428571,
            "recall": 0.49074074074074076,
            "auc": 0.9448245368471536,
            "avg_latency_ms": 0.03,
        },
    }
    panel = build_comparison_panel(metrics)
    assert panel == {
        "isolation_forest": {"precision": 0.054, "recall": 0.519, "auc": 0.939},
        "autoencoder": {"precision": 0.059, "recall": 0.491, "auc": 0.945},
    }
```

- [ ] **Step 4: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dashboard.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_comparison_panel'`

- [ ] **Step 5: Implement `build_comparison_panel` and wire the panel into `main()`**

In `src/fraud_detection/dashboard.py`, add `import requests` to the top imports, change the config import line from:
```python
from fraud_detection.config import API_WS_URL
```
to:
```python
from fraud_detection.config import API_BASE_URL, API_WS_URL
```

Add the new function after `highlight_flagged_row`:
```python
def build_comparison_panel(metrics: dict) -> dict:
    panel = {}
    for model_name in ("isolation_forest", "autoencoder"):
        model_metrics = metrics[model_name]
        panel[model_name] = {
            "precision": round(model_metrics["precision"], 3),
            "recall": round(model_metrics["recall"], 3),
            "auc": round(model_metrics["auc"], 3),
        }
    return panel
```

In `main()`, right after `st.title("Live Fraud Detection Feed")`, add:
```python
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        response.raise_for_status()
        panel = build_comparison_panel(response.json())
        st.subheader("Model Comparison")
        columns = st.columns(2)
        for column, model_name in zip(columns, ("isolation_forest", "autoencoder")):
            with column:
                st.write(model_name.replace("_", " ").title())
                st.metric("Precision", panel[model_name]["precision"])
                st.metric("Recall", panel[model_name]["recall"])
                st.metric("AUC", panel[model_name]["auc"])
    except requests.exceptions.RequestException:
        st.warning("Model metrics unavailable — is the API running?")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_dashboard.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Run the full suite to confirm no regressions**

Run: `./.venv/bin/pytest tests -v`
Expected: All tests PASS

- [ ] **Step 8: Manual verification**

With the full pipeline running (redis, `scorer_service`, `uvicorn fraud_detection.api:app`, `replay_producer`), run `streamlit run src/fraud_detection/dashboard.py` and confirm: the comparison panel appears at the top with both models' precision/recall/AUC; the live table below it shows `amount`/`time` columns; any row where `ensemble_flagged` is `True` has a red background. Then stop the API process and reload the dashboard page to confirm the panel degrades to the `st.warning` message instead of crashing, while the live table (once the API is back) still works.

- [ ] **Step 9: Commit**

```bash
git add src/fraud_detection/config.py requirements.txt src/fraud_detection/dashboard.py tests/test_dashboard.py
git commit -m "Add model-comparison panel to the dashboard"
```
