# Dashboard Enhancements — Design

## Purpose

The original dashboard shows only per-model scores and flags in a live rolling
table. Two real gaps: the underlying transaction's `Amount`/`Time` are computed
during scoring but discarded before reaching the dashboard, and the model
comparison metrics already computed by `train.py` (precision/recall/AUC) are
served by `GET /models` but never displayed anywhere. This is a small,
self-contained enhancement to the existing dashboard, not a new component.

## 1. Carry `Amount`/`Time` through the scoring pipeline

`fraud_detection.ensemble.combine()` is the single shared function both
`scorer_service.handle_message` and `api.score_endpoint` call to build the
scored-transaction payload. Its signature changes from:

```python
combine(transaction_id, if_score, if_flagged, ae_score, ae_flagged) -> dict
```

to:

```python
combine(transaction_id, if_score, if_flagged, ae_score, ae_flagged, amount, time) -> dict
```

returning:

```json
{
  "transaction_id": "...",
  "amount": 149.62,
  "time": 100.0,
  "isolation_forest": {"score": 0.87, "flagged": true},
  "autoencoder": {"score": 0.91, "flagged": true},
  "ensemble_flagged": true
}
```

Both call sites already have this data available:
- `scorer_service.handle_message` has the original `raw_data` dict (parsed from
  the stream message) before scoring; it passes `raw_data["Amount"]` and
  `raw_data["Time"]` into `combine()`.
- `api.score_endpoint` has `transaction.Amount` and `transaction.Time` directly
  from the validated `TransactionIn` request body.

This is a breaking change to `combine()`'s signature (two new required
positional params) — all existing call sites and tests referencing its
signature or output shape are updated in the same change, not left on an old
shape.

## 2. Red-highlighted flagged rows

`dashboard.py`'s live loop currently builds `rows` as a plain list of dicts and
renders it via `st.dataframe(rows)`. This changes to build a real
`pd.DataFrame(rows)` and render it through a pandas `Styler`:

```python
st.dataframe(pd.DataFrame(rows).style.apply(highlight_flagged_row, axis=1))
```

`highlight_flagged_row(row: pd.Series) -> list[str]` is a standalone,
unit-testable function: returns a light red background
(`background-color: #ffcccc`) for every cell in the row when
`row["ensemble_flagged"]` is `True`, and empty strings otherwise. It has no
Streamlit dependency and is tested directly with constructed `pd.Series`
inputs.

## 3. Model-comparison panel

At the top of the dashboard, above the live table, a one-time (not
per-refresh) panel shows both models' precision/recall/AUC side by side.

Flow in `main()`:
1. Before entering the live scoring loop, call `GET {API_BASE_URL}/models`
   (a new `API_BASE_URL = "http://localhost:8000"` constant added next to the
   existing `API_WS_URL` in `config.py`) via the `requests` library (added to
   `requirements.txt`).
2. On success, pass the parsed JSON into `build_comparison_panel(metrics: dict) -> dict`,
   a pure function that reshapes `metrics.json`'s `{"isolation_forest": {...},
   "autoencoder": {...}}` structure into a display-ready form: one entry per
   model with rounded precision/recall/AUC values, keyed the same way as the
   input so the render step stays a thin loop. Unit-tested directly with
   sample `metrics.json`-shaped dicts — no network or Streamlit dependency.
3. Render: `st.columns(2)`, one per model, each showing precision/recall/AUC
   via `st.metric`.
4. On request failure (e.g. `GET /models` returns non-200, or a connection
   error because the API isn't up yet), render `st.warning("Model metrics
   unavailable — is the API running?")` instead of the panel, and continue
   into the live loop as normal (a missing comparison panel must never block
   the live feed from starting).

## Testing

- `tests/test_ensemble.py`: existing tests updated to pass/assert `amount`
  and `time` through `combine()`.
- `tests/test_scorer_service.py`, `tests/test_pipeline_integration.py`,
  `tests/test_api.py`: updated wherever they assert `combine()`'s output shape
  or call `handle_message`/`score_endpoint`, to account for the two new
  fields.
- `tests/test_dashboard.py`: `format_row` test updated to include
  `amount`/`time` in both input and expected output. New tests added for
  `highlight_flagged_row` (flagged row gets the red background list; a
  non-flagged row gets empty strings) and `build_comparison_panel` (given a
  sample metrics dict, returns the expected reshaped/rounded structure).
- No test exercises `main()` itself (Streamlit + live network loop) — verified
  manually by running the dashboard against the live pipeline, same as the
  original build's Task 10.

## Non-Goals

- No changes to the scoring models, thresholds, or ensemble flagging logic —
  this only threads existing data through and adds display.
- No persistence of comparison metrics history — the panel reflects whatever
  `models/metrics.json` currently contains, fetched once per dashboard
  session.
- No live-updating comparison panel — models don't change mid-session, so a
  single fetch at startup is sufficient.
