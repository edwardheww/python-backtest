# NQ / GC Opening-Range Session Calibration

A research tool that rebuilds an opening-range trading system on historical futures bars (NQ and GC), searches its parameters on a **training period only**, and then decides which trading sessions to keep or filter out, confirming the decision on a **held-out test period**.

## How it works

1. **Rebuild the logic on history.** Three sessions (Asia, London, New York), each with its own opening range, an expansion-block rule, regime filters, an entry strategy, and simulated trade management (stop, target, time stop, trailing).
2. **Search parameters on training data.** Random search over a parameter space (`--samples`, default 150), one calibration per instrument, run in parallel with `--jobs`. The best set is chosen by training t-statistic.
3. **Judge each session from training evidence only.** A session is `KEEP` only if its training expectancy is positive, the median across parameter sets is positive, and at least 55% of parameter sets are positive in that session.
4. **Confirm on the test period.** The test period only confirms or rejects. A session that passes training but fails or lacks test data becomes `WATCH (paper trade only)`.

Guards against fooling yourself, built into the output:

- **Luck benchmark:** prints the t-statistic the best of N no-edge parameter sets would typically reach (`sqrt(2 ln N)`), and warns if the chosen set does not beat it.
- **Bootstrap confidence interval** (90%) on test-period expectancy. A session is flagged `KEEP (weak evidence)` if the interval includes zero.
- **Breadth:** the share of all parameter sets that are positive in each session, in train and test, so a result that depends on one lucky setting is visible.
- **Multiple-testing note:** the summary reminds you that several instrument/session cells plus a parameter search guarantee some false positives by chance.
- **1-minute tie resolution:** when stop and target both fall inside one bar, 1-minute data decides which was hit first. Without it, ties assume the stop was hit first (the conservative choice).

## Included result

`cal_session_filter.csv` is the output of one NQ run:

| Instrument | Session | Verdict | Train trades | Train exp (R) | Train t | Test trades | Test exp (R) | Sets positive (train / test) |
|---|---|---|---|---|---|---|---|---|
| NQ | NY | KEEP | 123 | +0.256 | 1.98 | 46 | +0.117 | 78% / 62% |

How to read it: the New York session passed the tool's own rules and was confirmed on the test period, but the evidence is modest (training t-statistic about 2, 46 test trades). Treat it as a candidate for forward testing, not proof of an edge. Results are in R-multiples (profit divided by initial risk) and include a placeholder cost of 1.0 point round trip for NQ.

## Setup

Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

**No market data is included.** Historical data from Databento and Yahoo Finance is licensed to the account holder and may not be redistributed, so you generate it yourself.

### Option A: free smoke test (about a minute)

Pulls roughly 59 days of 5-minute bars from Yahoo Finance. Too little data for real conclusions, but enough to see the pipeline run end to end.

```bash
python3 download_test_data.py
python3 calibrate_sessions.py --nq nq_5m.csv --gc gc_5m.csv run --preset balanced
```

### Option B: proper history from Databento (paid)

```bash
export DATABENTO_API_KEY="your-key"

# Step 1: price the request. Nothing is downloaded or charged.
python3 download_databento.py --symbols NQ --days 1825

# Step 2: download once you are happy with the cost.
python3 download_databento.py --symbols NQ --days 1825 --yes

# Step 3: calibrate
python3 calibrate_sessions.py --nq nq_1m.csv calibrate --samples 150 --jobs 4
```

Databento's continuous contracts are unadjusted, so the downloader back-adjusts them at each roll (difference method). Adjusted prices drift from real traded prices in older data. That is fine for ratios and R-multiples, but old price levels are not real quotes.

## Usage reference

```
python3 calibrate_sessions.py [run|calibrate] --nq FILE [--gc FILE] [options]
```

| Mode | What it does |
|---|---|
| `run` | Evaluates a single parameter set (default or `--preset`) and reports each session on the train and test periods. |
| `calibrate` | Random parameter search on the training period, then session verdicts confirmed on the test period. |

| Option | Meaning |
|---|---|
| `--preset strict\|balanced\|loose` | Named starting parameter set |
| `--set key=value` | Override one parameter (repeatable) |
| `--split YYYY-MM-DD` | Train/test boundary (default: 70% of history) |
| `--sessions NY` or `London,NY` | Analyse only these sessions |
| `--samples N`, `--seed N`, `--jobs N` | Search size, reproducibility, parallel workers |
| `--min-trades N`, `--min-session-trades N` | Minimum trade counts for choosing a set / giving a verdict |
| `--config sessions.json` | Override session windows or costs per instrument |
| `--no-1m` | Ignore 1-minute data (saves memory; ties assume stop first) |
| `--debug-date YYYY-MM-DD` | Print one session's features and trades, then exit |
| `--out-prefix NAME` | Prefix for output files (default `cal`) |

Input files: 1-minute (preferred) or 5-minute OHLCV, CSV or Parquet, including overnight bars. Naive timestamps are assumed to be America/New_York (`--tz` to change).

### Other scripts

- `calibrate_regimes.py`: explores how regime thresholds split your own 5-minute RTH data and shows a funnel of which condition blocks each regime. Writes `daily_metrics.csv`.
  ```bash
  python3 calibrate_regimes.py --csv my.csv    # columns: Datetime,Open,High,Low,Close,Volume
  ```
- `download_test_data.py`: Yahoo Finance smoke-test data (see above).
- `download_databento.py`: Databento downloader with a cost check, chunked caching, and roll back-adjustment.

## Outputs

| File | Contents |
|---|---|
| `cal_session_filter.csv` | Session verdicts, train/test statistics, breadth (included as an example) |
| `cal_<SYMBOL>_trades.csv` | Trade log from the chosen parameter set, including entry prices (derived from licensed data, so not committed) |
| `daily_metrics.csv` | Daily metrics from `calibrate_regimes.py`, including daily prices (not committed) |

## Limitations

- **Every threshold is a placeholder.** Defaults are starting points, not tuned values.
- **Cost model is simple:** a fixed round-trip slippage and fee in points per instrument (NQ 1.0, GC 0.3). Edit it with `--config` before trusting any result.
- **Small samples.** Per-session test periods can contain few trades, so confidence intervals are wide.
- **Parameter search plus several cells means false positives are expected.** A `KEEP` is a hypothesis to test forward, not a finding.
- **Yahoo data covers about 59 days of 5-minute bars.** Use it only as a smoke test.
- **Backtest only.** No execution, order handling, or live trading. Nothing here is financial advice.

## Files

```
calibrate_sessions.py    Session calibration and train/test verdicts (main tool)
calibrate_regimes.py     Regime threshold explorer
download_databento.py    Paid data downloader (cost check first)
download_test_data.py    Free Yahoo smoke-test data
cal_session_filter.csv   Example output
requirements.txt
```
