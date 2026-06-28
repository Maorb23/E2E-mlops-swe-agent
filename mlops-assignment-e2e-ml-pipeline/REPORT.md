# Airflow Coding-Agent Evaluation Pipeline

## Architecture overview

The new `evaluate_agent` DAG in `dags/evaluate_agent.py` turns the existing manual mini-swe-agent and SWE-bench commands into a reproducible Airflow workflow:

```text
prepare_run -> run_agent -> run_eval -> summarize_and_log
```

The implementation is intentionally simple. Airflow tasks call the same command style as the scripts in `scripts/` with `subprocess.run(..., check=True)`, then normalize artifacts into a local `runs/<run-id>/` directory. This first iteration does not use `DockerOperator` or remote object storage.

## Setup

From the repository root:

```bash
cp .env.example .env
uv sync
```

For real agent runs, make sure the environment has the required model/API credentials, such as `NEBIUS_API_KEY`, and that Docker is available for SWE-bench evaluation.

## Start Airflow

Use the provided helper:

```bash
./run-airflow-standalone.sh
```

The script sets `AIRFLOW_HOME`, points Airflow at the local `dags/` folder, disables example DAGs, and starts standalone Airflow through `uv tool run apache-airflow standalone`.

## Trigger the DAG

Open the Airflow UI, find the `evaluate_agent` DAG, and trigger it manually. Override DAG params in the trigger dialog when needed.

## DAG params

- `split`: SWE-bench split to run. Default: `test`.
- `subset`: Dataset subset. Default: `verified`.
- `workers`: Parallel workers for mini-swe-agent and SWE-bench. Default: `1`.
- `model`: Model name passed to mini-swe-agent. Default: `nebius/moonshotai/Kimi-K2.6`.
- `task_slice`: SWE-bench slice passed to mini-swe-agent. Default: `0:1` for a cheap first run.
- `run_id`: Optional local run identifier. If empty, the DAG generates a timestamp-based ID.
- `cost_limit`: mini-swe-agent cost limit. Default: `0`.

## Artifact layout

Each run writes:

```text
runs/<run-id>/
  config.json
  run-agent/
    preds.json
    trajectories/
  run-eval/
    logs/
    reports/
  metrics.json
  manifest.json
```

`config.json` records the effective DAG params and derived values. `manifest.json` points to the important files and includes notes if expected files were missing. The DAG also copies SWE-bench default `logs/run_evaluation/...` outputs into `runs/<run-id>/run-eval/logs/` when those logs exist.

## MLflow logging

`summarize_and_log` logs to MLflow experiment `coding-agent-evals`.

If `MLFLOW_TRACKING_URI` is set, that URI is used. Otherwise, MLflow writes to local `./mlruns`. The DAG logs config params, numeric metrics, the artifact root as a tag, and key JSON artifacts.

## Inspect a completed run

Check:

```bash
cat runs/<run-id>/config.json
cat runs/<run-id>/metrics.json
cat runs/<run-id>/manifest.json
ls runs/<run-id>/run-agent/trajectories
ls runs/<run-id>/run-eval/reports
```

Open MLflow against the chosen tracking URI to compare params and metrics across runs.

## Rerun examples

Use a different model:

```json
{
  "model": "nebius/another-model",
  "task_slice": "0:1"
}
```

Run more tasks:

```json
{
  "task_slice": "0:3",
  "workers": 2
}
```

Use a stable run id:

```json
{
  "run_id": "kimi-k2-verified-0-3"
}
```

## First-iteration limits

S3/Object Storage is not implemented in this version. The local run folder, `manifest.json`, and MLflow artifact references make each run reproducible and easy to hand off, but a production version should upload the run directory to durable storage and log that remote URI.

Production-style improvements would also include isolated `DockerOperator` tasks, task timeouts and retries, an Airflow/MLflow `docker compose` deployment, and stricter parsing for additional SWE-bench summary formats.
