import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.python import get_current_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
LOCAL_SWE_BENCH_CONFIG = (
    PROJECT_ROOT
    / "mini-swe-agent"
    / "src"
    / "minisweagent"
    / "config"
    / "benchmarks"
    / "swebench.yaml"
)
PACKAGE_SWE_BENCH_CONFIG = "benchmarks/swebench.yaml"
DEFAULT_MODEL = "nebius/moonshotai/Kimi-K2.6"
EXPERIMENT_NAME = "coding-agent-evals"

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_run_id(value: str | None) -> str:
    raw = value.strip() if value else f"eval-{utc_now().strftime('%Y%m%d-%H%M%S')}"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    return sanitized or f"eval-{utc_now().strftime('%Y%m%d-%H%M%S')}"


def coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc


def coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc


def dataset_name_for_subset(subset: str) -> str:
    if subset.lower() == "verified":
        return "princeton-nlp/SWE-bench_Verified"
    return "princeton-nlp/SWE-bench"


def swebench_config_spec() -> str:
    if LOCAL_SWE_BENCH_CONFIG.exists():
        return str(LOCAL_SWE_BENCH_CONFIG)
    return PACKAGE_SWE_BENCH_CONFIG


def build_manifest(config: dict[str, Any], notes: list[str] | None = None) -> dict[str, Any]:
    run_dir = Path(config["run_dir"])
    run_agent_dir = run_dir / "run-agent"
    run_eval_dir = run_dir / "run-eval"
    metrics_path = run_dir / "metrics.json"
    preds_path = run_agent_dir / "preds.json"
    trajectories_dir = run_agent_dir / "trajectories"

    missing_notes = list(notes or [])
    expected_paths = {
        "config_path": run_dir / "config.json",
        "preds_path": preds_path,
        "trajectories_dir": trajectories_dir,
        "eval_dir": run_eval_dir,
        "metrics_path": metrics_path,
        "artifact_root": run_dir,
    }
    for label, path in expected_paths.items():
        if not path.exists():
            missing_notes.append(f"Expected {label} is missing: {path}")

    return {
        "run_id": config["run_id"],
        "created_at": config["created_at"],
        "config_path": str(expected_paths["config_path"]),
        "preds_path": str(preds_path),
        "trajectories_dir": str(trajectories_dir),
        "eval_dir": str(run_eval_dir),
        "metrics_path": str(metrics_path),
        "artifact_root": str(run_dir),
        "notes": missing_notes,
    }


def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, **(env or {})},
        check=True,
    )


def normalize_predictions(run_agent_dir: Path, trajectories_dir: Path, model: str, split: str) -> Path:
    target = run_agent_dir / "preds.json"
    if target.exists():
        return target

    model_file = f"{model.replace('/', '__')}.{split}.json"
    candidates = [
        trajectories_dir / "preds.json",
        run_agent_dir / model_file,
        trajectories_dir / model_file,
    ]
    candidates.extend(run_agent_dir.rglob("preds.json"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            if candidate.resolve() != target.resolve():
                shutil.copy2(candidate, target)
            return target

    inspected = ", ".join(str(path) for path in candidates[:6])
    raise FileNotFoundError(
        f"Could not find SWE-bench predictions to normalize to {target}. "
        f"Inspected: {inspected}"
    )


def copy_eval_outputs(run_eval_dir: Path) -> list[str]:
    notes: list[str] = []
    default_logs_dir = PROJECT_ROOT / "logs" / "run_evaluation"
    logs_target = run_eval_dir / "logs"
    reports_target = run_eval_dir / "reports"
    logs_target.mkdir(parents=True, exist_ok=True)
    reports_target.mkdir(parents=True, exist_ok=True)

    if default_logs_dir.exists():
        shutil.copytree(default_logs_dir, logs_target / "run_evaluation", dirs_exist_ok=True)
    else:
        notes.append(f"SWE-bench default logs directory was not found: {default_logs_dir}")

    for index, report_path in enumerate(logs_target.rglob("report.json"), start=1):
        instance = report_path.parent.name or f"report-{index}"
        shutil.copy2(report_path, reports_target / f"{instance}.report.json")

    if not any(reports_target.glob("*.json")):
        notes.append("No report.json files were copied into run-eval/reports.")

    return notes


def collect_metrics(eval_dir: Path, run_id: str) -> dict[str, Any]:
    notes: list[str] = []
    report_paths = sorted(eval_dir.rglob("report.json")) + sorted((eval_dir / "reports").glob("*.json"))
    seen_instances: set[str] = set()
    resolved_instances: set[str] = set()

    for report_path in report_paths:
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            notes.append(f"Could not parse {report_path}: {exc}")
            continue

        if isinstance(data, dict):
            for instance_id, payload in data.items():
                if isinstance(payload, dict) and "resolved" in payload:
                    seen_instances.add(str(instance_id))
                    if payload.get("resolved") is True:
                        resolved_instances.add(str(instance_id))

    total = len(seen_instances)
    resolved = len(resolved_instances)
    unresolved = total - resolved
    inspected = [str(path) for path in report_paths]

    if total == 0:
        return {
            "run_id": run_id,
            "total_instances": 0,
            "resolved_instances": 0,
            "unresolved_instances": 0,
            "resolve_rate": 0.0,
            "status": "completed_but_metrics_not_found",
            "notes": notes + [f"Inspected files: {inspected or 'none'}"],
        }

    return {
        "run_id": run_id,
        "total_instances": total,
        "resolved_instances": resolved,
        "unresolved_instances": unresolved,
        "resolve_rate": resolved / total,
        "status": "completed",
        "notes": notes + [f"Inspected files: {inspected}"],
    }


def log_to_mlflow(config: dict[str, Any], metrics: dict[str, Any], manifest_path: Path) -> None:
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_dir = Path(config["run_dir"])
    with mlflow.start_run(run_name=config["run_id"]):
        mlflow.log_params(
            {
                "split": config["split"],
                "subset": config["subset"],
                "workers": config["workers"],
                "model": config["model"],
                "task_slice": config["task_slice"],
                "cost_limit": config["cost_limit"],
                "dataset_name": config["dataset_name"],
            }
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key, value)

        mlflow.set_tag("artifact_root", str(run_dir))
        mlflow.log_artifact(str(run_dir / "config.json"))
        mlflow.log_artifact(str(run_dir / "metrics.json"))
        mlflow.log_artifact(str(manifest_path))
        mlflow.log_text(str(run_dir), "artifact_root.txt")


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "split": Param("test", type="string"),
        "subset": Param("verified", type="string"),
        "workers": Param(1, type="integer", minimum=1),
        "model": Param(DEFAULT_MODEL, type="string"),
        "task_slice": Param("0:1", type="string"),
        "run_id": Param("", type="string"),
        "cost_limit": Param(0, type=["integer", "number"]),
    },
    tags=["coding-agent", "swe-bench", "mlflow"],
)
def evaluate_agent():
    @task(task_id="prepare_run")
    def prepare_run() -> dict[str, Any]:
        params = dict(get_current_context()["params"])
        run_id = safe_run_id(params.get("run_id"))
        created_at = utc_now().isoformat()
        run_dir = RUNS_ROOT / run_id
        run_agent_dir = run_dir / "run-agent"
        run_eval_dir = run_dir / "run-eval"

        for path in [
            run_dir,
            run_agent_dir / "trajectories",
            run_eval_dir / "logs",
            run_eval_dir / "reports",
        ]:
            path.mkdir(parents=True, exist_ok=True)

        config = {
            "run_id": run_id,
            "created_at": created_at,
            "split": params.get("split", "test"),
            "subset": params.get("subset", "verified"),
            "workers": coerce_int(params.get("workers", 1), "workers"),
            "model": params.get("model", DEFAULT_MODEL),
            "task_slice": params.get("task_slice", "0:1"),
            "cost_limit": coerce_float(params.get("cost_limit", 0), "cost_limit"),
            "dataset_name": dataset_name_for_subset(str(params.get("subset", "verified"))),
            "swebench_config": swebench_config_spec(),
            "run_dir": str(run_dir),
        }

        write_json(run_dir / "config.json", config)
        write_json(run_dir / "manifest.json", build_manifest(config))
        logger.info("Prepared run directory: %s", run_dir)
        return config

    @task(task_id="run_agent")
    def run_agent(config: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(config["run_dir"])
        run_agent_dir = run_dir / "run-agent"
        trajectories_dir = run_agent_dir / "trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "mini-extra",
            "swebench",
            "--subset",
            str(config["subset"]),
            "--split",
            str(config["split"]),
            "--model",
            str(config["model"]),
            "--slice",
            str(config["task_slice"]),
            "--config",
            str(config["swebench_config"]),
            "--workers",
            str(config["workers"]),
            "-o",
            str(trajectories_dir),
        ]
        run_checked(cmd, env={"MSWEA_COST_TRACKING": "ignore_errors"})
        preds_path = normalize_predictions(
            run_agent_dir,
            trajectories_dir,
            str(config["model"]),
            str(config["split"]),
        )
        logger.info("Normalized predictions to %s", preds_path)
        return {**config, "preds_path": str(preds_path)}

    @task(task_id="run_eval")
    def run_eval(config: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(config["run_dir"])
        run_eval_dir = run_dir / "run-eval"
        preds_path = Path(config["preds_path"])
        if not preds_path.exists():
            raise FileNotFoundError(f"Predictions file does not exist: {preds_path}")

        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            str(config["dataset_name"]),
            "--predictions_path",
            str(preds_path),
            "--max_workers",
            str(config["workers"]),
            "--run_id",
            str(config["run_id"]),
        ]
        run_checked(cmd)
        notes = copy_eval_outputs(run_eval_dir)
        return {**config, "eval_notes": notes}

    @task(task_id="summarize_and_log")
    def summarize_and_log(config: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(config["run_dir"])
        run_eval_dir = run_dir / "run-eval"
        metrics = collect_metrics(run_eval_dir, str(config["run_id"]))
        if config.get("eval_notes"):
            metrics["notes"].extend(config["eval_notes"])

        metrics_path = run_dir / "metrics.json"
        manifest_path = run_dir / "manifest.json"
        write_json(metrics_path, metrics)
        write_json(manifest_path, build_manifest(config, notes=metrics.get("notes", [])))
        log_to_mlflow(config, metrics, manifest_path)
        logger.info("Wrote metrics to %s", metrics_path)
        return metrics

    summarize_and_log(run_eval(run_agent(prepare_run())))


evaluate_agent()
