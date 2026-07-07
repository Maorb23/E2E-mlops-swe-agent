import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST_PROJECT_ROOT = os.environ.get("HOST_PROJECT_ROOT", str(PROJECT_ROOT))
DOCKER_IMAGE = os.environ.get("AGENT_EVAL_IMAGE", "mlops-assignment-agent:latest")
DOCKER_NETWORK = os.environ.get("DOCKER_NETWORK", "mlops-eval-net")
DEFAULT_MODEL = "nebius/moonshotai/Kimi-K2.6"

RUN_ID_TEMPLATE = "{{ dag_run.conf.get('run_id') or params.run_id or dag_run.run_id }}"
SPLIT_TEMPLATE = "{{ dag_run.conf.get('split', params.split) }}"
SUBSET_TEMPLATE = "{{ dag_run.conf.get('subset', params.subset) }}"
WORKERS_TEMPLATE = "{{ dag_run.conf.get('workers', params.workers) }}"
MODEL_TEMPLATE = "{{ dag_run.conf.get('model', params.model) }}"
TASK_SLICE_TEMPLATE = "{{ dag_run.conf.get('task_slice', params.task_slice) }}"
COST_LIMIT_TEMPLATE = "{{ dag_run.conf.get('cost_limit', params.cost_limit) }}"

COMMON_ENV = {
    "PROJECT_ROOT": "/workspace",
    "MSWEA_COST_TRACKING": "ignore_errors",
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "MLFLOW_ALLOW_FILE_STORE": "true",
    "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
}

PROJECT_MOUNT = Mount(source=HOST_PROJECT_ROOT, target="/workspace", type="bind")
DOCKER_SOCKET_MOUNT = Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind")

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


def docker_task(task_id: str, command: str, timeout_hours: int) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image=DOCKER_IMAGE,
        command=command,
        working_dir="/workspace",
        environment=COMMON_ENV,
        mounts=[PROJECT_MOUNT, DOCKER_SOCKET_MOUNT],
        mount_tmp_dir=False,
        network_mode=DOCKER_NETWORK,
        auto_remove="success",
        docker_url=os.environ.get("DOCKER_URL", "unix://var/run/docker.sock"),
        execution_timeout=timedelta(hours=timeout_hours),
    )


with DAG(
    dag_id="evaluate_agent_docker",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    params={
        "split": Param("test", type="string"),
        "subset": Param("verified", type="string"),
        "workers": Param(1, type="integer", minimum=1),
        "model": Param(DEFAULT_MODEL, type="string"),
        "task_slice": Param("0:1", type="string"),
        "run_id": Param("", type="string"),
        "cost_limit": Param(0, type=["integer", "number"]),
    },
    tags=["coding-agent", "swe-bench", "mlflow", "docker", "phase-3"],
) as dag:
    prepare_run = docker_task(
        "prepare_run",
        "python scripts/evaluate_agent_pipeline.py prepare-run "
        f"--run-id '{RUN_ID_TEMPLATE}' "
        f"--split '{SPLIT_TEMPLATE}' "
        f"--subset '{SUBSET_TEMPLATE}' "
        f"--workers '{WORKERS_TEMPLATE}' "
        f"--model '{MODEL_TEMPLATE}' "
        f"--task-slice '{TASK_SLICE_TEMPLATE}' "
        f"--cost-limit '{COST_LIMIT_TEMPLATE}'",
        timeout_hours=1,
    )
    run_agent = docker_task(
        "run_agent",
        f"python scripts/evaluate_agent_pipeline.py run-agent --run-id '{RUN_ID_TEMPLATE}'",
        timeout_hours=6,
    )
    run_eval = docker_task(
        "run_eval",
        f"python scripts/evaluate_agent_pipeline.py run-eval --run-id '{RUN_ID_TEMPLATE}'",
        timeout_hours=8,
    )
    summarize_and_log = docker_task(
        "summarize_and_log",
        f"python scripts/evaluate_agent_pipeline.py summarize-and-log --run-id '{RUN_ID_TEMPLATE}'",
        timeout_hours=1,
    )

    prepare_run >> run_agent >> run_eval >> summarize_and_log
