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

RUN_ID_TEMPLATE = "{{ params.run_id or dag_run.run_id }}"

COMMON_ENV = {
    "PROJECT_ROOT": "/workspace",
    "EVAL_RUN_ID": RUN_ID_TEMPLATE,
    "EVAL_SPLIT": "{{ params.split }}",
    "EVAL_SUBSET": "{{ params.subset }}",
    "EVAL_WORKERS": "{{ params.workers }}",
    "EVAL_MODEL": "{{ params.model }}",
    "EVAL_TASK_SLICE": "{{ params.task_slice }}",
    "EVAL_COST_LIMIT": "{{ params.cost_limit }}",
    "MSWEA_COST_TRACKING": "ignore_errors",
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "MLFLOW_ALLOW_FILE_STORE": "true",
    "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
    "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
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
        "python scripts/evaluate_agent_pipeline.py prepare-run",
        timeout_hours=1,
    )
    run_agent = docker_task(
        "run_agent",
        "python scripts/evaluate_agent_pipeline.py run-agent",
        timeout_hours=6,
    )
    run_eval = docker_task(
        "run_eval",
        "python scripts/evaluate_agent_pipeline.py run-eval",
        timeout_hours=8,
    )
    summarize_and_log = docker_task(
        "summarize_and_log",
        "python scripts/evaluate_agent_pipeline.py summarize-and-log",
        timeout_hours=1,
    )

    prepare_run >> run_agent >> run_eval >> summarize_and_log
