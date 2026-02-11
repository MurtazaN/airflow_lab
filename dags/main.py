# File: main.py
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.task.trigger_rule import TriggerRule


from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    load_model,
    get_model_accuracy,
    check_accuracy_threshold,
)

# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="Airflow-Lab2 DAG Description",
    schedule="@daily",
    catchup=False,
    tags=["example"],
    owner_links={"Murtaza Nipplewala": "https://github.com/MurtazaN/airflow_lab.git"},
    max_active_runs=1,
)

# ---------- Tasks ----------
owner_task = BashOperator(
    task_id="task_using_linked_owner",
    bash_command="echo 1",
    owner="Murtaza Nipplewala",
    dag=dag,
)

send_email = EmailOperator(
    task_id="send_email",
    to="murtaza.sn786@gmail.com",
    subject="Notification from Airflow",
    html_content="<p>This is a notification email sent from Airflow.</p>",
    dag=dag,
)

load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

separate_data_outputs_task = PythonOperator(
    task_id="separate_data_outputs_task",
    python_callable=separate_data_outputs,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id="build_save_model_task",
    python_callable=build_model,
    op_args=[separate_data_outputs_task.output, "model.sav"],
    dag=dag,
)

load_model_task = PythonOperator(
    task_id="load_model_task",
    python_callable=load_model,
    op_args=[separate_data_outputs_task.output, "model.sav"],
    dag=dag,
)

# Fire-and-forget trigger so this DAG can finish cleanly.
trigger_dag_task = TriggerDagRunOperator(
    task_id="my_trigger_task",
    trigger_dag_id="Airflow_Lab2_Flask",
    conf={"message": "Data from upstream DAG"},
    reset_dag_run=False,
    wait_for_completion=False,          # don't block
    trigger_rule=TriggerRule.ALL_DONE,  # still run even if something upstream fails
    dag=dag,
)

# ========== NEW: Branching Tasks ==========

# model accuracy is 97%
# changing threshold to 99% should trigger the failure email instead of the trigger task
ACCURACY_THRESHOLD = 0.90

def decide_next_task(**context):
    """Read accuracy from XCom, use check_accuracy_threshold to decide."""
    accuracy = context['ti'].xcom_pull(task_ids='evaluate_model_task')
    if check_accuracy_threshold(accuracy, ACCURACY_THRESHOLD):
        return 'my_trigger_task'
    return 'model_quality_failed_email'


evaluate_model_task = PythonOperator(
    task_id="evaluate_model_task",
    python_callable=get_model_accuracy,
    op_args=[separate_data_outputs_task.output, "model.sav"],
    dag=dag,
)

branch_on_quality = BranchPythonOperator(
    task_id="check_model_quality",
    python_callable=decide_next_task,
    dag=dag,
)

model_quality_failed_email = EmailOperator(
    task_id="model_quality_failed_email",
    to="murtaza.sn786@gmail.com",
    subject="Model Quality Below Threshold",
    html_content=f"""
    <h2>Model Training Alert</h2>
    <p>The model's accuracy is below the required {ACCURACY_THRESHOLD:.0%} threshold.</p>
    <p><b>Action Required:</b> Review training data and retrain.</p>
    """,
    dag=dag,
)

# ---------- Dependencies ----------

# Original pipeline (unchanged)
owner_task >> load_data_task >> data_preprocessing_task >> \
    separate_data_outputs_task >> build_save_model_task >> \
    load_model_task

# Optional: email after model loads (independent branch)
# load_model_task >> send_email

# NEW: Branching after load_model_task
# load_model_task → evaluate → check_quality → trigger_task (if passed)
#                                              → failed_email (if failed)
load_model_task >> evaluate_model_task >> branch_on_quality
branch_on_quality >> trigger_dag_task
branch_on_quality >> model_quality_failed_email
