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
    # NEW: Functions for parallel model training
    train_logistic_regression,
    train_random_forest,
    compare_models,
    check_best_model_threshold,
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

# NEW: Branching Tasks

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


# NEW: Parallel Model Training & Comparison

# Task 1: Train Logistic Regression (runs in parallel with Random Forest)
train_lr_task = PythonOperator(
    task_id="train_logistic_regression",
    python_callable=train_logistic_regression,
    # Pass the file path from separate_data_outputs_task
    op_args=[separate_data_outputs_task.output],
    dag=dag,
)

# Task 2: Train Random Forest (runs in parallel with Logistic Regression)
train_rf_task = PythonOperator(
    task_id="train_random_forest",  
    python_callable=train_random_forest,
    op_args=[separate_data_outputs_task.output],
    dag=dag,
)

# Task 3: Compare models and pick the best one
# This task WAITS for both training tasks to complete.
# It reads their results from XCom using .output
compare_models_task = PythonOperator(
    task_id="compare_models",
    python_callable=compare_models,
    # op_args passes positional arguments to the function
    # train_lr_task.output = the dict returned by train_logistic_regression
    # train_rf_task.output = the dict returned by train_random_forest
    op_args=[train_lr_task.output, train_rf_task.output],
    dag=dag,
)

# Decision function for model comparison branch
def decide_after_comparison(**context):
    """
    After comparing models, decide if the BEST model meets our threshold.
    
    This reads the comparison result from XCom and decides:
    - If best model accuracy >= threshold → proceed to Flask API
    - If best model accuracy < threshold → send failure email
    """
    # Read the comparison result from XCom
    comparison_result = context['ti'].xcom_pull(task_ids='compare_models')
    
    # Use our function to check the threshold
    if check_best_model_threshold(comparison_result, ACCURACY_THRESHOLD):
        return 'my_trigger_task'
    return 'comparison_failed_email'


# Branch after model comparison
branch_after_comparison = BranchPythonOperator(
    task_id="branch_after_comparison",
    python_callable=decide_after_comparison,
    dag=dag,
)

# Email for when even the best model isn't good enough
comparison_failed_email = EmailOperator(
    task_id="comparison_failed_email",
    to="murtaza.sn786@gmail.com",
    subject="Best Model Below Threshold",
    html_content=f"""
    <h2>Model Comparison Alert</h2>
    <p>Both models were trained and compared, but even the best model
    is below the required {ACCURACY_THRESHOLD:.0%} threshold.</p>
    <p><b>Action Required:</b> Try different models or more data.</p>
    """,
    dag=dag,
)

# ---------- Dependencies ----------
#
# ORIGINAL PIPELINE (unchanged - single model)
# =============================================
#
# owner → load_data → preprocess → separate → build_model → load_model
#                                                               ↓
#                                                         evaluate_model
#                                                               ↓
#                                                       check_model_quality
#                                                           ↙         ↘
#                                                   trigger_task    failed_email
#
owner_task >> load_data_task >> data_preprocessing_task >> \
    separate_data_outputs_task >> build_save_model_task >> \
    load_model_task

# Branching after single model evaluation
load_model_task >> evaluate_model_task >> branch_on_quality
branch_on_quality >> trigger_dag_task
branch_on_quality >> model_quality_failed_email


#
#                                        ↗ train_logistic_regression ↘
# owner → load_data → preprocess → separate                            → compare_models → branch
#                                        ↘ train_random_forest       ↗           ↙         ↘
#                                                                        trigger_task  comparison_failed_email
#
#

# Fork: After separate_data, run BOTH training tasks in parallel
separate_data_outputs_task >> [train_lr_task, train_rf_task]

# Join: compare_models waits for BOTH training tasks to complete
[train_lr_task, train_rf_task] >> compare_models_task

# After comparison, branch based on best model's accuracy
compare_models_task >> branch_after_comparison
branch_after_comparison >> trigger_dag_task
branch_after_comparison >> comparison_failed_email
