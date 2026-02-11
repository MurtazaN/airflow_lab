# Airflow Lab

## Lab objective

This lab demonstrates how to use **Apache Airflow** to orchestrate a machine learning pipeline. The pipeline:

1. **Loads data** from a CSV file (advertising.csv)
2. **Preprocesses** the data (scaling, splitting into train/test)
3. **Trains** a Logistic Regression model
4. **Evaluates** the model's accuracy
5. **Sends email notifications** on completion
--- My Addition ---
6. **Branches** based on model quality:
   - If accuracy ≥ threshold → triggers Flask API
   - If accuracy < threshold → sends failure email alert

### Pipeline Visualization

```
owner_task → load_data → preprocess → separate → build_model → load_model
                                                                    ↓
                                                          evaluate_model
                                                                    ↓
                                                          check_model_quality
                                                              ↙         ↘
                                                    (≥ threshold)    (< threshold)
                                                          ↓                ↓
                                                  trigger_flask_api   failure_email
```

---

## Prerequisites

- **Docker** and **Docker Compose** installed
- **Git** (to clone the repo)
- A **Gmail account** with an App Password (for email notifications)

---

## Quick Start (Step-by-Step)

### 1. Clone the Repository

```bash
git clone https://github.com/MurtazaN/airflow_lab.git
cd airflow_lab
```

### 2. Set Up Environment

Create required directories and set permissions:

```bash
mkdir -p logs plugins
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

Update the following in `docker-compose.yaml`:

```yaml
# Do not load examples
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'

# Output dir (add to volumes section)
- ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data

# Change default admin credentials
_AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow2}
_AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow2}
```

### 3. Configure Email (Optional)

If you want email notifications to work:

1. Go to [Google App Passwords](https://support.google.com/accounts/answer/185833)
2. Generate an app password for "Mail"
3. Add your credentials to the `.env` file (this file is not committed to git):

```bash
echo "SMTP_USER=your-email@gmail.com" >> .env
echo "SMTP_PASSWORD=your-app-password" >> .env
```

The `docker-compose.yaml` will automatically read these variables.

### 4. Start Airflow
1. Initialize the database (this will take a couple of minutes):

```bash
docker compose up airflow-init
```

2. Run Airflow

```bash
docker compose up -d
```

Wait until terminal outputs something like:

```
airflow-webserver-1  | 127.0.0.1 - - [19/Feb/2024:17:16:53 +0000] "GET /health HTTP/1.1" 200 318 "-" "curl/7.88.1"
```

All services should show "healthy" or "running".

### 5. Access Airflow Web UI

Open your browser and go to: **http://localhost:8080**

- **Username:** `airflow` (by default, or use the one you created)
- **Password:** `airflow` (by default, or use the one you created)

### 6. Run the DAG

1. In the Airflow UI, find `Airflow_Lab2` in the DAG list
2. Toggle the DAG "ON" (switch on the left)
3. Click the "Play" button (▶) to trigger a manual run
4. Click on the DAG name to watch the tasks execute

---

## Stopping and Restarting

### Stop Airflow (keep data)

```bash
docker compose down
```

### Stop Airflow (delete all data)

```bash
docker compose down -v
```

### Restart Airflow

```bash
docker compose up -d
```

---


## Customization

### Change the Accuracy Threshold

In `dags/main.py`, find this line:

```python
ACCURACY_THRESHOLD = 0.99  # Change this value
```

- Set to `0.8` (80%) for normal operation
- Set to `0.99` (99%) to trigger the failure path (model is ~97% accurate)

### Change Email Recipient

In `dags/main.py`, update the `to` field in `EmailOperator` tasks:

```python
to="your-email@example.com"
```

---

## Change Overview (Branching Addition)

### `dags/main.py`

Added for branching:
- `ACCURACY_THRESHOLD` - configurable threshold (default 0.8 = 80%)
- `decide_next_task()` - reads accuracy from XCom, returns next task_id
- `evaluate_model_task` - PythonOperator that calls `get_model_accuracy()`
- `branch_on_quality` - BranchPythonOperator that decides which path to take
- `model_quality_failed_email` - EmailOperator for failure notification

### `dags/src/model_development.py`

Added for branching:
- `get_model_accuracy()` - calculates and returns model accuracy score
- `check_accuracy_threshold()` - compares accuracy to threshold, returns True/False

