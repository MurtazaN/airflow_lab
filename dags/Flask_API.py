from __future__ import annotations

import os
import time
# import base64  # No longer needed - Airflow 3.x uses JWT, not Basic Auth
import pendulum
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from flask import Flask, redirect, render_template

# ---------- Config ----------
# Ensure WEBSERVER points to 'airflow-apiserver' as defined in your compose
WEBSERVER = os.getenv("AIRFLOW_WEBSERVER", "http://airflow-apiserver:8080")
AF_USER   = os.getenv("AIRFLOW_USERNAME", "airflow")
AF_PASS   = os.getenv("AIRFLOW_PASSWORD", "airflow")
TARGET_DAG_ID = os.getenv("TARGET_DAG_ID", "Airflow_Lab2")

# Set up robust paths for Docker Workers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# ---------- Flask app ----------
app = Flask(__name__, template_folder=TEMPLATE_DIR)

def get_jwt_token():
    """
    Airflow 3.x uses JWT tokens for API authentication.
    POST to /auth/token with username/password to get an access token.
    """
    token_url = f"{WEBSERVER}/auth/token"
    try:
        r = requests.post(
            token_url,
            json={"username": AF_USER, "password": AF_PASS},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        # Note: Airflow 3.x returns 201 Created for successful token generation
        if r.status_code in (200, 201):
            return r.json().get("access_token")
        else:
            print(f"Failed to get JWT token: {r.status_code} - {r.text}", flush=True)
            return None
    except Exception as e:
        print(f"Exception getting JWT token: {e}", flush=True)
        return None

def get_latest_run_info():
    # Airflow 3 prefers ordering by 'run_after' for the most recent activity    
    url = f"{WEBSERVER}/api/v2/dags/{TARGET_DAG_ID}/dagRuns?order_by=-run_after&limit=1"

    info = {
        "state": "unknown",
        "run_id": "N/A",
        "logical_date": "N/A",
        "start_date": "N/A",
        "end_date": "N/A",
        "note": ""
    }

    # OLD: Basic Auth (doesn't work in Airflow 3.x for protected endpoints)
    # auth_bytes = f"{AF_USER}:{AF_PASS}".encode("ascii")
    # base64_auth = base64.b64encode(auth_bytes).decode("ascii")
    # headers = {"Accept": "application/json", "Authorization": f"Basic {base64_auth}"}

    # NEW: JWT Bearer Token Auth (required for Airflow 3.x)
    token = get_jwt_token()
    if not token:
        info["note"] = "Failed to authenticate with Airflow API"
        return False, info
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code != 200:
            info["note"] = f"API Error {r.status_code}"
            return False, info

        data = r.json()
        runs = data.get("dag_runs", [])
        
        if not runs:
            info["note"] = "Waiting for DAG run to appear..."
            return False, info

        run = runs[0]
        state = run.get("state", "unknown")

        print(state)

        info.update({
            "state": state,
            "run_id": run.get("dag_run_id", "N/A"),
            "logical_date": run.get("logical_date") or run.get("run_after", "N/A"),
            "start_date": run.get("start_date", "N/A"),
            "end_date": run.get("end_date", "N/A"),
        })
        
        # LOGIC CHANGE: If it's running, we don't want to say "Failure"
        # We return False to stay on failure/loading page, but update the note
        if state == "queued" or state == "running":
            info["note"] = "Model pipeline is still in progress..."
            return False, info
            
        return state == "success", info

    except Exception as e:
        info["note"] = f"Network issue: {str(e)}"
        return False, info


@app.route("/")
def index():
    ok, _ = get_latest_run_info()
    return redirect("/success" if ok else "/failure")

@app.route("/success")
def success():
    _, info = get_latest_run_info()
    return render_template("success.html", **info)

@app.route("/failure")
def failure():
    _, info = get_latest_run_info()
    return render_template("failure.html", **info)

@app.route("/health")
def health():
    return "ok", 200

def start_flask_app():
    """
    Starts the Flask server. 
    Note: Blocks the worker slot. Disable reloader for Docker.
    """
    print(f"Starting Flask on 0.0.0.0:5555 checking {WEBSERVER}...", flush=True)
    app.run(host="0.0.0.0", port=5555, debug=False, use_reloader=False)

# ---------- DAG Definition ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

dag = DAG(
    dag_id="Airflow_Lab2_Flask",
    default_args=default_args,
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
)

PythonOperator(
    task_id="start_Flask_API",
    python_callable=start_flask_app,
    dag=dag,
)