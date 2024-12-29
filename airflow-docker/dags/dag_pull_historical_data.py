from datetime import datetime
from airflow import DAG
from utils.pull_historical_data import PullHistoricalData
from airflow.operators.python import PythonOperator
import logging

import sys
sys.path.insert(0, "/opt/airflow/dags")

pull = PullHistoricalData()

def pull_historical_data(**context) : 
    # First run to get the historical data
    seasons = ["2019/2020", "2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"]
    logging.info(f"Pulling data for seasons: {seasons}")
    pull.pull_datas(seasons)


default_args = { 
    "owner" : "lukiwa",
    "start_date": datetime(2024, 9, 9),
    "retries" : 3
}

with DAG(
    "pull_historical_data",
    default_args = default_args,
    schedule = "@once",
    catchup = False,
    tags = ["pull-hist-data"]) as dag :
    
    task_pull_hist_data = PythonOperator(
    task_id = "task_pull_hist_data",
    python_callable = pull_historical_data,
    ) 
    


