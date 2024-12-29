from datetime import datetime

from airflow import DAG
from utils.scrap_season_data import ScrapSeasonData
from airflow.operators.python import PythonOperator

import sys
sys.path.insert(0, "/opt/airflow/dags")

scrap = ScrapSeasonData()

def scrap_season_data(): 
    scrap.scrap_season_data()
    

default_args = { 
    "owner" : "lukiwa",
    "start_date": datetime(2024, 9, 9),
    "retries" : 3
}

with DAG (
    "scrap_season_data",
    default_args = default_args,
    schedule = "0 6 * * 1",
    catchup = False,
    tags = ["scrap-season-data"]) as dag : 
    
    task_pull_recent_data = PythonOperator(
    task_id = "task_scrap_season_data",
    python_callable = scrap_season_data,
) 