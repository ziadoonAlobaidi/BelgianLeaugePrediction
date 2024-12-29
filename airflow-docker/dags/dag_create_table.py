# from airflow import DAG
# from airflow.operators.postgres_operator import PostgresOperator
# from datetime import datetime

# default_args = {
#     'owner': 'airflow',
#     'start_date': datetime(2023, 9, 1),
#     'retries': 1,
# }

# with DAG(dag_id='example_postgres_dag',
#          default_args=default_args,
#          schedule_interval='@none',
#          catchup=False) as dag:

#     # Create table task
#     create_table = PostgresOperator(
#         task_id='create_table',
#         postgres_conn_id='my_postgres_db',  # Connection ID created in the UI
#         sql='''
#         CREATE TABLE IF NOT EXISTS example_table (
#             id SERIAL PRIMARY KEY,
#             name VARCHAR(50),
#             age INT
#         );
        # '''
    # )