from datetime import datetime
import logging
from bs4 import BeautifulSoup
import requests
import os 
import pandas as pd

class PullHistoricalData :
    def __init__(self, season = "2024/2025") : 
        self.url = "https://www.football-data.co.uk"
        self.belgium = "/belgiumm.php"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": f"{self.url}{self.belgium}",
    }
        
    def transform_season(self, season):
        parts = season.split("/")
        return f"{parts[0][-2:]}_{parts[1][-2:]}" # Extract the last two digits of each part and join them with an underscore
        
    def save_csv_if_new(self,csv_response, file_path) :
        """Helper function to save the CSV file if it's new or larger."""
        try:
            if not os.path.exists(file_path) or os.stat(file_path).st_size < len(csv_response.text.encode('utf-8')):
                with open(file_path, "w+") as f:
                    f.write(csv_response.text)
                logging.info(f"File saved at {file_path}.")
            else:
                logging.info(f"File {file_path} already exists and is up to date.")
        except Exception as e:
            logging.error(f"Error saving file {file_path}: {e}")


    def pull_one_data(self, season):
        csv_dir = "/opt/airflow/data/csv"
        response = requests.get(f"{self.url}{self.belgium}", headers=self.headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            last_season = soup.find("i", string=f"Season {season}")

            if last_season:
                csv_url = last_season.find_next("a")["href"]
                csv_response = requests.get(f"{self.url}/{csv_url}")

                if csv_response.status_code == 200:
                    if not os.path.exists(csv_dir):
                        os.makedirs(csv_dir)
                        
                    csv_path = f"{csv_dir}/{self.transform_season(season)}_B1.csv"
                    self.save_csv_if_new(csv_response, csv_path)
                else:
                    logging.error("Error fetching CSV file.")
            else:
                logging.error(f"Couldn't find the 'i' tag with the {self.season} text.")
        else:
            logging.error("Error fetching the main page.")

            
    def pull_datas(self, seasons):
        current_year = datetime.now().year
        for season in seasons:
            start_year = int(season.split('/')[0])
            
            if season == f"{current_year}/{current_year + 1}":  # Always scrape the current season
                logging.info(f"{file_name} is for the current season. Scraping data.")
                self.pull_one_data(season)  # Scrape for the current season
                
            elif start_year <= current_year:
                file_name = self.transform_season(season) + "_B1.csv"
                file_path = os.path.join("/opt/airflow/data/csv/", file_name)
                
                if not os.path.exists(file_path):
                    logging.info(f"{file_name} does not exist. Scraping data for {season}.")
                    self.pull_one_data(season)
                else:
                    logging.info(f"{file_name} exists. Skipping scraping.")
            