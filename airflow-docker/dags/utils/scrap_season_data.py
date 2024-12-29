import ast
from datetime import datetime
import logging
from bs4 import BeautifulSoup
import requests
import os 
import pandas as pd
import re 

class ScrapSeasonData:
    def __init__(self) : 
        self.url = "https://www.walfoot.be/belgique/jupiler-pro-league/calendrier"
        self.headers = headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": f"{self.url}",
        }
        self.soup = self.connect_to_url(self.url)

    def connect_to_url(self, link):
        response = requests.get(f"{link}", headers=self.headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup
        else:
            logging.error(f"Error connecting to {link}.")
            return None
        
    def create_dict(self) : 
        col_titles = [
        "Week " + title.text.split(" ")[1]
        for title in self.soup.find_all("a", id=re.compile(r"^calendar_matchday_"))
        if title.text.startswith("Journée")
        ]

        match_dict = {week: [] for week in col_titles}
        return match_dict
    
    def get_week_links(self) : 
        week_links = self.soup.find("select", class_="form-control").find_all("option")
        week_links = ["/" + link["value"].split("/")[4] for link in week_links if re.search(r"journee-\d+(.*)", link["value"])]
        return week_links

    def scrap_season_data(self) : 
        match_dict = self.create_dict()
        
        for i, link in enumerate(self.get_week_links()) : 
            response2 = requests.get(f"{self.url}{link}", headers=self.headers)
            soup2 = BeautifulSoup(response2.text, "html.parser")

            matches = soup2.find_all("tr", class_="table-active")
            for match_data in matches:
                match_data = match_data.find_all("td")
                match_info = {
                    "date": match_data[0].text.split(" ")[0].strip(),
                    "time": match_data[0].text.split(" ")[1].strip(),
                    "home_team": match_data[1].text.strip(),
                    "away_team": match_data[3].text.strip(),
                    "home_team_score": match_data[2].text.split("-")[0].strip() if match_data[2].text.split("-")[0].strip() != "..." else None,
                    "away_team_score": match_data[2].text.split("-")[1].strip() if match_data[2].text.split("-")[0].strip() != "..." else None,
                    "home_team_logo": match_data[1].find("img")['src'],
                    "away_team_logo": match_data[3].find("img")['src']
                }

                match_dict[f"Week {i + 1}"].append(match_info)
                
        self.save_data(match_dict, "24_25_all_matches_B1.csv")
        return match_dict

    def save_data(self, data : dict, filename : str) -> None : 
        all_weeks = []

        # Iterate through the weeks and matches in the dictionary
        for week_index, (week_name, matches) in enumerate(data.items(), start=1):
            for match in matches:
                match['week'] = week_name
                all_weeks.append(match)

        df = pd.DataFrame(all_weeks)

        columns = ['week'] + [col for col in df.columns if col != 'week']
        df = df[columns]

        # Define the CSV directory
        csv_dir = "/opt/airflow/data/csv"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        # Save the DataFrame to a CSV file
        file_path = os.path.join(csv_dir, filename)
        df.to_csv(file_path, index=False)

        print(f"Data saved to {file_path}")

