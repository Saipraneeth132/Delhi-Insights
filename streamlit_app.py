import streamlit as st
import pandas as pd
import math
import requests
from bs4 import BeautifulSoup
from datetime import datetime ,timedelta
from pathlib import Path
import os
import csv
import joblib



# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
    layout="wide",
)


#-----------------------------------------------------------------------------
#DATE TIME

todaydate = datetime.now() - timedelta(1)

todayyear = todaydate.year
todaymonth = todaydate.month
todayday = todaydate.day


def importdata(filedate,period):
    year = filedate[0:4]
    month = filedate[5:7]
    day = filedate[8:10]
    print(filedate)
    print(year,month,day)
    addre = f"{ Path(__file__).parent}/data/load/{year}/{month}/{day}-{month}-{year}.csv"
    if os.path.exists(addre):
        return getcsvfile(addre,period)
    else:
        return scrapdate(year,month,day,period)



def scrapdate(year,month,day,period):
	# URL of the website to scrape
    url = "http://www.delhisldc.org/Loaddata.aspx?mode="

    target_dir = "data/load"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("Target directory created.")
		
    month_dir,date_str = os.path.join(target_dir, str(year), f"{month}"),f"{day}/{month}/{year}"

    if not os.path.exists(month_dir):
        os.makedirs(month_dir)
        print(f"Directory created for {month}/{year}")
	
    try:
        if (day == '6') and month == '1' and year == '2024':
            print("06/01/2024 data Not Available..")
        elif (int(day) > int(todayday) and int(month) >= int(todaymonth) and int(year) >= int(todayyear)) or (int(month) > int(todaymonth) and int(year) == int(todayyear)) or (int(year) > int(todayyear)):
            print("This is Actuall Data. Future Data Can't be found")
        else: 
            print(f"Scraping data for {date_str}")

             # Send an HTTP GET request to the URL with the date
            response = requests.get(url + date_str)

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'lxml')

            # Find the table with the specific ID
            table = soup.find('table', {'id': 'ContentPlaceHolder3_DGGridAv'})
			
            # Extract headers and data rows from the table
            headers = []
            rows = []
			
            for i, row in enumerate(table.find_all('tr')):
                if i == 0:
                    # Extract headers from the first row
                    headers = [el.text.strip() for el in row.find_all('td')];
                else:
                    # Extract data rows from subsequent rows
                    rows.append([el.text.strip() for el in row.find_all('td')])
			
             # Check if there's data in the table
            if len(rows) > 0:
                # Construct the CSV filename
                csv_filename = os.path.join(month_dir, f"{date_str.replace('/', '-')}.csv")
                # Remove the existing CSV file if it exists
                if os.path.exists(csv_filename):
                    os.remove(csv_filename)
                    print(f"Removed existing CSV file: {csv_filename}")
				
                # Write headers and data to the CSV file
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(rows)
					
                # Optionally clean and save the CSV using pandas (commented out)
                df = pd.read_csv(csv_filename)
                df_clean = df.dropna()
                df_clean['Other'] = round((df_clean['DELHI'] - (df_clean['BRPL']+df_clean['BYPL']+df_clean['NDPL']+df_clean['NDMC']+df_clean['MES'])),2)
                df_clean['Date'] = f"{date_str.replace('/', '-')}"
                df_clean.to_csv(csv_filename, index=False)
            return importdata(f"{year}-{month}-{day}",period)

    
    except Exception as e:
        print(f"Error occurred while scraping {date_str}: {str(e)}")


def getcsvfile(addre,period):
    with open(addre, 'r') as file:
        df = pd.read_csv(addre)
        dataTimeslot = df['TIMESLOT']
        dataDelhi = df['DELHI']
        dataBRPL = df['BRPL']
        dataBYPL = df['BYPL']
        dataNDPL = df['NDPL']
        dataNDMC = df['NDMC']
        dataMES = df['MES']
        dataOther = df['Other']
        dataDate = df['Date']
        return df

print(importdata(f"2025-01-12",1))

#-----------------------------------------------------------------------------
#graph

chart_data = pd.DataFrame(importdata(f"2025-01-12",1))

st.dataframe(chart_data,use_container_width=True)

st.line_chart(
    data=chart_data,
    x = "TIMESLOT",
    y= ("DELHI","BRPL","BYPL", "NDPL", "NDMC", "MES", "Other")
)

model1 = joblib.load(Path(__file__).parent'/model/sarimaxforestmodel.pkl')

y_pred = model1.predict(X_test)

