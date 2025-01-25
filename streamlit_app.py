import streamlit as st
import pandas as pd
import math
import requests
from bs4 import BeautifulSoup
from datetime import datetime ,timedelta
from pathlib import Path
import os
import csv



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
	addre = f"Path(__file__).parent/'data/load/{year}/{month}/{day}-{month}-{year}.csv"
	if os.path.exists(addre):
		return getcsvfile(addre,period)
	else:
		return scrapdate(year,month,day,period)



def scrapdate(year,month,day,period):
	# URL of the website to scrape
    url = "http://www.delhisldc.org/Loaddata.aspx?mode="

    target_dir = "Path(__file__).parent/'data/load"
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
		csv_reader = csv.reader(file)
		labelscsv = next(csv_reader)  # Read the header row
	scrapd = pd.read_csv(addre)
	dwhole = list(scrapd)
	adTs = list(scrapd["TIMESLOT"])
	addh = list(scrapd["DELHI"])
	adbrpl = list(scrapd["BRPL"])
	adbypl = list(scrapd["BYPL"])
	adndpl = list(scrapd["NDPL"])
	adndmc = list(scrapd["NDMC"])
	admes = list(scrapd["MES"])
	adothers = list(scrapd["Other"])
    return dwhole,dts,ddh,dbrpl,dbypl,dndpl,dndmc,dmes,dothers,labels,period	



# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )
