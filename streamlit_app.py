import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
import os
import csv
import holidays
import joblib
import datetime as dt



# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Prediction Dashboard',
    page_icon=':earth_americas:',  # This is an emoji shortcode. Could be a URL too.
    layout="wide",
)

#-----------------------------------------------------------------------------
# CUSTOM CSS FOR ENHANCED UI

st.markdown(
    """
    <style>
    .stDataFrame {
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stHeader {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .stSubheader {
        font-size: 18px;
        font-weight: bold;
        color: #34495e;
    }
    .stMetric {
        background-color: #f9f9f9;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-label {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        padding: 5px;
        border-radius: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def colored_metric(label, value, delta=None, delta_color="normal"):
    """
    Displays a metric with custom colors.
    """
    st.markdown(
        f"""
        <div style="
            padding: 10px;
            border-radius: 10px;
            background-color: #f0f2f6;
            text-align: center;
            margin: 5px;
        ">
            <div style="font-size: 14px; color: #555;">{label}</div>
            <div style="font-size: 24px; font-weight: bold; color: #333;">{value}</div>
            {f'<div style="font-size: 14px; color: {delta_color};">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

#-----------------------------------------------------------------------------
# DATE PICKER

# Add a date picker for the user to select a date
st.header("📅 Select a Date")
selected_date = st.date_input("", datetime.now() - timedelta(days=1), key="date_picker")

# Format the selected date to match the required format (YYYY-MM-DD)
filedate = selected_date.strftime("%Y-%m-%d")

#-----------------------------------------------------------------------------
# DATA IMPORT AND PROCESSING

def importdata(filedate, period):
    year, month, day = filedate.split('-')
    addre = f"{Path(__file__).parent}/data/load/{year}/{month}/{day}-{month}-{year}.csv"
    if os.path.exists(addre):
        return getcsvfile(addre, period)
    else:
        return scrapdate(year, month, day, period)

def scrapdate(year, month, day, period):
    url = "http://www.delhisldc.org/Loaddata.aspx?mode="
    target_dir = "data/load"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("Target directory created.")

    month_dir = os.path.join(target_dir, str(year), f"{month}")
    date_str = f"{day}/{month}/{year}"

    if not os.path.exists(month_dir):
        os.makedirs(month_dir)
        print(f"Directory created for {month}/{year}")

    try:
        input_date = datetime(int(year), int(month), int(day))
        if input_date > datetime.now():
            st.warning("Future data cannot be scraped.")
            return None
        else:
            print(f"Scraping data for {date_str}")
            response = requests.get(url + date_str)
            soup = BeautifulSoup(response.text, 'lxml')
            table = soup.find('table', {'id': 'ContentPlaceHolder3_DGGridAv'})

            if table:
                headers = [el.text.strip() for el in table.find_all('tr')[0].find_all('td')]
                rows = [[el.text.strip() for el in row.find_all('td')] for row in table.find_all('tr')[1:]]

                if len(rows) > 0:
                    csv_filename = os.path.join(month_dir, f"{date_str.replace('/', '-')}.csv")
                    if os.path.exists(csv_filename):
                        os.remove(csv_filename)
                        print(f"Removed existing CSV file: {csv_filename}")

                    with open(csv_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerows(rows)

                    df = pd.read_csv(csv_filename)
                    df_clean = df.dropna()
                    df_clean['Other'] = round((df_clean['DELHI'] - (df_clean['BRPL'] + df_clean['BYPL'] + df_clean['NDPL'] + df_clean['NDMC'] + df_clean['MES'])), 2)
                    df_clean['Date'] = f"{date_str.replace('/', '-')}"
                    # Convert the 'Date' column to datetime format
                    df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d-%m-%Y')
                    # Function to determine the season based on the month
                    def get_season(date):
                        month = date.month
                        if month in [12, 1, 2]:
                            return 'Winter'
                        elif month in [3, 4, 5]:
                            return 'Spring'
                        elif month in [6, 7, 8]:
                            return 'Summer'
                        elif month in [9, 10, 11]:
                            return 'Autumn'
                    # Add the 'Season' column
                    df_clean['Season'] = df_clean['Date'].apply(get_season)
                    # Add the 'DayOfWeek' column
                    df_clean['DayOfWeek'] = df_clean['Date'].dt.day_name()
                    # Initialize the holidays object for India
                    india_holidays = holidays.India()
                    # Add the 'IsHoliday' column using the holidays package
                    df_clean['IsHoliday'] = df_clean['Date'].apply(lambda x: x in india_holidays)

                    df_clean.to_csv(csv_filename, index=False)
                    print(df_clean)
                    return df_clean
            else:
                st.warning(f"No table found for {date_str}")
                return None
    except Exception as e:
        st.error(f"Error occurred while scraping {date_str}: {str(e)}")
        return None

def getcsvfile(addre, period):
    df = pd.read_csv(addre)
    return df


#-----------------------------------------------------------------------------
# DATA PREDICTION

def prediction(date, model_path):
    """
    Makes predictions using a LightGBM model for a given date.

    Args:
        date (datetime): The date for which predictions are to be made.
        model_path (str): The path to the LightGBM model file.

    Returns:
        pd.DataFrame: A DataFrame containing the timeslots and corresponding predictions.
    """
    # Load the LightGBM model using joblib
    model = joblib.load(model_path)
    date_obj = pd.to_datetime(date)
    india_holidays = holidays.India()
    is_holiday = lambda x: x in india_holidays
    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return 3
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'

    # Define the timeslots for the given date
    timeslots = pd.date_range(start=date, end=date + pd.Timedelta(days=1), freq='5T')[:-1]


    # Prepare the input data
    data = {
        'TIMESLOT': timeslots.strftime('%H:%M').tolist(),
        'Year': date_obj.year,
        'Month': date_obj.month,
        'Day': date_obj.day,
        'Season': get_season(date),  # Example: Season 4 (adjust based on your data)
        'DayOfWeek': date_obj.weekday(),  # Monday=0, Sunday=6
        'IsHoliday': is_holiday(date)# Example: Not a holiday (0)
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    if model_path == f"{Path(__file__).parent}/scrap/model/delhi_random_forest_model.pkl":
        n=0
        tlist = []
        TimeSlot = df['TIMESLOT'].astype(str)
        for i in df["TIMESLOT"]:
            tlist.append(n)
            n = n+1
        df['TIMESLOT'] = tlist
        print(tlist)
    print(df.info())
    print(df)

    # Mark categorical features
    categorical_features = ['TIMESLOT', 'Season', 'DayOfWeek', 'IsHoliday']  # Adjust based on your model
    for feature in categorical_features:
        df[feature] = df[feature].astype('category')

    # Make predictions
    predictions = model.predict(df)

    df['Prediction'] = predictions
    # Add predictions to the DataFrame
    if model_path == f"{Path(__file__).parent}/scrap/model/delhi_random_forest_model.pkl":
        df['TIMESLOT']= TimeSlot
        df.to_csv(f"{Path(__file__).parent}/scrap/data/pred/delhi_random_forest_model.csv", index=False)
    elif model_path == f"{Path(__file__).parent}/scrap/model/delhi_lgbm_model.pkl":
        df.to_csv(f"{Path(__file__).parent}/scrap/data/pred/delhi_lgbm_model.csv", index=False)
    elif model_path == f"{Path(__file__).parent}/scrap/model/delhi_xgboost_model.pkl":
        df.to_csv(f"{Path(__file__).parent}/scrap/data/pred/delhi_xgboost_model.csv", index=False)
    return df

#-----------------------------------------------------------------------------
# FILTER HOURLY DATA

def filter_hourly_data(df):
    # Filter rows where TIMESLOT ends with ':00' (hourly data)
    df['TIMESLOT'] = df['TIMESLOT'].astype(str)
    hourly_data = df[df['TIMESLOT'].str.endswith(':00')]
    return hourly_data

#-----------------------------------------------------------------------------
# MAIN LAYOUT AND VISUALIZATIONS

# Import data for the selected date
chart_data = importdata(filedate, 1)

if chart_data is None:
    def predict_and_visualize(select_date, model_path, efficiency):
        if model_path == "home":
            result_df = pd.read_csv(f"{Path(__file__).parent}/scrap/data/pred/delhi_random_forest_model.csv")
            result_df1 = pd.read_csv(f"{Path(__file__).parent}/scrap/data/pred/delhi_lgbm_model.csv")
            result_df2 = pd.read_csv(f"{Path(__file__).parent}/scrap/data/pred/delhi_xgboost_model.csv")
            result_df['Prediction'] = round(((result_df['Prediction'].astype(float)+result_df1['Prediction'].astype(float)+result_df2['Prediction'].astype(float))/3),2)
        else:
            result_df = prediction(select_date,model_path)
        # Calculate metrics
        total_usage = result_df['Prediction'].sum()
        peak_usage = result_df['Prediction'].max()
        peak_timeslot = result_df.loc[result_df['Prediction'].idxmax(), 'TIMESLOT']
        min_usage = result_df['Prediction'].min()
        min_timeslot = result_df.loc[result_df['Prediction'].idxmin(), 'TIMESLOT']

        # Display metrics in columns with custom colors
        st.header("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
            f"""
            <div class="stMetric">
                <div class="metric-label" style="background-color: #3498db;">Model Efficiency</div>
                <div class="metric-value">{efficiency}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with col2:
            st.markdown(
            f"""
            <div class="stMetric">
                <div class="metric-label" style="background-color: #9b59b6;">Total Usage for Delhi</div>
                <div class="metric-value">{f"{total_usage:.2f} MW"}</div>
            </div>
            """,
            unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
            f"""
            <div class="stMetric">
                <div class="metric-label" style="background-color: #e74c3c;">Peak Usage</div>
                <div class="metric-value">{f"{peak_usage:.2f} MW"} <small style="color: #ff0000;">{f"at {peak_timeslot}"}</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with col4:
            st.markdown(
            f"""
            <div class="stMetric">
                <div class="metric-label" style="background-color: #2ecc71;">Minimum Usage</div>
                <div class="metric-value">{f"{min_usage:.2f} MW"} <small style="color: #06a56a;">{f"at {min_timeslot}"}</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Line chart for predictions over time
        st.header("Predictions Over Time")
        st.line_chart(filter_hourly_data(result_df).set_index('TIMESLOT')['Prediction'])

        # Bar chart for average predictions by hour
        st.header("Average Predictions by Hour")
        result_df['Hour'] = pd.to_datetime(result_df['TIMESLOT'], format='%H:%M').dt.hour
        hourly_avg = result_df.groupby('Hour')['Prediction'].mean().reset_index()
        st.bar_chart(hourly_avg.set_index('Hour'))

        # Use an expander to hide the predictions table
        with st.expander("View Detailed Predictions Table"):
            st.dataframe(result_df)


    st.info("This is predicted data")
    tab1, tab2, tab3, tab4 = st.tabs(["Home","Random Forest", "LightGBM", "XGBoost"])
    with tab4:
        # Make predictions
        predict_and_visualize(selected_date, f"{Path(__file__).parent}/scrap/model/delhi_xgboost_model.pkl","91.8%")
    with tab2:
        # Make predictions
        predict_and_visualize(selected_date, f"{Path(__file__).parent}/scrap/model/delhi_random_forest_model.pkl","93.2%")
    with tab3:
        # Make predictions
        predict_and_visualize(selected_date, f"{Path(__file__).parent}/scrap/model/delhi_lgbm_model.pkl","92.2%")
    with tab1:
        # Make predictions
        predict_and_visualize(selected_date, f"home","92.4%")
        
elif chart_data is not None:

    st.info("This is actual data")

    # Filter for hourly data
    hourly_data = filter_hourly_data(chart_data)

    # Round all numeric columns to 2 decimal places
    hourly_data = hourly_data.round(2)

    # Display the filtered data in an expander
    with st.expander("📂 View Raw Data"):
        st.dataframe(chart_data, use_container_width=True)

    # Metrics for total values with custom colors
    st.header("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="stMetric">
                <div class="metric-label" style="background-color: #3498db;">Total DELHI</div>
                <div class="metric-value">{round(chart_data['DELHI'].sum(), 2)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="stMetric">
                <div class="metric-label" style="background-color: #2ecc71;">Total BRPL</div>
                <div class="metric-value">{round(chart_data['BRPL'].sum(), 2)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="stMetric">
                <div class="metric-label" style="background-color: #e74c3c;">Total BYPL</div>
                <div class="metric-value">{round(chart_data['BYPL'].sum(), 2)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="stMetric">
                <div class="metric-label" style="background-color: #9b59b6;">Total Other</div>
                <div class="metric-value">{round(chart_data['Other'].sum(), 2)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Line Chart
    st.header("📈 Hourly Load Distribution")
    st.line_chart(
        data=hourly_data,
        x="TIMESLOT",
        y=["DELHI", "BRPL", "BYPL", "NDPL", "NDMC", "MES", "Other"]
    )

    # Bar Chart for Total Load by Category
    st.header("📊 Total Load by Category")
    total_load = chart_data[["DELHI", "BRPL", "BYPL", "NDPL", "NDMC", "MES", "Other"]].sum().round(2)
    st.bar_chart(total_load)

    # Area Chart for Cumulative Load
    st.header("📊 Cumulative Load Over Time")
    cumulative_data = hourly_data[["DELHI", "BRPL", "BYPL", "NDPL", "NDMC", "MES", "Other"]].cumsum().round(2)
    cumulative_data["TIMESLOT"] = hourly_data["TIMESLOT"]
    st.area_chart(cumulative_data, x="TIMESLOT", y=["DELHI", "BRPL", "BYPL", "NDPL", "NDMC", "MES", "Other"])

else:
    st.warning("No data available for the selected date.")