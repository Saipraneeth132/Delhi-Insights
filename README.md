# 🌆 Delhi Insights: Electricity Load Prediction Dashboard

Welcome to **Delhi Insights**, a powerful Streamlit application designed to provide insights into electricity load data for Delhi. This dashboard allows users to visualize actual load data, predict future load using advanced machine learning models, and explore key metrics through interactive visualizations.

---

## 🚀 Features

- **📅 Date Selection:** Choose a specific date to view or predict electricity load data.
- **📊 Actual Data Visualization:** View historical load data with interactive charts and key metrics.
- **🤖 Prediction Models:** Compare predictions from multiple machine learning models, including Random Forest, LightGBM, and XGBoost.
- **📈 Key Metrics:** Display essential metrics like total usage, peak usage, and minimum usage.
- **🎨 Interactive Visualizations:** Explore data with line charts, bar charts, and area charts.
- **✨ Custom UI:** Enhanced user interface with custom CSS for a seamless experience.

---

## 🛠️ Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/delhi-insights.git
   cd delhi-insights
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install streamlit
   pip install pandas
   pip install requests
   pip install beautifulsoup4
   pip install lxml
   pip install holidays
   pip install joblib
   pip install openpyxl

   ```

3. **Run the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the Dashboard:**
   Open your browser and navigate to `http://localhost:8501`.

---

## 🖥️ Usage

### 1. **Select a Date**
   - Use the date picker to select a date for which you want to view or predict electricity load data.
   - The default date is set to yesterday.

   ![Date Picker](https://via.placeholder.com/600x200.png?text=Date+Picker+UI)

### 2. **View Actual Data**
   - If data is available for the selected date, the dashboard will display:
     - **Key Metrics:** Total load for Delhi, BRPL, BYPL, and other categories.
     - **Charts:** Hourly load distribution, total load by category, and cumulative load over time.
     - **Raw Data:** Expand the "View Raw Data" section to see the complete dataset.

   ![Actual Data](https://via.placeholder.com/600x200.png?text=Actual+Data+UI)

### 3. **Predict Load Data**
   - If no data is available for the selected date, the dashboard will provide predictions using:
     - **Random Forest Model**
     - **LightGBM Model**
     - **XGBoost Model**
     - **Ensemble Model (Average of all models)**
   - Explore predictions through interactive charts and detailed tables.

   ![Prediction](https://via.placeholder.com/600x200.png?text=Prediction+UI)

### 4. **Compare Models**
   - Switch between tabs to compare predictions from different models.
   - View model efficiency and key metrics for each prediction.

   ![Model Comparison](https://via.placeholder.com/600x200.png?text=Model+Comparison+UI)

---

## 📊 Key Metrics and Visualizations

### Metrics:
- **Total Usage:** Sum of predicted or actual load for the selected date.
- **Peak Usage:** Maximum load observed and the corresponding timeslot.
- **Minimum Usage:** Minimum load observed and the corresponding timeslot.

### Visualizations:
- **Line Chart:** Hourly load distribution over time.
- **Bar Chart:** Average predictions by hour.
- **Area Chart:** Cumulative load over time.

---

## 🎨 Custom UI

The dashboard features a custom UI with enhanced styling:
- **Custom CSS:** Improved aesthetics for buttons, headers, and metrics.
- **Colored Metrics:** Key metrics are displayed with colored backgrounds for better visibility.
- **Interactive Elements:** Expandable sections for raw data and detailed predictions.

---

## 📂 File Structure

```
delhi-insights/
├── streamlit_app.py                  # Main Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # List of dependencies
├── data/                   # Directory for storing data
│   └── load/               # Load data files
├── scrap/                  # Directory for scraped data and models
│   ├── model/              # Trained model files
│   └── data/               # Predicted data files
└── assets/                 # Static assets (e.g., images)
```

---

## 📝 Notes

- **Data Source:** The application scrapes data from [Delhi SLDC](http://www.delhisldc.org/Loaddata.aspx?mode=).
- **Model Efficiency:** The efficiency values for the models are based on their performance on the training dataset.
- **Future Enhancements:** Support for additional models, dynamic efficiency calculation, and more interactive visualizations.

---

## 🤝 Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework.
- [Delhi SLDC](http://www.delhisldc.org/) for providing the data.

---

Enjoy exploring **Delhi Insights** and uncovering the trends in electricity load data! 🌟
