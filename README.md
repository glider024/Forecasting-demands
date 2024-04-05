# Energy Demand Forecasting with Streamlit

This project demonstrates how to use Streamlit to create a web application for forecasting energy demand of Turkey using machine learning techniques. The application loads a pre-trained machine learning model and allows users to input parameters such as hour, day of the week, and month to predict energy demand.

## Installation

1. Clone the repository:

    ```bash
    https://github.com/glider024/Forecasting-demands.git
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run appp.py
    ```

2. Open your web browser and go to `http://localhost:8503` to access the Streamlit app.

3. Use the slider to input the parameters for hour, day of the week, and month.

4. The app will display the predicted energy demand based on the input parameters.

## Files

- `appp.py`: Main Python script containing the Streamlit application code.
- `turkey energy consumption.csv`: Dataset containing Turkey's energy demand data.
- `random_forest_model.joblib`: Pre-trained Random Forest model for energy demand forecasting, saved in joblib format
- `requirements.txt`: List of Python packages required for running the application.

## Exploratory Data Analysis (EDA)

- `Energy Forecasting EDA .ipynb`: Jupyter Notebook containing exploratory data analysis (EDA) of the energy demand dataset. It includes data visualization about the data

## Dependencies

- Streamlit
- pandas
- scikit-learn


