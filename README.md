# Cryptocurrency Price Forecasting with Facebook Prophet

This project demonstrates the use of the Facebook Prophet library for time series forecasting on cryptocurrency prices. The primary focus is on predicting future closing prices for various cryptocurrencies, including Bitcoin, Ethereum, Bitcoin Cash, XRP Ripple, Litecoin, and Tether.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Running the Project](#running-the-project)
- [Results](#results)
- [Model Evaluation](#model-evaluation)
- [Future Improvements](#future-improvements)

## Project Overview

The goal of this project is to forecast the future closing prices of cryptocurrencies using historical data. The project applies a systematic approach to time series forecasting with the **Facebook Prophet** model. It focuses on:
- Importing historical data of cryptocurrencies
- Data exploration and transformation
- Applying the Facebook Prophet model
- Visualizing the results
- Evaluating the model’s performance

Cryptocurrencies used:
- **Bitcoin**
- **Ethereum**
- **Bitcoin Cash**
- **XRP Ripple**
- **Litecoin**
- **Tether**

## Project Workflow

1. **Data Import and Exploration**: Historical price data for various cryptocurrencies is imported and visualized to explore trends and distributions.
2. **Data Transformation**: 
    - Dates are converted to a timestamp format.
    - Logarithmic transformation is applied to closing prices to stabilize variance.
3. **Facebook Prophet Model Implementation**: 
    - Data is split into training and test sets.
    - The Prophet model is applied to the training set.
    - Future prices are predicted using the trained model.
4. **Model Evaluation**: 
    - The predicted results are compared to the test data.
    - Metrics such as **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** are calculated to evaluate the model’s accuracy.
5. **Visualization**: 
    - Forecasted prices are plotted alongside actual prices.
    - Rolling statistics and histograms are used for data exploration.

## Technologies Used

- **Python**: Programming language for data manipulation and model development.
- **Facebook Prophet**: Time series forecasting library used for making predictions.
- **Pandas**: Data manipulation library.
- **Matplotlib/Seaborn**: Libraries for data visualization.
- **Numpy**: Used for numerical computations.
- **Scikit-learn**: For model evaluation (calculating MSE, MAE).
- **Plotly**: For interactive plotting.

## Running the Project

### Prerequisites

Before running the project, ensure you have the following dependencies installed:
- Python 3.7+
- Pandas
- Numpy
- Facebook Prophet
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

You can install all dependencies using the following command:

For example, for **Bitcoin**, the results are:
``bash
Mean Squared Error: [Calculated Value]
Mean Absolute Error: [Calculated Value]
Root Mean Squared Error: [Calculated Value]

Running the Project
-------------------

### Prerequisites

Ensure the following libraries are installed:

bash

`pip install pandas numpy matplotlib seaborn plotly fbprophet scikit-learn`

### Instructions to Run

1.  **Clone the repository**:

    bash

    `git clone https://github.com/your-username/crypto-price-forecasting.git
    cd crypto-price-forecasting`

2.  **Prepare the Data**: Ensure that you have the CSV files for the cryptocurrencies:

    -   Bitcoin (`Bitcoin1.csv`)
    -   Ethereum (`Ethereum1.csv`)
    -   Bitcoin Cash (`Bitcoincash1.csv`)
    -   XRP Ripple (`Xrpripple1.csv`)
    -   Litecoin (`Litecoin1.csv`)
    -   Tether (`Tether1.csv`)

    Update the file paths in the script to point to your CSV data.

3.  **Run the Script**: Execute the Python script to train the model and generate predictions.

    bash

    `python crypto_forecast.py`

4.  **View Results**: After running the script, the program will output the **MSE** and **MAE** for each cryptocurrency, along with graphical visualizations.

Results and Visualizations
--------------------------

### Bitcoin Forecast Visualization:

-   The following plot showcases the original and predicted closing prices for **Bitcoin** over the next 90 days.

-   Similar visualizations and metrics are available for other cryptocurrencies like Ethereum, Litecoin, and XRP Ripple.

### Rolling Mean and Standard Deviation Example:

-   This chart helps in identifying trends and seasonality by visualizing the **Rolling Mean** and **Rolling Standard Deviation**.

Conclusion
----------

This project demonstrates a systematic approach to time series forecasting using **Facebook Prophet** for predicting cryptocurrency prices. While the model provides a solid forecast, future improvements could involve adding more features (e.g., market volume, external financial indicators) and experimenting with more complex models to further improve accuracy.

Overall, the project provides a solid foundation for time series analysis in the financial domain.

Future Improvements
-------------------

-   Use more advanced models like **ARIMA**, **LSTM**, or **XGBoost** for comparison with **Facebook Prophet**.
-   Incorporate external factors such as market volume, macroeconomic indicators, or news sentiment analysis to improve model accuracy.
