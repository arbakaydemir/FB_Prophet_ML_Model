# Cryptocurrency Price Forecasting with Facebook Prophet

This project demonstrates the implementation of a machine learning model for **time series forecasting** using the [Facebook Prophet](https://facebook.github.io/prophet/) library. The main goal of the model is to predict future closing prices of various cryptocurrencies, including:

- Bitcoin (BTC)
- Ethereum (ETH)
- Bitcoin Cash (BCH)
- XRP Ripple (XRP)
- Litecoin (LTC)
- Tether (USDT)

The project involves importing historical price data, preprocessing the data, applying the Prophet model for forecasting, and evaluating the model's performance using statistical metrics such as **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Workflow Summary](#workflow-summary)
4. [Model Evaluation](#model-evaluation)
5. [How to Run the Project](#how-to-run-the-project)
6. [Results and Visualizations](#results-and-visualizations)
7. [Conclusion](#conclusion)

## Project Overview
The purpose of this project is to build a time series forecasting model to predict the closing prices of various cryptocurrencies. By using historical price data, the model attempts to forecast future prices based on trends and seasonality patterns. This model is particularly useful for understanding potential market movements and gaining insights into cryptocurrency price behavior.

## Technologies Used
- **Python**: Programming language used for data manipulation, modeling, and visualization.
- **Facebook Prophet**: Library for time series forecasting with seasonality and trend detection.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib / Seaborn**: For data visualization and exploration.
- **NumPy**: For mathematical operations.
  
## Workflow Summary

1. **Data Import and Exploration**:
   - Historical price data for multiple cryptocurrencies is imported.
   - Visualizations are used to explore trends and distributions of closing prices over time.
  
2. **Data Transformation**:
   - The time series data is preprocessed, converting date columns to timestamp formats.
   - Logarithmic transformation is applied to stabilize variance in the closing prices.
  
3. **Facebook Prophet Model Implementation**:
   - The Facebook Prophet model is applied to the transformed data.
   - The model is trained on the historical data and tested on the remaining data to forecast future closing prices.
  
4. **Model Evaluation**:
   - The model’s predictive performance is evaluated using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** metrics.
  
5. **Visualization**:
   - The forecasted closing prices are visualized alongside actual prices for comparative analysis. A specific example using Bitcoin is included to demonstrate the model's performance.
  
6. **Function for Model Application**:
   - A reusable function is implemented to apply the Prophet model across all cryptocurrencies. It automatically calculates and prints the MSE and MAE for each currency.

## Model Evaluation

The model is evaluated based on its performance in predicting the closing prices for each cryptocurrency. The following metrics are calculated:

- **Mean Squared Error (MSE)**: Measures the average of the squared differences between the predicted and actual values.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.

The metrics provide a quantitative evaluation of the model's accuracy for each cryptocurrency.

## How to Run the Project

### Prerequisites
To run this project, you need the following libraries installed in your Python environment:
```bash
pip install pandas matplotlib seaborn prophet
