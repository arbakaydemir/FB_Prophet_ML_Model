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

```bash
pip install pandas numpy fbprophet matplotlib seaborn plotly scikit-learn
