# gold_prediction_system

# Gold Price Tracker & Forecaster 🪙📈

An automated Python-based tool designed to track daily gold prices and generate short-term price predictions using time-series forecasting.

## Overview
This project automates the extraction of daily gold futures data (GC=F) from Yahoo Finance, maintains a historical dataset, and utilizes **Facebook Prophet** to forecast the next day's price. Built with a focus on robust data manipulation and statistical forecasting, it serves as a practical application of data science concepts to financial market trends.

## Key Features
* **Automated Data Retrieval:** Fetches real-time closing prices for gold via the `yfinance` API, ensuring data is always up-to-date.
* **Data Persistence:** Seamlessly updates and maintains a local CSV database (`altin_gunluk_veri.csv`) to build a comprehensive historical dataset over time.
* **Predictive Modeling:** Implements the `Prophet` library to analyze trends and weekly seasonality, generating a prediction for the next trading day's closing price along with an 80% confidence interval.
* **Accuracy Tracking:** Includes a built-in evaluation module to calculate the mean prediction error of the last 30 days, allowing for continuous monitoring of model performance.

## Technologies & Libraries
* **Python 3**
* **Pandas & NumPy:** For data manipulation, cleaning, and statistical calculations.
* **Prophet:** For robust time-series forecasting.
* **yfinance:** For reliable financial market data extraction.

## Installation & Usage

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)

   
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
