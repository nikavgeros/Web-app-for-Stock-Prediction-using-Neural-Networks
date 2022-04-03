# Web-app-for-Stock-Prediction-using-Neural-Networks

### Context 
Analyzing time series and predicting their future prices is one of the most important problems facing scientists in various fields, such as health, economic, environmental, and social sciences, business, and industry, as well as investors who want to create a profitable portfolio. The need predicting future prices or events is crucial, as it is a key factor in strategy planning and decision making for both businesses and investors. In recent years, various studies have been conducted on stock prices prediction which end in the proposal of various techniques. Artificial Neural Networks as method of flexible computing are considered as more accurate and widely used predictive models and for that reason, they often find application for solving real world problems.

### Content 
The financial information and the stock prices are available at yfinance package, developed by Ran Aroussi and maintained by himself along with other contributors. It is an open-source tool that uses Yahoo’s publicly available APIs and is intended for reach and educational purposes. The Ticker module allows us to get valuable information on the company such as business summary, recommendations, dividends, splits, major stakeholders, and many more. The stock prices are available in a format that includes the Date, the stock prices such as Open, Close, High, Low, Adj Close and the Volume which refers to the number of traded shares. We specify a date range within five years, beginning from 2016/01/01 till 2021/09/30 containing 1453 historical prices. Most of the features are numerical except from Date feature which refers to the dates of the Stock Market Calendar.

### Goal 
Implementation of Neural Networks with different achritectures and great predictive capability to integrate them into a web-base application for stock price prediction.

### This repository contains the following files:
<ol>
  <li>Images folder, --contains the plots where we compare the actual prices with predicted prices on the training phase:</li>
  <li>Logos folder --contains the logos for each company</li>
  <li>Models folder --contains the neural networks we created which are ready for use.</li>
  <li>Scalers folder --contains all the scalers created on the training phase for each stock</li>
  <li>one_layer_nn.py --builds neural networks with one layer</li>
  <li>two_layer_nn.py --builds neural networks with two layer</li>
  <li>three_layer_nn.py --builds neural networks with three layer</li>
  <li>utils.py --contains all the functions needed for the develepment of neural networks</li>
  <li>stock_prediction_app.py --contains the web application implemented with streamlit</li>
  <li>requirements.txt --contains the versions for the libraries I use and its required to deploye the app</li>
</ol>

### Books
* Chollet F, (2018). Deep Learning with Python, Manning
* Geron A., (2019). Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow of Deep Learning, O’Reilly
* Peter J. Brockwell, Richard A. Davis (2016). Introduction to TimeSeries and Forecasting 3rd Edition, Springer
* Wilfredo Palma (2016). Time Series Analysis, Wiley
---
<i>N.Avgeros (February 2022)</li>
