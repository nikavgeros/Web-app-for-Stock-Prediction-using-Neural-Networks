# Web-app-for-Stock-Prediction-using-Neural-Networks

### Context 
Analyzing time series and predicting their future prices is one of the most important problems facing scientists in various fields, such as health, economic, environmental, and social sciences, business, and industry, as well as investors who want to create a profitable portfolio. The need predicting future prices or events is crucial, as it is a key factor in strategy planning and decision making for both businesses and investors. In recent years, various studies have been conducted on stock prices prediction which end in the proposal of various techniques. Artificial Neural Networks as method of flexible computing are considered as more accurate and widely used predictive models and for that reason, they often find application for solving real world problems.

### Content 
The financial information and the stock prices are available at yfinance package, developed by Ran Aroussi and maintained by himself along with other contributors. It is an open-source tool that uses Yahoo’s publicly available APIs and is intended for reach and educational purposes. The Ticker module allows us to get valuable information on the company such as business summary, recommendations, dividends, splits, major stakeholders, and many more. The stock prices are available in a format that includes the Date, the stock prices such as Open, Close, High, Low, Adj Close and the Volume which refers to the number of traded shares. We specify a date range within five years, beginning from 2016/01/01 till 2021/09/30 containing 1453 historical prices. Most of the features are numerical except from Date feature which refers to the dates of the Stock Market Calendar.

### Goal 
implementation of Neural Networks with different achritectures and great predictive capability to integrate them into a web-base application for stock price prediction.

### This repository contains the following files:
<ol>
  <li>Restaurant Customer Reviews.ipynb, -- the main jupyter notebook of this project, containing 6 parts:</li>
  <ul>
    <li>Data collection and cleaning</li>
    <li>Explanatory Data Analysis on reviews</li>
    <li>Various model fitting</li>
    <li>Models evaluation</li>
  </ul>
  <li>Restaurant_Reviews.tsv --the dataset that contains 1000 observations of reviews, and likes</li>
  <li>final_results.csv --the predictions from the models</li>
  <li>CountVectorizer --the converter of text documents to a matrix of token counts in binary file</li>
  <li>RandomForestClassifier --the best model in binary file</li>
</ol>

### References
* Paul Deitel, Dr.Harvey Deitel - Python for Programmers_ with Big Data and Artificial Intelligence Case Studies (2019, Pearson Higher Ed)
* Aurélien Géron - Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (2017, O'Reilly )

---
<i>N.Avgeros (February 2022)</li>
