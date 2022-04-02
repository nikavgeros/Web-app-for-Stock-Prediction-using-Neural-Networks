# LIBRARIES
import streamlit as st
from datetime import date
import pickle
from keras.models import load_model
import pandas as pd
import numpy as np
from PIL import Image
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO


st.set_page_config(layout="wide")
# CONSTANT VARIABLES
MODEL_PATH = "Models"
START = "2016-1-1"
TODAY = date.today().strftime("%Y-%m-%d")
STOCKS =('AAPL', 'AMZN', 'GE', 'GOOGL', 'IBM', 'MSFT', 'TSLA')
COLUMN = ('Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume')
MODEL = ['LSTM', 'GRU', 'SimpleRNN']
TIMESTEPS = 60
IMAGE = Image.open('AI-Series.jpeg')
MENU = ["Home", "Financial Analysis", "Forecast"]


#### Streamlit APP ####

# USER INPUT SIDEBAR
st.sidebar.subheader("Menu")
menu = st.sidebar.selectbox("Select page", MENU)
st.sidebar.subheader("Stocks")
selected_stock = st.sidebar.selectbox("Select stock", STOCKS)

# LOAD DATA FROM YAHOO FINANCE
@st.cache
def load_data(selected_stock, start_date, end_date):
    data = yf.download(selected_stock, start_date, end_date)
    data.index = data.index.strftime('%Y-%m-%d')
    return data

@st.cache
def plot_data(selected_stock):
    data = yf.download(selected_stock, START, TODAY)
    return data

@st.cache(allow_output_mutation=True)
def get_ticker(selected_stock):
    data = yf.Ticker(selected_stock)
    return data


if menu == 'Home':
    # HEADER
    col21, col22, col23 = st.columns([8, 10, 5]) 
    with col21:
        st.write("")
        
    with col22:
        st.title("Stock Trend Forecast")
        
    with col23:
        st.write("")
    
    # Image
    col21, col22, col23 = st.columns([5, 10, 5])#4,8,15
    with col21:
        st.write("")
    with col22:
        st.image(IMAGE, width=800)
    with col23:
        st.write("")
    
    st.write('---')
    st.subheader('About')
    st.markdown(""" 
    This is app uses Deep Learning Algorithms to forecast future stock closing price and its trend in the Forecast page.\n
    It is also provide financial information about the available companies in the Financial Analysis page.
    * **Python Libraries:** Pandas, Numpy, Matplotlib, Keras, Tensorflow, yfinance, streamlit
    * **Data Source:** [Yahoo Finance] (https://finance.yahoo.com/)
    """)
    st.write('---')
    company = get_ticker(selected_stock)

    # STOCK INFORMATION
    col1, col2, col3 = st.columns([11,5,10])
    with col1:
        st.write("")
    with col2:
        stock_logo = Image.open(f'{selected_stock}.png')
        st.image(stock_logo, width=100)
    with col3:
        st.write("")
    st.write(company.info['longBusinessSummary'])
    st.write('---')



elif menu == 'Financial Analysis':

    # DOWNLOAD BUTTONS
    @st.cache
    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    @st.cache
    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name=f'{selected_stock}')
        workbook = writer.book
        worksheet = writer.sheets[f'{selected_stock}']
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)  
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    # SIDEBAR
    selected_column = st.sidebar.selectbox("Select column to visualize", COLUMN)
    start_date = st.sidebar.date_input('Start date', value = pd.to_datetime(START))
    end_date = st.sidebar.date_input('End date', value = pd.to_datetime(TODAY))
    st.sidebar.write("##")


    # STOCK DATAFRAME
    data = load_data(selected_stock, start_date=start_date, end_date=end_date)
    col_df, col_describe = st.columns([10, 10])
    with col_df:
        st.subheader(f"**{selected_stock} Data from Yahoo Finance**")
        st.write(f'Data Dimension: {data.shape[0]} rows and {data.shape[1]} columns')
        st.dataframe(data)
    with col_describe:
        st.subheader(f"**{selected_stock} Data Statistical Information**")
        st.dataframe(data.describe())

    
    data_csv = convert_df(data)
    data_xlsx = to_excel(data)


    col1, col2, col3, col4 = st.columns([2, 5, 8, 20])#2, 5, 7, 10
    with col1:
        st.write("")
    with col2:
        st.download_button(label='📥 Download Excel',
                                    data=data_xlsx,
                                    file_name= f'{selected_stock}.xlsx')
    with col3:
        st.download_button(label='📥 Download CSV',
                                    data=data_csv,
                                    file_name= f'{selected_stock}.csv',
                                    mime='text/csv')
    with col4:
        st.write("")

    st.write('---')

    # LINE CHART 
    plot_data = plot_data(selected_stock)
    st.subheader(f'**{selected_stock} {selected_column} column**')
    st.line_chart(plot_data[selected_column])

    
    st.write('---')
    company = get_ticker(selected_stock)
    st.subheader("News")
    news = pd.DataFrame(company.news)
    news = news[['title', 'publisher', 'link', 'providerPublishTime']]
    st.table(news)
    st.write('---')
    col21, col22 = st.columns((1,1))
    with col21:
        st.subheader("Dividends and Splits")
        st.write(company.actions)
    with col22:
        st.subheader("Major Holders")
        st.write(company.major_holders)
    st.write('---')
    st.subheader("Institutional Holders")
    st.write(company.institutional_holders)
    st.write('---')
    st.subheader("Recommendations")
    st.write(company.recommendations)
    st.write('---')
    st.subheader("Balance Sheet")
    st.write(company.balance_sheet)
else:

    # Artificial Neural Network Parameters
    col1, col2, col3 = st.columns([7,8,7])
    with col1:
        st.write("")
    with col2:
        st.subheader("Artificial Neural Network Parameters")
    with col3:
        st.write("")

    
    # Select Neural Network architecture
    col4, col5, col6 = st.columns((1, 1, 1))
    with col4:
        selected_model = st.selectbox("Select Neural Network architecture", MODEL)
    with col5:
        hidden_layers = st.number_input("Hidden layers:", 1, 3)-1
    with col6:
        n_days = st.slider("Days of prediction:", 1, 60)

    data = load_data(selected_stock, start_date=START, end_date=TODAY)

    # PREDICT BUTTON AND DISPLAY PREDICTIONS
    if st.button('Predict'):

        # LOADING THE MODEL
        local_model = load_model(MODEL_PATH + f'\{selected_stock}_{selected_model}_{hidden_layers}h.h5')
        
        # LOADING THE SCALER 
        scaler = MODEL_PATH + f"\{selected_stock}_Scaler_{hidden_layers}h.pickle"
        local_scaler = pickle.load(open(scaler, 'rb'))

        # CREATING NUMPY ARRAY
        last_timesteps_days = pd.DataFrame(data['Close'].tail(TIMESTEPS))
        last_timesteps_days = last_timesteps_days.values

        # SCALING 
        to_predict_scaled = local_scaler.fit_transform(last_timesteps_days)

        # LOOP TO MAKE N DAYS PREDICTION LIST 
        for day in range(n_days):
            inputs = to_predict_scaled[day:]
            inputs_reshaped = inputs.reshape(1, inputs.shape[0], 1)
            predict = local_model.predict(inputs_reshaped)
            inputs = np.append(inputs, predict[0])
            to_predict_scaled = np.append(to_predict_scaled, predict[0])

        to_predict_scaled = to_predict_scaled.reshape(-1,1)

        predictions_n_days = local_scaler.inverse_transform(to_predict_scaled)
        predictions_n_days = predictions_n_days[-n_days:]
        final_predictions = pd.DataFrame(predictions_n_days, columns=['Predicted Closing Price'], dtype=object)


        
        def plot_predictions_data(dataset):
            df = pd.DataFrame(dataset)
            plt.style.use('fivethirtyeight')
            fig = plt.figure(figsize = (14,4))
            plt.plot(df, color = 'red', label = 'Predicted Closing Price')
            plt.title(f'{selected_stock} Closing Price Prediction')
            plt.xlabel('Days')
            plt.ylabel(f'Closing Price $')
            plt.legend()
            return st.pyplot(fig)

        st.write('---')
        
        col7, col8, col9 = st.columns([7,8,7])
        with col7:
            st.write("")
        with col8:
            st.subheader(f"{selected_model} with {hidden_layers+1} hidden layers")
        with col9:
            st.write("")

        col10, col11 = st.columns((1, 2))
        with col10:
            st.write(f'Predictions for {n_days} days')
            st.write(final_predictions)
        with col11:
            st.write(f"Predicted Closing Price Line Chart for {n_days} days")
            plot_predictions_data(final_predictions) 
        st.write('---')
    else:
        st.write('---')

        
# streamlit run stock_prediction_app.py