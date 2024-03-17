import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from yahoo_fin.stock_info import get_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

today = datetime.now().strftime("%d/%m/%Y")
print(today, type(today))

st.title("Analysis and Predict Stock")

def load_data(code):
    data = get_data(code, start_date="01/01/2015", end_date=today, index_as_date = True, interval="1d")
    return data

def sma(data, n):
    return sum(data[-n:]) / n

def ema(data,n):
    sma_value = sma(data,n)
    multiplier = 2/(n+1)
    ema_value = (data[-1] - sma_value) * multiplier + sma_value
    return ema_value

def rsi(data,n):
    delta = np.diff(data)
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    avg_gain = np.mean(gains[:n])
    avg_loss = -np.mean(losses[:n])
    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))
    return rsi

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data.ewm(span=short_period, adjust=False).mean()
    long_ema = data.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


col1, col2 = st.columns([1,4])
with col1:
    code = st.radio(
        "Chọn mã chứng khoán",
        ["MSFT","AAPL","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO"],
    )
with col2:
    data = load_data(code)
    data.dropna(inplace=True)
    data_use = data.iloc[-30:]
    start_day = data_use.index[0].strftime("%d/%m/%Y")
    end_day = data_use.index[-1].strftime("%d/%m/%Y")
    data_use['Date'] = data_use.index.to_list()
    
    sma_value = round(sma(data['close'], 50),2)
    ema_value = round(ema(data["close"], 14),2)
    rsi_value = round(rsi(data["close"], 30),2)
    macd_line, signal_line, histogram = calculate_macd(data["close"])
    col2_21, col2_22, col2_23 = st.columns(3)
    with col2_21:
        st.subheader("SMA 50")
        st.write(sma_value)
    with col2_22:
        st.subheader("EMA 14")
        st.write(ema_value)
    with col2_23:
        st.subheader("RSI 30")
        st.write(rsi_value)

    fig = go.Figure()
    fig = go.Figure(data=[go.Candlestick(x=data_use['Date'],
            open=data_use['open'],
            high=data_use['high'],
            low=data_use['low'],
            close=data_use['close'])])
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=350,
        margin=dict(t=0, b=0, l=0, r=0),
        )# Xóa các margin để làm sát biên
    st.plotly_chart(fig)
    st.write(f'The data get from {start_day} to {end_day}')

st.subheader("MACD, SIGNAL LINE AND HISTOGRAM")
index = [i for i in range(1,len(macd_line)+1)]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=index, y=macd_line, mode='lines', name='MACD Line',line=dict(color='green', width=2)))
fig2.add_trace(go.Scatter(x=index, y=signal_line, mode='lines', name='SIGNAL Line', line=dict(color='red', width=2)))
fig2.add_trace(go.Bar(x=index, y=histogram, name='Histogram'))

fig2.update_layout(
                  xaxis_title='Date',
                  yaxis_title='Value')

# Hiển thị biểu đồ trong Streamlit
st.plotly_chart(fig2)

st.subheader("Predict the stock price of the next 30 days")
data["pre_close"] = data['close'].shift()
data["pre_open"] = data['open'].shift()
data.dropna(inplace=True)

X_predictClose = data[["pre_close","open"]].values
y_predictClose = data['close'].values
X_predictOpen = data[["pre_close","pre_open"]].values
y_predictOpen = data['open'].values
predicts = []

def pre_open(X_predictOpen, y_predictOpen):
    X_train, X_test, y_train, y_test = train_test_split(X_predictOpen, y_predictOpen, random_state=42, test_size=1/3)
    ln = LinearRegression().fit(X_train, y_train)
    x = np.asarray([[y_predictClose[-1], y_predictOpen[-1]]])
    y_pred = ln.predict(x)
    X_predictOpen = np.vstack((X_predictOpen,x))
    y_predictOpen = np.append(y_predictOpen, y_pred)

def pre_close(X_predictClose, y_predictClose):
    X_train, X_test, y_train, y_test = train_test_split(X_predictClose, y_predictClose, random_state=42, test_size=1/3)
    ln = LinearRegression().fit(X_train, y_train)
    x = np.asarray([[y_predictClose[-2], y_predictOpen[-1]]])
    y_pred = ln.predict(x)
    X_predictClose = np.vstack((X_predictClose,x))
    y_predictClose = np.append(y_predictClose, y_pred)

for i in range(30):
    pre_open(X_predictOpen, y_predictOpen)
    pre_close(X_predictClose, y_predictClose)

y_predictClose = np.round(y_predictClose, 2)
start_date = data_use.index[-1]
days = [start_date + timedelta(days=x) for x in range(1, 31)]
str_days = []
for i in range(len(days)):
    str_days.append(days[i].strftime("%d/%m/%Y"))

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=str_days, y=y_predictClose[-30:],
                          mode='lines+markers+text', 
                          text=y_predictClose[-30:],
                          name='Price close',
                          line=dict(color='green', width=4),
                          marker=dict(color='red', size=6)))
st.plotly_chart(fig3)


    