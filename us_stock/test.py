from common.database_client import WindDBClient
import pandas as pd
import numpy as np
import pytz
import datetime as dt
import matplotlib.pyplot as plt
from fredapi import Fred  # https://github.com/mortada/fredapi
import yfinance as yf
import pandas as pd

# Define the ETF tickers
tickers = ["TQQQ", "TMF", "TMV"]

# Fetch data from Yahoo Finance
data = yf.download(tickers, start="2010-01-01", end="2025-01-01", group_by="ticker", auto_adjust=True)

# Display the data for inspection
print(data)

start_date = 20150101
end_date = dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d")

ticker_list = ["TQQQ.OF", "TMF.P", "TMV.P"]
wind_client = WindDBClient()