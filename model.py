import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from config import data_base_path
import random
import requests
import retrying

forecast_price = {}

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
MAX_DATA_SIZE = 100  # Giá»›i háº¡n sá»‘ lÆ°á»£ng dá»¯ liá»‡u tá»‘i Ä‘a khi lÆ°u trá»¯
INITIAL_FETCH_SIZE = 100  # Sá»‘ lÆ°á»£ng náº¿n láº§n Ä‘áº§u táº£i vá»

@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_prices(symbol, interval="1m", limit=100, start_time=None, end_time=None):
    try:
        base_url = "https://fapi.binance.com"
        endpoint = f"/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        url = base_url + endpoint
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f'Failed to fetch prices for {symbol} from Binance API: {str(e)}')
        raise e

def download_data(token):
    symbols = f"{token.upper()}USDT"
    interval = "5m"
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())
    
    # ÄÆ°á»ng dáº«n file CSV Ä‘á»ƒ lÆ°u trá»¯
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")
    # file_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")

    # Kiá»ƒm tra xem file cÃ³ tá»“n táº¡i hay khÃ´ng
    if os.path.exists(file_path):
        # TÃ­nh thá»i gian báº¯t Ä‘áº§u cho 100 cÃ¢y náº¿n 5 phÃºt
        start_time = int((current_datetime - timedelta(minutes=500)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, 100, start_time, end_time)
    else:
        # Náº¿u file khÃ´ng tá»“n táº¡i, táº£i vá» sá»‘ lÆ°á»£ng INITIAL_FETCH_SIZE náº¿n
        start_time = int((current_datetime - timedelta(minutes=INITIAL_FETCH_SIZE*5)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, INITIAL_FETCH_SIZE, start_time, end_time)

    # Chuyá»ƒn dá»¯ liá»‡u thÃ nh DataFrame
    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])

    # Kiá»ƒm tra vÃ  Ä‘á»c dá»¯ liá»‡u cÅ© náº¿u tá»“n táº¡i
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        # Káº¿t há»£p dá»¯ liá»‡u cÅ© vÃ  má»›i
        combined_df = pd.concat([old_df, new_df])
        # Loáº¡i bá» cÃ¡c báº£n ghi trÃ¹ng láº·p dá»±a trÃªn 'start_time'
        combined_df = combined_df.drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    # Giá»›i háº¡n sá»‘ lÆ°á»£ng dá»¯ liá»‡u tá»‘i Ä‘a
    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    # LÆ°u dá»¯ liá»‡u Ä‘Ã£ káº¿t há»£p vÃ o file CSV
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")

def format_data(token):
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_5m_data.csv")

    if not os.path.exists(file_path):
        print(f"No data file found for {token}")
        return

    df = pd.read_csv(file_path)

    # Sá»­ dá»¥ng cÃ¡c cá»™t sau (Ä‘Ãºng vá»›i dá»¯ liá»‡u báº¡n Ä‘Ã£ lÆ°u)
    columns_to_use = [
        "start_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]

    # Kiá»ƒm tra náº¿u táº¥t cáº£ cÃ¡c cá»™t cáº§n thiáº¿t tá»“n táº¡i trong DataFrame
    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df.columns = [
            "start_time", "open", "high", "low", "close", "volume",
            "end_time", "quote_asset_volume", "n_trades", 
            "taker_volume", "taker_volume_usd"
        ]
        df.index = pd.to_datetime(df["start_time"], unit='ms')
        df.index.name = "date"

        output_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
        df.sort_index().to_csv(output_path)
        print(f"Formatted data saved to {output_path}")
    else:
        print(f"Required columns are missing in {file_path}. Skipping this file.")

def train_model(token):
    # Hiá»ƒn thá»‹ thá»i gian dá»± Ä‘oÃ¡n hiá»‡n táº¡i
    time_start = datetime.now()

    # Load the token price data
    price_data = pd.read_csv(os.path.join(data_base_path, f"{token.lower()}_price_data.csv"))
    df = pd.DataFrame()

    # Convert 'date' to datetime
    price_data["date"] = pd.to_datetime(price_data["date"])

    # Set the date column as the index for resampling
    price_data.set_index("date", inplace=True)

    # Resample the data to 20-minute frequency and compute the mean price
    df = price_data.resample('20T').mean()

    # Prepare data for Linear Regression
    df = df.dropna()  # Loáº¡i bá» cÃ¡c giÃ¡ trá»‹ NaN (náº¿u cÃ³)
    X = np.array(range(len(df))).reshape(-1, 1)  # Sá»­ dá»¥ng chá»‰ sá»‘ thá»i gian lÃ m Ä‘áº·c trÆ°ng
    y = df['close'].values  # Sá»­ dá»¥ng giÃ¡ Ä‘Ã³ng cá»­a lÃ m má»¥c tiÃªu

    # Khá»Ÿi táº¡o mÃ´ hÃ¬nh Linear Regression
    model = LinearRegression()
    model.fit(X, y)  # Huáº¥n luyá»‡n mÃ´ hÃ¬nh

    # Dá»± Ä‘oÃ¡n giÃ¡ tiáº¿p theo
    next_time_index = np.array([[len(df)]])  # GiÃ¡ trá»‹ thá»i gian tiáº¿p theo
    predicted_price = model.predict(next_time_index)[0]  # Dá»± Ä‘oÃ¡n giÃ¡

    # XÃ¡c Ä‘á»‹nh khoáº£ng dao Ä‘á»™ng xung quanh giÃ¡ dá»± Ä‘oÃ¡n
    fluctuation_range = 0.001 * predicted_price  # Láº¥y 0.1% cá»§a giÃ¡ dá»± Ä‘oÃ¡n lÃ m khoáº£ng dao Ä‘á»™ng
    min_price = predicted_price - fluctuation_range
    max_price = predicted_price + fluctuation_range

    # Chá»n ngáº«u nhiÃªn má»™t giÃ¡ trá»‹ trong khoáº£ng dao Ä‘á»™ng
    price_predict = random.uniform(min_price, max_price)
    # gia_tri_2 = round(random.uniform(min_price, max_price), 2)
    # gia_tri_3 = round(random.uniform(min_price, max_price), 2)
    # print(f"{gia_tri_1} - {gia_tri_2} - {gia_tri_3}")

    forecast_price[token] = price_predict

    print(f"Predicted_price: {predicted_price}, Min_price: {min_price}, Max_price: {max_price}")
    print(f"Forecasted price for {token}: {forecast_price[token]}")

    time_end = datetime.now()
    print(f"Time elapsed forecast: {time_end - time_start}")

def update_data():
    # tokens = ["ETH"]
    tokens = ["ETH", "BNB", "ARB"]
    for token in tokens:
        download_data(token)
        format_data(token)
        train_model(token)

if __name__ == "__main__":
    update_data()
