import sys
import os
import json
import random
import asyncio
import websockets
import pandas as pd
import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from deriv_api import DerivAPI, APIError

# Configuration
APP_ID = 63411
API_TOKEN = os.getenv('DERIV_TOKEN', 'v2aXvKrGdarZo59')
BET_AMOUNT = 400
MARTINGALE_MULTIPLIER = 2
DALEMBERT_STEP = 400
TOTAL_ROUNDS = 50  # Reduced for simplicity
URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
CSV_FILE = 'ticks50.csv'

# Ensure the API token is set
if not API_TOKEN:
    sys.exit("DERIV_TOKEN environment variable is not set")

# Global variables
trade_counter = 0  # Counter to track the number of trades
model, scaler = None, None  # Model and scaler initialized to None

# Function to implement the Berlekamp-Massey Algorithm for binary sequences
def berlekamp_massey_algorithm(binary_sequence):
    """
    Implements Berlekamp-Massey algorithm to find the shortest linear feedback shift register (LFSR) 
    for a given binary sequence.
    """
    n = len(binary_sequence)
    c = [0] * n  # Connection polynomial
    b = [0] * n  # Backup polynomial
    c[0], b[0] = 1, 1
    l, m, i = 0, -1, 0
    
    for i in range(n):
        discrepancy = binary_sequence[i]
        for j in range(1, l + 1):
            discrepancy ^= c[j] & binary_sequence[i - j]
        
        if discrepancy == 1:
            temp = c.copy()
            for j in range(i - m, n):
                if j >= 0:
                    c[j] ^= b[j - (i - m)]
            if 2 * l <= i:
                l = i + 1 - l
                m = i
                b = temp
        
    return c[:l + 1], l  # Return the connection polynomial and length of the LFSR

# Example Usage:
# binary_sequence = [1, 0, 1, 1, 0, 1, 0, 0]  # Binary sequence of even/odd or other indicator
# connection_poly, lfsr_length = berlekamp_massey_algorithm(binary_sequence)
# Use `lfsr_length` as a feature in ML model

# Function to extract the binary indicator from price data
def extract_binary_indicator(prices):
    """
    Extract binary sequence based on even/odd classification of last digit.
    """
    binary_sequence = [1 if int(str(price)[-1]) % 2 == 0 else 0 for price in prices]
    return binary_sequence

# Function to add the result of Berlekamp's algorithm as a feature
def add_berlekamp_feature(df, price_column='Price'):
    """
    Adds the LFSR length obtained from Berlekamp-Massey algorithm as a feature to the dataframe.
    """
    binary_sequence = extract_binary_indicator(df[price_column].tolist())
    _, lfsr_length = berlekamp_massey_algorithm(binary_sequence)
    
    # Add the LFSR length as a new feature
    df['LFSR_Length'] = lfsr_length
    
    return df
# Function to calculate the Average Directional Index (ADX)
def calculate_adx(df, window_size=14):
    df['High'] = df['Price']  # Mock high prices for now (replace with actual)
    df['Low'] = df['Price']  # Mock low prices for now (replace with actual)
    df['Close'] = df['Price']  # Mock close prices for now (replace with actual)
    
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    df['+DI'] = 100 * (df['+DM'] / df['TR']).ewm(span=window_size, adjust=False).mean()
    df['-DI'] = 100 * (df['-DM'] / df['TR']).ewm(span=window_size, adjust=False).mean()
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).fillna(0)
    df['ADX'] = df['DX'].ewm(span=window_size, adjust=False).mean()
    
    return df['ADX']

# Function to calculate MACD
def calculate_macd(prices, short_window=1, long_window=10, signal_window=1):
    """
    Calculate the MACD (Moving Average Convergence Divergence) and Signal Line.
    
    Parameters:
    - prices: A pandas Series of prices.
    - short_window: The window size for the short-term EMA (default is 12).
    - long_window: The window size for the long-term EMA (default is 26).
    - signal_window: The window size for the signal line EMA (default is 9).
    
    Returns:
    - macd: The MACD line (difference between short-term and long-term EMAs).
    - signal_line: The signal line (9-period EMA of MACD).
    - macd_histogram: The MACD histogram (difference between MACD and Signal Line).
    """
    # Calculate the short-term EMA (12-period)
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    
    # Calculate the long-term EMA (26-period)
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    
    # MACD Line: Short-term EMA - Long-term EMA
    macd = short_ema - long_ema
    
    # Signal Line: 9-period EMA of MACD
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    
    # MACD Histogram: Difference between MACD and Signal Line
    macd_histogram = macd - signal_line
    
    return macd, signal_line, macd_histogram

# Example usage:
# df['MACD'], df['Signal_Line'], df['MACD_Histogram'] = calculate_macd(df['Price'])



# Function to calculate Standard Deviation
def calculate_standard_deviation(prices, window_size=14):
    return prices.rolling(window=window_size).std()

# Function to calculate the Exponential Moving Average (EMA)
def calculate_ema(prices, window_size):
    return prices.ewm(span=window_size, adjust=False).mean()

# Function to calculate the Bollinger Bands
def calculate_bollinger_bands(prices, window_size=10, num_std_dev=1):
    """
    Calculate Bollinger Bands: upper, middle, and lower bands.
    """
    middle_band = prices.rolling(window=window_size).mean()
    std_dev = prices.rolling(window=window_size).std()
    upper_band = middle_band + (num_std_dev * std_dev)
    lower_band = middle_band - (num_std_dev * std_dev)
    return upper_band, middle_band, lower_band

def calculate_atr(df, window_size=7):
    df['High_Low'] = df['Price'].rolling(window=window_size).apply(lambda x: max(x) - min(x))
    df['ATR'] = df['High_Low'].ewm(span=window_size, adjust=False).mean()
    return df['ATR']

# Function to calculate the Envelope
def calculate_envelope(prices, window_size=1, deviation=0.02):
    """
    Calculate Envelope bands based on a given moving average and deviation percentage.
    """
    ema = calculate_ema(prices, window_size)  # Use EMA as the base moving average
    upper_band = ema * (1 + deviation)  # Deviation is a percentage (e.g., 0.02 for 2%)
    lower_band = ema * (1 - deviation)
    return upper_band, ema, lower_band


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Load or train the ML model with hyperparameter tuning
def train_ml_model():
    global model, scaler  # Update the global model and scaler

    # Load the full tick data from CSV
    if not os.path.exists(CSV_FILE):
        print(f"{CSV_FILE} not found. Collecting initial data.")
        return None, None
    
    df = pd.read_csv(CSV_FILE, header=None, names=['Price'])

    def is_even(number):
        return number % 2 == 0

    # Extract last digit for even/odd classification
    df['Last_Digit'] = df['Price'].apply(lambda x: int(str(x)[-1]))
    df['Even'] = df['Last_Digit'].apply(is_even)
    
    # Modify this value to change the sharpness of the EMA
    ema_window_size = 3  # Set the window size for the EMA

    # Calculate EMA and Bollinger Bands as features
    df['EMA'] = calculate_ema(df['Price'], window_size=ema_window_size)
    df['Price_diff'] = df['Price'].diff()
    df['Momentum'] = df['Price'].diff(5)
    df['SMA'], df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df['Price'], window_size=10, num_std_dev=1)
    df['ATR'] = calculate_atr(df, window_size=14)
    df['ADX'] = calculate_adx(df, window_size=14)  # Add ADX calculation
    df['Std_Dev'] = calculate_standard_deviation(df['Price'], window_size=14)  # Add Standard Deviation
    df['Envelope_Upper'], df['Envelope_Middle'], df['Envelope_Lower'] = calculate_envelope(df['Price'], window_size=1, deviation=0.02)# Add Envelope bands as features
    df['MACD'], df['Signal_Line'], df['MACD_Histogram'] = calculate_macd(df['Price'])
    df = add_berlekamp_feature(df, price_column='Price')
    
    df = df.dropna()  # Drop rows with NaN values resulting from rolling calculations

    # Check if there's enough data to train the model
    if df.empty:
        print("Not enough data for training. Waiting for more tick data.")
        return None, None

    # Define the features and labels for training
    X = df[['EMA', 'Price_diff', 'Momentum', 'SMA', 'Upper_Band', 'Lower_Band', 'ATR', 'ADX', 'Std_Dev', 'Envelope_Upper', 'Envelope_Middle', 'Envelope_Lower','MACD','Signal_Line','LFSR_Length']]  # Features (now using EMA and Bollinger Bands)
    y = df['Even']  # Target (even/odd classification)

    # Debugging: Print a preview of the training data
    print("Training data preview:")
    print(df.tail())  # Show the last few rows of the DataFrame

    # Scale the features for the model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the model and the hyperparameter space
    rf_model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Use GridSearchCV or RandomizedSearchCV for hyperparameter tuning
    search = GridSearchCV(rf_model, param_grid, cv=3, verbose=2, n_jobs=-1)  # For Grid Search
    # search = RandomizedSearchCV(rf_model, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1)  # For Randomized Search
    
    search.fit(X_scaled, y)
    
    # Best model after hyperparameter tuning
    model = search.best_estimator_

    print(f"Best parameters found: {search.best_params_}")
    print("Model training with hyperparameter tuning completed.")
    
    return model, scaler

# Function to handle retraining the model after every 3 trades
def check_and_retrain_model():
    global trade_counter
    trade_counter += 1
    print(f"Trade counter: {trade_counter}")
    
    if trade_counter >= 3:
        print("Retraining the model after 3 trades...")
        model, scaler = train_ml_model()  # Retrain the model
        trade_counter = 0  # Reset trade counter after retraining

# Function to ensure price formatting
def format_price(price):
    return f"{price:.4f}"  # Ensure two decimal places (e.g., 1234.5 -> 1234.50)

# Function to collect and store tick data
async def collect_data():
    async with websockets.connect(URL) as websocket:
        await websocket.send(json.dumps({
            "ticks_history": "R_50",
            "adjust_start_time": 1,
            "count": 50,
            "end": "latest",
            "start": 1,
            "style": "ticks"
        }))

        response = await websocket.recv()
        data = json.loads(response)

        if 'history' in data and 'prices' in data['history']:
            prices = data['history']['prices']

            # Format prices and print them
            formatted_prices = [format_price(float(price)) for price in prices]
            print(f"Formatted prices: {formatted_prices[-5:]}")  # Print last 5 formatted prices

            # Extract last digit of formatted prices
            digits = [int(str(price)[-1]) for price in formatted_prices]

            # Append the formatted tick data to the CSV file
            pd.DataFrame(formatted_prices).to_csv(CSV_FILE, mode='a', header=False, index=False)
            
            print(f"Extracted last digits: {digits[-5:]}")  # Debugging: Show last 5 extracted digits

            return digits
        else:
            print("Failed to retrieve data")
            return None

# Function to place a trade and manage post-trade actions
async def place_trade(prediction):
    global BET_AMOUNT

    # Mockup of placing a trade using prediction (even=1, odd=0)
    contract_type = "EVEN" if prediction == 1 else "ODD"
    print(f"Placing trade on: {contract_type} with amount: {BET_AMOUNT}")

    # Simulating trade result (mockup for now)
    trade_result = random.choice([True, False])  # Simulate a win or loss
    print(f"Trade result: {'Win' if trade_result else 'Loss'}")
    
    # Handle betting strategies based on the result
    if trade_result:
        BET_AMOUNT = max(BET_AMOUNT - DALEMBERT_STEP, 100)  # Reduce bet size after a win
    else:
        BET_AMOUNT *= MARTINGALE_MULTIPLIER  # Apply Martingale strategy after a loss

    check_and_retrain_model()  # Check if it's time to retrain the model

# Make predictions using the machine learning model
def predict_even_odd(prices):
    global model, scaler

    print(f"Received prices for prediction: {prices[-5:]}")  # Debugging: print last 5 prices

    if model is None or scaler is None:
        model, scaler = train_ml_model()
        if model is None or scaler is None:
            print("Using random fallback due to untrained model.")
            return random.randint(0, 1)

    # Convert to DataFrame and extract features
    df = pd.DataFrame(prices, columns=['Price'])
    
    # Modify this value to change the sharpness of the EMA
    ema_window_size = 1  # Set the window size for the EMA

    df['EMA'] = calculate_ema(df['Price'], window_size=ema_window_size)
    df['Price_diff'] = df['Price'].diff()
    df['Momentum'] = df['Price'].diff(5)
    df['SMA'], df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df['Price'], window_size=10, num_std_dev=1)
    df['ATR'] = calculate_atr(df, window_size=7)
    df['ADX'] = calculate_adx(df, window_size=7)  # Add ADX calculation for prediction
    df['Std_Dev'] = calculate_standard_deviation(df['Price'], window_size=7)  # Add Standard Deviation
    df['Envelope_Upper'], df['Envelope_Middle'], df['Envelope_Lower'] = calculate_envelope(df['Price'], window_size=1, deviation=0.02)# Add Envelope bands as features
    df['MACD'], df['Signal_Line'], df['MACD_Histogram'] = calculate_macd(df['Price'])
    df = add_berlekamp_feature(df, price_column='Price')

    # Drop NaNs and scale the data
    df = df.dropna()

    print("Data after feature engineering (last 5 rows):")
    print(df.tail())  # Debugging: Print last few rows after feature calculation

    if df.empty:
        print("Not enough data for prediction. Using random fallback.")
        return random.randint(0, 1)

    X = df[['EMA', 'Price_diff', 'Momentum', 'SMA', 'Upper_Band', 'Lower_Band', 'ATR', 'ADX', 'Std_Dev', 'Envelope_Upper', 'Envelope_Middle', 'Envelope_Lower','MACD','Signal_Line','LFSR_Length']]
    X_scaled = scaler.transform(X)

    # Make prediction for the last row of data
    prediction = model.predict(X_scaled[-1].reshape(1, -1))[0]
    print(f"Prediction made by model: {'Even' if prediction == 1 else 'Odd'}")
    return prediction

# Main bot logic with Martingale and D'Alembert strategies
async def trading_bot():
    api = DerivAPI(app_id=APP_ID)
    await api.authorize(API_TOKEN)

    balance = await api.balance()
    initial_balance = balance['balance']['balance']
    current_bet_amount = BET_AMOUNT
    dalembert_level = 0

    for round_num in range(1, TOTAL_ROUNDS + 1):
        await asyncio.sleep(-99)

        # Collect tick data and update ticks.csv
        prices = await collect_data()
        if prices is None:
            continue

        # Predict even or odd using the ML model
        predicted_even_odd = predict_even_odd(prices)
        print(f"Predicted even/odd: {'Even' if predicted_even_odd == 1 else 'Odd'}")

        # Define contract type based on prediction
        contract_type = "DIGITEVEN" if predicted_even_odd == 1 else "DIGITODD"
        print(f"Round {round_num}: Placing {contract_type} trade with bet amount {current_bet_amount}")

        try:
            # Convert current_bet_amount to float and validate
            current_bet_amount = float(current_bet_amount)
        except ValueError:
            print(f"Invalid bet amount: {current_bet_amount}")
            continue

        # Ensure the bet amount is not below the minimum allowed bet
        MIN_BET_AMOUNT = 0.35  # Minimum bet
        if current_bet_amount < MIN_BET_AMOUNT:
            print(f"Bet amount {current_bet_amount} is below the minimum allowed bet. Adjusting to {MIN_BET_AMOUNT}.")
            current_bet_amount = MIN_BET_AMOUNT

        # Debugging: Print the current bet amount before sending the proposal
        print(f"Sending proposal request with amount: {current_bet_amount}")

        # Get proposal for trade
        try:
            proposal = await api.proposal({
                "proposal": 1,
                "amount": current_bet_amount,
                "barrier": "0",
                "basis": "stake",  # Change from payout to stake
                "contract_type": contract_type,
                "currency": "USD",
                "duration": 1,
                "duration_unit": "t",
                "symbol": "R_100"
            })
            
            proposal_id = proposal.get('proposal', {}).get('id')
            if not proposal_id:
                print("Failed to get proposal")
                continue
        except APIError as e:
            print(f"Failed to get proposal: {e}")
            continue

        try:
            # Place trade
            buy_response = await api.buy({"buy": proposal_id, "price": current_bet_amount})
            contract_id = buy_response.get('buy', {}).get('contract_id')
            if not contract_id:
                print("Failed to get contract ID")
                continue
        except APIError as e:
            print(f"Failed to buy: {e}")
            continue

        # Wait for trade to complete
        await asyncio.sleep(1)

        # Check profit/loss from the profit_table API
        try:
            profit_table = await api.profit_table({"profit_table": 1, "limit": 1})
            if profit_table and 'profit_table' in profit_table and profit_table['profit_table'].get('transactions'):
                last_trade = profit_table['profit_table']['transactions'][0]
                sell_price = last_trade.get('sell_price', 0)
                print(f"Sell price: {sell_price}")

                profit = sell_price - current_bet_amount
                print(f"Profit from last trade: {profit}")

                # Martingale & D'Alembert strategy logic
                if sell_price == 0:  # Loss
                    print("Loss encountered. Applying Martingale.")
                    current_bet_amount *= MARTINGALE_MULTIPLIER  # Double the bet
                    dalembert_level += 1
                else:  # Win
                    print("Win encountered. Applying D'Alembert.")
                    if dalembert_level > 0:
                        current_bet_amount -= DALEMBERT_STEP
                        dalembert_level -= 1

                        # Ensure bet doesn't go below the minimum
                        if current_bet_amount < MIN_BET_AMOUNT:
                            print(f"Bet amount after D'Alembert adjustment is too low. Adjusting to {MIN_BET_AMOUNT}.")
                            current_bet_amount = MIN_BET_AMOUNT
                    else:
                        current_bet_amount = BET_AMOUNT  # Reset to initial bet
            else:
                print("No transaction data found")
        except APIError as e:
            print(f"Failed to get profit_table: {e}")
            continue

    print("Trading session completed")

# Scheduler setup
def start_scheduler():
    scheduler = AsyncIOScheduler()

    # Schedule the bot to run at a specific time daily (e.g., 10:00 AM)
    scheduler.add_job(trading_bot, 'cron', hour=16, minute=21)
    # Start the scheduler
    scheduler.start()
    print("Scheduler started. Bot will run at 17:03 daily.")

# Start scheduler
if __name__ == "__main__":
    start_scheduler()

    # Keep the script running to allow scheduled jobs to execute
    asyncio.get_event_loop().run_forever()
