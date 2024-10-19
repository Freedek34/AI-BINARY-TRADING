import sys
import os
import json
import random
import asyncio
import websockets
import pandas as pd
import numpy as np
import random 
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from deriv_api import DerivAPI, APIError

# Configuration
APP_ID = 63411
API_TOKEN = os.getenv('DERIV_TOKEN', 'v2aXvKrGdarZo59')
BET_AMOUNT = 200
MARTINGALE_MULTIPLIER = 2
DALEMBERT_STEP = 200
TOTAL_ROUNDS = 50  # Reduced for simplicity
URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
CSV_FILE = 'ticks100.csv'

# Ensure the API token is set
if not API_TOKEN:
    sys.exit("DERIV_TOKEN environment variable is not set")

# Global variables
trade_counter = 0  # Counter to track the number of trades
model, scaler = None, None  # Model and scaler initialized to None

def calculate_support_resistance_pivot(df, period=3):
    # Calculate Support and Resistance
    df['Support'] = df['Price'].rolling(window=period, min_periods=1).min()
    df['Resistance'] = df['Price'].rolling(window=period, min_periods=1).max()

    # Calculate Pivot Points
    df['Pivot_Point'] = (df['Resistance'] + df['Support'] + df['Price']) / 3
    
    return df


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
def calculate_adx(df, window_size=5):
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

# Function to calculate RSI with adjustments based on price last digit
def calculate_adjusted_rsi(prices, window=5):
    # Standard RSI Calculation
    delta = prices.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Adjust RSI based on the last digit of the price
    adjusted_rsi = []
    
    for i in range(len(prices)):
        price_last_digit = round(prices.iloc[i], 2) * 100 % 10  # Extract the last digit of the price
        rsi_value = rsi.iloc[i]
        
        if pd.notna(rsi_value):  # Only proceed if rsi_value is not NaN
            rounded_rsi = round(rsi_value, 0)  # Round RSI to nearest integer

            # Ensure RSI is even when price's last digit is even, and odd when it's odd
            if price_last_digit % 2 == 0:  # Even last digit price
                if rounded_rsi % 2 != 0:  # If RSI is odd, make it even
                    rounded_rsi += 1
            else:  # Odd last digit price
                if rounded_rsi % 2 == 0:  # If RSI is even, make it odd
                    rounded_rsi += 1

            adjusted_rsi.append(rounded_rsi)
        else:
            adjusted_rsi.append(np.nan)  # Preserve NaN values in the output

    return pd.Series(adjusted_rsi, index=prices.index)

# Function to calculate CCI with adjustments based on price last digit
def calculate_adjusted_cci(prices, window=7):
    # Standard CCI calculation
    typical_price = prices  # Using the price directly as 'typical price'
    
    sma = typical_price.rolling(window=window).mean()  # Simple moving average
    mad = (typical_price - sma).abs().rolling(window=window).mean()  # Mean absolute deviation
    cci = (typical_price - sma) / (0.015 * mad)  # CCI formula
    
    # Adjust CCI based on the last digit of the price
    adjusted_cci = []
    
    for i in range(len(prices)):
        price_last_digit = round(prices.iloc[i], 2) * 100 % 10  # Extract the last digit of the price
        cci_value = cci.iloc[i]
        
        if pd.notna(cci_value):  # Only proceed if cci_value is not NaN
            rounded_cci = round(cci_value, 0)  # Round CCI to nearest integer

            # Ensure CCI is even when price's last digit is even, and odd when it's odd
            if price_last_digit % 2 == 0:  # Even last digit price
                if rounded_cci % 2 != 0:  # If CCI is odd, make it even
                    rounded_cci += 1
            else:  # Odd last digit price
                if rounded_cci % 2 == 0:  # If CCI is even, make it odd
                    rounded_cci += 1

            adjusted_cci.append(rounded_cci)
        else:
            adjusted_cci.append(np.nan)  # Preserve NaN values in the output

    return pd.Series(adjusted_cci, index=prices.index)

class HeatSeekingBox:
    def __init__(self, initial_box_size=1.0, min_box_size=0.1, max_box_size=5.0):
        self.box_size = initial_box_size  # Initial size of the box (prediction range)
        self.min_box_size = min_box_size  # Minimum box size (precision when accurate)
        self.max_box_size = max_box_size  # Maximum box size (expand when inaccurate)
        self.accuracy_count = 0  # Track correct guesses

    def update_box_size(self, was_correct):
        """ Adjust the size of the box based on accuracy """
        if was_correct:
            # If correct, shrink the box for more precision
            self.box_size = max(self.min_box_size, self.box_size * 0.9)
            self.accuracy_count += 1
        else:
            # If incorrect, expand the box for more tolerance
            self.box_size = min(self.max_box_size, self.box_size * 1.1)
            self.accuracy_count = 0  # Reset the accuracy streak

    def get_midpoint(self, price):
        """ Calculate the midpoint based on the current price and box size """
        lower_bound = price - (self.box_size / 2)
        upper_bound = price + (self.box_size / 2)
        midpoint = (lower_bound + upper_bound) / 2
        return midpoint

    def guess_even_or_odd(self, price):
        """ Use the midpoint to guess if the next price will be even or odd """
        midpoint = self.get_midpoint(price)
        rounded_midpoint = round(midpoint)  # Round to nearest integer
        return rounded_midpoint % 2 == 0  # Return True for even, False for odd

    def run_eyeball(self, current_price):
        """ Perform a guess and update the box based on whether it was correct """
        predicted_even = self.guess_even_or_odd(current_price)
        
        # Simulate next price tick
        next_price = self.simulate_next_price(current_price)
        actual_even = round(next_price) % 2 == 0
        
        # Check if the prediction was correct
        was_correct = predicted_even == actual_even
        
        # Update the box size based on the accuracy of the prediction
        self.update_box_size(was_correct)
        
        return predicted_even, actual_even, was_correct, next_price

    def simulate_next_price(self, current_price):
        """ Simulate the next price using randomness (this would be fetched in real trading) """
        return current_price + random.uniform(-self.box_size, self.box_size)

# Assuming you've already defined your HeatSeekingBox class as earlier.

def calculate_heat_seeking_features(df):
    box = HeatSeekingBox()
    midpoints = []
    box_sizes = []

    for price in df['Price']:
        midpoint = box.get_midpoint(price)
        midpoints.append(midpoint)
        box_sizes.append(box.box_size)

        # Simulate next price to update the box size
        # In real use-case, replace `simulate_next_price` with live data fetching
        next_price = box.simulate_next_price(price)
        was_correct = (round(next_price) % 2 == 0) == (round(midpoint) % 2 == 0)
        box.update_box_size(was_correct)

    df['Eyeball_Midpoint'] = midpoints
    df['Box_Size'] = box_sizes
    return df

# Function to calculate MACD
def calculate_macd(prices, short_window=1, long_window=2, signal_window=1):
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
    
    return macd, signal_line

# Example usage:
def get_last_digit(price):
    """Extracts the last digit of the price and converts it to binary (even -> 2, odd -> 1)."""
    last_digit = abs(int(str(price)[-1]))  # Extract last digit, ensure non-negative number
    if last_digit == 0:
        return 2  # 0 is even
    else:
        return 2 if last_digit % 2 == 0 else 1  # 2 for even, 1 for odd



def update_binary_indicator(prices):
    """
    Updates binary indicator based on price changes and converts the last digit
    of the price into both a binary value (even -> 0, odd -> 1) and its complement (even -> 1, odd -> 0).
    Returns both the binary indicator and its complement.
    """
    if len(prices) == 0:
        raise ValueError("The price list is empty.")
    
    binary_indicator = [0]  # Starting value of the binary indicator
    binary_complement = []  # List for the complement (inverse of binary value)
    binary_values = []  # List for storing binary values for each price

    for i in range(len(prices)):
        current_price = prices[i]
        current_binary = get_last_digit(current_price)
        current_complement = 1 - current_binary  # Correct complement calculation (inverse)

        if i > 0:
            previous_price = prices[i - 1]
            if current_price > previous_price:
                binary_indicator.append(binary_indicator[-1] + current_binary)
            else:
                binary_indicator.append(binary_indicator[-1] - current_binary)
        else:
            binary_indicator[0] = current_binary

        binary_complement.append(current_complement)
        binary_values.append(current_binary)

    return binary_indicator, binary_values

# Function to calculate Standard Deviation
def calculate_standard_deviation(prices, window_size=5):
    return prices.rolling(window=window_size).std()

# Function to calculate the Exponential Moving Average (EMA)
def calculate_ema(prices, window_size):
    return prices.ewm(span=window_size, adjust=False).mean()

# Function to calculate the Bollinger Bands
def calculate_bollinger_bands(prices, window_size=2, num_std_dev=1):
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
def calculate_envelope(prices, window_size=3, deviation=0.02):
    """
    Calculate Envelope bands based on a given moving average and deviation percentage.
    """
    ema = calculate_ema(prices, window_size)  # Use EMA as the base moving average
    upper_band = ema * (1 + deviation)  # Deviation is a percentage (e.g., 0.02 for 2%)
    lower_band = ema * (1 - deviation)
    return upper_band, ema, lower_band

def calculate_fibonacci_retracement_levels(prices):
    # Check if the prices array has enough data points
    if len(prices) < 2:
        raise ValueError("Not enough price data to calculate Fibonacci levels")

    # Calculate the minimum and maximum prices
    max_price = prices.max()
    min_price = prices.min()

    # Calculate the Fibonacci levels
    difference = max_price - min_price
    level_0 = min_price
    level_236 = min_price + difference * 0.236
    level_382 = min_price + difference * 0.382
    level_500 = min_price + difference * 0.500
    level_618 = min_price + difference * 0.618
    level_100 = max_price

    # Return the Fibonacci levels as a dictionary
    return {
        'Level_0': level_0,
        'Level_236': level_236,
        'Level_382': level_382,
        'Level_500': level_500,
        'Level_618': level_618,
        'Level_100': level_100
    }


def calculate_trend_line(prices):
    x = np.arange(len(prices))  # Generate x-values
    y = prices.values  # Get y-values from the prices DataFrame

    # Fit a linear regression line (y = mx + b)
    coefficients = np.polyfit(x, y, 1)
    trend_line = np.polyval(coefficients, x)

    return trend_line

def calculate_quantile_levels(prices):
    q1 = prices.quantile(0.25)
    q2 = prices.quantile(0.50)  # Median
    q3 = prices.quantile(0.75)

    levels = {
        'Q1': q1,
        'Q2': q2,
        'Q3': q3
    }

    return levels


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
    df['Envelope_Upper'], df['Envelope_Middle'], df['Envelope_Lower'] = calculate_envelope(df['Price'], window_size=5, deviation=0.02)# Add Envelope bands as features
    df['MACD'], df['Signal_Line']= calculate_macd(df['Price'])
    df = add_berlekamp_feature(df, price_column='Price')
    df['Binary_Indicator'], df['Binary_Value'] = update_binary_indicator(df['Price'].tolist())
    df['Adjusted_RSI'] = calculate_adjusted_rsi(df['Price'], window=7)
    df = calculate_heat_seeking_features(df)
    df['Adjusted_CCI'] = calculate_adjusted_cci(df['Price'], window=14)
    period = 3 # Set this to your desired period for support/resistance calculations
    df = calculate_support_resistance_pivot(df, period)
    # Calculate Fibonacci retracement levels
    fibonacci_levels = calculate_fibonacci_retracement_levels(df['Price'])
    for level_name, level_value in fibonacci_levels.items():
        df[level_name] = level_value

    # Calculate quantile levels
    quantile_levels = calculate_quantile_levels(df['Price'])
    for level_name, level_value in quantile_levels.items():
        df[level_name] = level_value

    # Calculate trend line and add it to the DataFrame
    df['Trend_Line'] = calculate_trend_line(df['Price'])




    df = df.dropna()  # Drop rows with NaN values resulting from rolling calculations

    # Check if there's enough data to train the model
    if df.empty:
        print("Not enough data for training. Waiting for more tick data.")
        return None, None

    # Define the features and labels for training
    X = df[['Price', 'Support', 'Resistance', 'Pivot_Point','EMA', 'Price_diff', 'Momentum', 'SMA', 'Upper_Band', 'Lower_Band', 'ATR', 'ADX', 'Std_Dev', 'Envelope_Upper', 'Envelope_Middle', 'Envelope_Lower','MACD','Signal_Line','LFSR_Length','Binary_Indicator','Binary_Value','Adjusted_RSI','Eyeball_Midpoint', 'Box_Size', 'Adjusted_CCI','Level_0','Level_236','Level_382','Level_500','Level_618','Level_100','Q1','Q2','Q3','Trend_Line']]  # Features (now using EMA and Bollinger Bands)
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
    return f"{price:.2f}"  # Ensure two decimal places (e.g., 1234.5 -> 1234.50)

# Function to collect and store tick data
async def collect_data():
    async with websockets.connect(URL) as websocket:
        await websocket.send(json.dumps({
            "ticks_history": "R_100",
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
    df['MACD'], df['Signal_Line'] = calculate_macd(df['Price'])
    df = add_berlekamp_feature(df, price_column='Price')
    df['Binary_Indicator'], df['Binary_Value'] = update_binary_indicator(df['Price'].tolist())
    df['Adjusted_RSI'] = calculate_adjusted_rsi(df['Price'], window=7)
    df = calculate_heat_seeking_features(df)
    df['Adjusted_CCI'] = calculate_adjusted_cci(df['Price'], window=14)
    period = 3  # Set this to your desired period for support/resistance calculations
    df = calculate_support_resistance_pivot(df, period)
    # Calculate Fibonacci retracement levels
    fibonacci_levels = calculate_fibonacci_retracement_levels(df['Price'])
    for level_name, level_value in fibonacci_levels.items():
        df[level_name] = level_value

    # Calculate quantile levels
    quantile_levels = calculate_quantile_levels(df['Price'])
    for level_name, level_value in quantile_levels.items():
        df[level_name] = level_value

    # Calculate trend line and add it to the DataFrame
    df['Trend_Line'] = calculate_trend_line(df['Price'])

    # Drop NaNs and scale the data
    df = df.dropna()

    print("Data after feature engineering (last 5 rows):")
    print(df.tail())  # Debugging: Print last few rows after feature calculation

    if df.empty:
        print("Not enough data for prediction. Using random fallback.")
        return random.randint(0, 1)

    X = df[['Price', 'Support', 'Resistance', 'Pivot_Point','EMA', 'Price_diff', 'Momentum', 'SMA', 'Upper_Band', 'Lower_Band', 'ATR', 'ADX', 'Std_Dev', 'Envelope_Upper', 'Envelope_Middle', 'Envelope_Lower','MACD','Signal_Line','LFSR_Length','Binary_Indicator','Binary_Value','Adjusted_RSI','Eyeball_Midpoint', 'Box_Size', 'Adjusted_CCI','Level_0','Level_236','Level_382','Level_500','Level_618','Level_100','Q1','Q2','Q3','Trend_Line']]
    X_scaled = scaler.transform(X)

    # Make prediction for the last row of data
    prediction = model.predict(X_scaled[-1].reshape(1, -1))[0]
    print(f"Prediction made by model: {'Even' if prediction == 1 else 'Odd'}")
    return prediction


async def trading_bot():
    api = DerivAPI(app_id=APP_ID)
    await api.authorize(API_TOKEN)
    await asyncio.sleep(1)

    balance = await api.balance()
    initial_balance = balance['balance']['balance']
    current_bet_amount = BET_AMOUNT
    dalembert_level = 0

    for round_num in range(1, TOTAL_ROUNDS + 1):
        await asyncio.sleep(1)

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

        # Wait for a random time before placing the next trade
        random_wait_time = random.uniform(1, 3)  # Random wait time between 1 to 5 seconds
        print(f"Waiting for {random_wait_time:.2f} seconds before the next trade...")
        await asyncio.sleep(random_wait_time)

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
    scheduler.add_job(trading_bot, 'cron', hour=19 , minute=1)
    # Start the scheduler
    scheduler.start()
    print("Scheduler started. Bot will run at 17:03 daily.")

# Start scheduler
if __name__ == "__main__":
    start_scheduler()

    # Keep the script running to allow scheduled jobs to execute
    asyncio.get_event_loop().run_forever()
