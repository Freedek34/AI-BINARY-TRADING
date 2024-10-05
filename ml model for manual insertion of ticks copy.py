import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler


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


# Function to calculate MACD
def calculate_macd(prices, slow=1, fast=1, signal=1):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

# Read ticks from CSV
df = pd.read_csv('ticks100.csv', header=None, names=['Price'])

# Convert the 'Price' column to numeric to avoid the error
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop rows with NaN values (if there are non-numeric prices)
df = df.dropna(subset=['Price'])

# Extract the last digit of the prices
df['Last_Digit'] = df['Price'].apply(lambda x: int(str(int(x))[-1]))  # Convert to int, then extract the last digit

# Define a function to determine if the last digit is even or odd
def is_even(number):
    return number % 2 == 0

df['Even'] = df['Last_Digit'].apply(is_even)

# Define window parameters for MACD and RSI
macd_slow_window = 1
macd_fast_window = 1
macd_signal_window = 2
ema_window_size = 3  # Set the window size for the EMA

# Calculate RSI and MACD
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Price'], slow=macd_slow_window, fast=macd_fast_window, signal=macd_signal_window)
df['EMA'] = calculate_ema(df['Price'], window_size=ema_window_size)
df['Price_diff'] = df['Price'].diff()
df['Momentum'] = df['Price'].diff(5)
df['SMA'], df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df['Price'], window_size=20, num_std_dev=2)
df['ATR'] = calculate_atr(df, window_size=14)
df['ADX'] = calculate_adx(df, window_size=14)  # Add ADX calculation
df['Std_Dev'] = calculate_standard_deviation(df['Price'], window_size=14)  # Add Standard Deviation
df['Envelope_Upper'], df['Envelope_Middle'], df['Envelope_Lower'] = calculate_envelope(df['Price'], window_size=1, deviation=0.02)# Add Envelope bands as features
df = add_berlekamp_feature(df, price_column='Price')

    

# Calculate other features like differences, momentum, etc.
df['Price_diff'] = df['Price'].diff()
df['Momentum'] = df['Price'].diff(5)

# Drop rows with NaN values due to rolling window and differencing
df = df.dropna()

# Features and target
X = df[['MACD', 'MACD_Signal', 'MACD_Hist', 'Price_diff', 'Momentum','EMA', 'SMA', 'Upper_Band', 'Lower_Band', 'ATR', 'ADX', 'Std_Dev', 'Envelope_Upper', 'Envelope_Middle', 'Envelope_Lower','LFSR_Length']]
y = df['Even']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for RandomForest using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier()

# Perform Grid Search with Cross Validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model after tuning
best_rf_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_rf_model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Generate a detailed classification report
report = classification_report(y_test, y_pred, target_names=['Odd', 'Even'])

# Print evaluation metrics
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'\nClassification Report:\n{report}')
print(f'Accuracy after tuning: {accuracy:.2f}')

# Output the best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Predict if the next last digit will be even or odd
def predict_next_even_odd(new_price):
    # Extract the last digit of the new price
    last_digit = int(str(int(new_price))[-1])
    
    # Concatenate the new price with the existing price series
    price_series = pd.concat([df['Price'], pd.Series([new_price])], ignore_index=True)
    
    
    # Drop NaNs and scale the new data
    new_data = new_data.dropna()
    new_data_scaled = scaler.transform(new_data[['MACD', 'MACD_Signal', 'MACD_Hist', 'EMA', 'Price_diff', 'Momentum', 'SMA', 'Upper_Band', 'Lower_Band', 'ATR', 'ADX', 'Std_Dev', 'Envelope_Upper', 'Envelope_Middle', 'Envelope_Lower','LFSR_Length']])
    
    # Make prediction for the new data
    prediction = best_rf_model.predict(new_data_scaled)[0]
    return 'Even' if prediction == 1 else 'Odd'

# Example usage with a new price
new_price = 1260.75  # Replace with the actual next price
prediction = predict_next_even_odd(new_price)
print(f'The predicted next last digit is: {prediction}')
