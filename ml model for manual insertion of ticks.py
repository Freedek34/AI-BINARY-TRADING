import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to calculate RSI
def calculate_rsi(prices, window=7):
    delta = prices.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 3000 - (100 / (1 + rs))  # Custom RSI calculation
    return rsi

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
rsi_window = 7

# Calculate RSI and MACD
df['RSI'] = calculate_rsi(df['Price'], window=rsi_window)
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Price'], slow=macd_slow_window, fast=macd_fast_window, signal=macd_signal_window)

# Calculate other features like differences, momentum, etc.
df['Price_diff'] = df['Price'].diff()
df['Momentum'] = df['Price'].diff(5)

# Drop rows with NaN values due to rolling window and differencing
df = df.dropna()

# Features and target
X = df[['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Price_diff', 'Momentum']]
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
print(f'Accuracy after tuning: {accuracy:.2f}')

# Output the best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Predict if the next last digit will be even or odd
def predict_next_even_odd(new_price):
    # Extract the last digit of the new price
    last_digit = int(str(int(new_price))[-1])
    
    # Concatenate the new price with the existing price series
    price_series = pd.concat([df['Price'], pd.Series([new_price])], ignore_index=True)
    
    # Calculate features based on the new price
    new_data = pd.DataFrame({
        'Price': [new_price],
        'RSI': [calculate_rsi(price_series, window=rsi_window).iloc[-1]],
        'MACD': [calculate_macd(price_series, slow=macd_slow_window, fast=macd_fast_window, signal=macd_signal_window)[0].iloc[-1]],
        'MACD_Signal': [calculate_macd(price_series, slow=macd_slow_window, fast=macd_fast_window, signal=macd_signal_window)[1].iloc[-1]],
        'MACD_Hist': [calculate_macd(price_series, slow=macd_slow_window, fast=macd_fast_window, signal=macd_signal_window)[2].iloc[-1]],
        'Price_diff': [new_price - df['Price'].iloc[-1]],
        'Momentum': [new_price - df['Price'].iloc[-5]]
    })
    
    # Drop NaNs and scale the new data
    new_data = new_data.dropna()
    new_data_scaled = scaler.transform(new_data[['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Price_diff', 'Momentum']])
    
    # Make prediction for the new data
    prediction = best_rf_model.predict(new_data_scaled)[0]
    return 'Even' if prediction == 1 else 'Odd'

# Example usage with a new price
new_price = 1260.75  # Replace with the actual next price
prediction = predict_next_even_odd(new_price)
print(f'The predicted next last digit is: {prediction}')
