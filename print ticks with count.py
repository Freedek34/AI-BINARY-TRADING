import json
import asyncio
import websockets
import os
import sys
import csv

# Configuration
APP_ID = 63411
API_TOKEN = os.getenv('DERIV_TOKEN', 'v2aXvKrGdarZo59')
URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
CSV_FILE = "ticks50.csv"  # File to store ticks

# Ensure the API token is set
if not API_TOKEN:
    sys.exit("DERIV_TOKEN environment variable is not set")

# Function to append tick data to a CSV file
def append_to_csv(prices):
    file_exists = os.path.isfile(CSV_FILE)
    
    # Open the CSV file in append mode
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writerow(['Price'])
        
        # Format prices to two decimal places and write them to the CSV
        for price in prices:
            formatted_price = f"{float(price):.4f}"  # Format price to two decimals
            writer.writerow([formatted_price])

# Function to collect historical tick data
async def collect_data():
    async with websockets.connect(URL) as websocket:
        # Request historical tick data
        await websocket.send(json.dumps({
            "ticks_history": "R_50",
            "adjust_start_time": 1,
            "count": 1000,  # Collect 40 ticks
            "end": "latest",
            "start": 1,
            "style": "ticks"
        }))
        
        response = await websocket.recv()
        data = json.loads(response)

        if 'history' in data and 'prices' in data['history']:
            prices = data['history']['prices']
            print(f"Received {len(prices)} ticks, appending to {CSV_FILE}...")
            append_to_csv(prices)
        else:
            print("Failed to retrieve data")

# Main function to start the data collection
async def main():
    await collect_data()

# Start the data collection
if __name__ == "__main__":
    asyncio.run(main())
