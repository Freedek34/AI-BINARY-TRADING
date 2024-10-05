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

# Function to subscribe to tick data and print it continuously
async def stream_ticks():
    async with websockets.connect(URL) as websocket:
        # Subscribe to real-time tick data
        await websocket.send(json.dumps({
            "ticks": "R_50",  # Replace "R_100" with the appropriate market if necessary
            "subscribe": 1
        }))
        
        # Keep receiving tick data continuously
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            if 'tick' in data and 'quote' in data['tick']:
                tick_price = data['tick']['quote']
                formatted_price = f"{float(tick_price):.4f}"
                print(f"Received Tick: {formatted_price}")
                append_to_csv([formatted_price])
            else:
                print("Error: No tick data in response")

# Main function to start the WebSocket connection and print ticks
async def main():
    await stream_ticks()

# Start the WebSocket tick stream
if __name__ == "__main__":
    asyncio.run(main())
