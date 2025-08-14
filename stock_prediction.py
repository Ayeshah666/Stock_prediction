import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Select a stock and fetch historical data
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Prepare the data for prediction
def prepare_data(data):
    """
    Prepare data for machine learning:
    - Use today's Open, High, Low, Volume to predict tomorrow's Close
    """
    df = data.copy()
    df['Target'] = df['Close'].shift(-1)  # Tomorrow's closing price
    df = df.dropna()  # Remove rows with NaN values
    
    features = ['Open', 'High', 'Low', 'Volume']
    X = df[features]
    y = df['Target']
    
    return X, y, df

# Step 3: Train and evaluate models
def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate Linear Regression and Random Forest models
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'r2': r2
        }
    
    return results, y_test, X_test.index

# Step 4: Plot actual vs predicted prices
def plot_results(y_test, predictions, dates, model_name, ticker):
    """
    Plot actual vs predicted closing prices
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test, label='Actual Price', color='blue')
    plt.plot(dates, predictions, label='Predicted Price', color='red', linestyle='--')
    plt.title(f'{model_name} - Actual vs Predicted Closing Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Configuration
    TICKER = 'AAPL'  # Apple stock
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    
    print(f"Fetching data for {TICKER} from {START_DATE} to {END_DATE}...")
    stock_data = fetch_stock_data(TICKER, START_DATE, END_DATE)
    
    print("\nPreparing data for modeling...")
    X, y, df = prepare_data(stock_data)
    
    print("\nTraining and evaluating models...")
    results, y_test, test_dates = train_and_evaluate(X, y)
    
    # Display results and plots
    for model_name, result in results.items():
        print(f"\n{model_name} Results:")
        print(f"MSE: {result['mse']:.4f}")
        print(f"R² Score: {result['r2']:.4f}")
        
        plot_results(
            y_test, 
            result['predictions'], 
            test_dates, 
            model_name, 
            TICKER
        )