# Step-by-Step Explanation
# Data Collection:

-We use the yfinance library to fetch historical stock data for Apple (AAPL) from 2020-2021

-The data includes Open, High, Low, Close prices and Volume

# Data Preparation:

-We create a target variable that represents the next day's closing price

-Features used are today's Open, High, Low, and Volume

-We split the data into training and testing sets (80/20 split)

# Model Training:

-We train two models:

-Linear Regression (simple baseline)

-Random Forest (more complex, often better for financial data)

-We scale the features using StandardScaler

# Evaluation:

-We calculate Mean Squared Error (MSE) and R² score for both models

-We plot the actual vs predicted prices for visual comparison

# Visualization:

-The script generates plots comparing actual and predicted prices for both models

# How to Use This Solution
Install required packages:

bash
pip install yfinance numpy pandas scikit-learn matplotlib
Run the script:

Save the code to a file (e.g., stock_prediction.py)

Run it with Python: python stock_prediction.py

# Customize:

Change TICKER to analyze different stocks (e.g., 'TSLA' for Tesla)

Adjust date ranges as needed

Modify model parameters (e.g., number of trees in Random Forest)

# Expected Output
The script will:

Print evaluation metrics for both models

Display two plots comparing actual vs predicted prices
