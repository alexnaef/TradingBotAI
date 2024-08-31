# Import necessary libraries and modules
from lumibot.brokers import Alpaca  # Broker for trading via Alpaca
from lumibot.backtesting import YahooDataBacktesting  # For backtesting with Yahoo Finance data
from lumibot.strategies.strategy import Strategy  # Base class for creating trading strategies
from lumibot.traders import Trader  # Class for running trading strategies
from datetime import datetime  # For handling date and time
from alpaca_trade_api import REST  # Alpaca's REST API for fetching data and placing orders
from timedelta import Timedelta  # For handling time deltas
from finbert_utils import estimate_sentiment  # Utility for estimating sentiment from news data

# API credentials for connecting to Alpaca
API_KEY = "PKPAKJB1ZGX3RT2Z2UN7" 
API_SECRET = "0EUerdnBeHoidtkxGihCjhyxXhkdHqnUmThw7FLu" 
BASE_URL = "https://paper-api.alpaca.markets"

trading_ticker = "SPY"
banchmark_ticker = "^RUT"

# Store Alpaca credentials in a dictionary
ALPACA_CREDS = {
    "API_KEY": API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True  # Indicates that this is a paper trading account
}

# Define the MLTrader class that inherits from the Strategy base class
class MLTrader(Strategy):
    
    # Initialization method for setting up the strategy
    def initialize(self, symbol: str = trading_ticker, cash_at_risk: float = .5): 
        self.symbol = symbol  # Trading symbol (e.g., "IWM")
        self.sleeptime = "24H"  # Time between trading iterations
        self.last_trade = None  # Keep track of the last trade (buy/sell)
        self.cash_at_risk = cash_at_risk  # Percentage of cash to risk in a trade
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)  # Connect to Alpaca's REST API

    # Method to calculate position sizing based on available cash
    def position_sizing(self): 
        cash = self.get_cash()  # Get available cash
        last_price = self.get_last_price(self.symbol)  # Get the latest price of the trading symbol
        quantity = round(cash * self.cash_at_risk / last_price, 0)  # Calculate the number of shares to trade
        return cash, last_price, quantity  # Return cash, last price, and calculated quantity

    # Method to get the current date and the date three days prior
    def get_dates(self): 
        today = self.get_datetime()  # Get the current date and time
        three_days_prior = today - Timedelta(days=3)  # Calculate the date three days ago
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')  # Return formatted dates

    # Method to estimate sentiment from news articles
    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()  # Get today’s date and three days ago date
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today)  # Fetch news for the symbol between the specified dates
        news = [ev.__dict__["_raw"]["headline"] for ev in news]  # Extract headlines from news events
        probability, sentiment = estimate_sentiment(news)  # Estimate sentiment using FinBERT
        return probability, sentiment  # Return sentiment probability and sentiment (positive/negative)

    # Method that runs on each trading iteration
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()  # Get position sizing information
        probability, sentiment = self.get_sentiment()  # Get sentiment information

        if cash > last_price:  # Ensure there is enough cash to buy at least one share
            if sentiment == "positive" and probability > .999:  # Check if sentiment is positive with high probability
                if self.last_trade == "sell":  # If the last trade was a sell, sell all current positions
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "buy",  # Place a buy order
                    type="bracket",  # Use a bracket order to manage risk
                    take_profit_price=last_price * 1.20,  # Set take profit price 20% above the last price
                    stop_loss_price=last_price * .95  # Set stop loss price 5% below the last price
                )
                self.submit_order(order)  # Submit the order
                self.last_trade = "buy"  # Update last trade to buy
            elif sentiment == "negative" and probability > .999:  # Check if sentiment is negative with high probability
                if self.last_trade == "buy":  # If the last trade was a buy, sell all current positions
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell",  # Place a sell order
                    type="bracket",  # Use a bracket order to manage risk
                    take_profit_price=last_price * .8,  # Set take profit price 20% below the last price
                    stop_loss_price=last_price * 1.05  # Set stop loss price 5% above the last price
                )
                self.submit_order(order)  # Submit the order
                self.last_trade = "sell"  # Update last trade to sell

# Set the backtesting period
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 6, 1) 

# Create an Alpaca broker instance using the credentials
broker = Alpaca(ALPACA_CREDS) 

# Initialize the MLTrader strategy
strategy = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol": trading_ticker, 
                                "cash_at_risk": .5})

# Backtest the strategy using Yahoo Finance data
strategy.backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, 
    benchmark_asset= banchmark_ticker,
    parameters={"symbol": trading_ticker, "cash_at_risk": .5}
)

# Uncomment the lines below to run the strategy live with a trader
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
