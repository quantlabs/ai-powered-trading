import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
from scipy.stats import norm
import os
import threading
import yfinance as yf
import json
import logging
import argparse

# Set up logging
logging.basicConfig(
    filename='trading_engine.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Trading Engine with Yahoo Finance data')
parser.add_argument('--headless', action='store_true', help='Run in headless mode without UI')
parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
args = parser.parse_args()

# Load configuration
try:
    with open(args.config, 'r') as f:
        config = json.load(f)
    logging.info(f"Configuration loaded from {args.config}")
except FileNotFoundError:
    logging.warning(f"Configuration file {args.config} not found. Using default values.")
    config = {
        "refresh_rate": 10,
        "initial_portfolio_value": 2500,
        "max_positions_per_instrument": 3,
        "risk_free_rate": 0.05,
        "si_volatility": 0.25,
        "cc_volatility": 0.30,
        "data_cache_minutes": 15
    }

# Class to manage application state
class AppState:
    def __init__(self):
        self.refresh_rate = config.get("refresh_rate", 10)
        self.last_refresh = datetime.now()
        self.auto_refresh = True
        self.portfolio_value = config.get("initial_portfolio_value", 2500)
        self.order_book = {'SI': {'bids': [], 'asks': []}, 'CC': {'bids': [], 'asks': []}}
        self.trades = {'SI': [], 'CC': []}
        self.positions = {'SI': [], 'CC': []}
        self.pnl_history = {'SI': [0], 'CC': [0], 'portfolio': [self.portfolio_value]}
        self.price_history = {'SI': [], 'CC': []}
        self.timestamps = []
        self.current_view = "market_overview"  # Default view
        
        # Data caching
        self.data_cache = {}
        self.data_cache_expiry = {}
        self.data_cache_minutes = config.get("data_cache_minutes", 15)
        
        # Initialize with some data
        self.initialize_data()
    
    def initialize_data(self):
        """Initialize with historical data from Yahoo Finance"""
        try:
            # Fetch initial data
            self.fetch_market_data('SI')
            self.fetch_market_data('CC')
            logging.info("Initial market data fetched successfully")
        except Exception as e:
            logging.error(f"Error fetching initial market data: {e}")
            # Fall back to simulated data if fetching fails
            self.initialize_simulated_data()
    
    def initialize_simulated_data(self):
        """Initialize with simulated data as fallback"""
        logging.warning("Initializing with simulated data as fallback")
        now = datetime.now()
        self.timestamps = [(now - timedelta(days=30-i)).strftime("%Y-%m-%d") for i in range(30)]
        
        # Silver typically trades around $20-25
        self.price_history['SI'] = [20.5 + random.uniform(-0.2, 0.2) for _ in range(30)]
        
        # Cocoa typically trades around $3500-4000
        self.price_history['CC'] = [3850 + random.uniform(-10, 10) for _ in range(30)]
    
    def fetch_market_data(self, ticker):
        """Fetch market data from Yahoo Finance with caching"""
        yahoo_ticker = 'SI=F' if ticker == 'SI' else 'CC=F'
        
        # Check if we have cached data that's still valid
        now = datetime.now()
        if (yahoo_ticker in self.data_cache and 
            yahoo_ticker in self.data_cache_expiry and 
            now < self.data_cache_expiry[yahoo_ticker]):
            return self.data_cache[yahoo_ticker]
        
        # Fetch new data
        try:
            logging.info(f"Fetching data for {yahoo_ticker}")
            data = yf.download(
                yahoo_ticker, 
                period="1mo",  # Get 1 month of data
                interval="1d",  # Daily data
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {yahoo_ticker}")
            
            # Cache the data
            self.data_cache[yahoo_ticker] = data
            self.data_cache_expiry[yahoo_ticker] = now + timedelta(minutes=self.data_cache_minutes)
            
            # Update price history
            if ticker == 'SI':
                self.price_history['SI'] = data['Close'].tolist()
                
                # Add most recent timestamps
                if len(self.timestamps) < len(data.index):
                    self.timestamps = [d.strftime('%Y-%m-%d') for d in data.index]
            elif ticker == 'CC':
                self.price_history['CC'] = data['Close'].tolist()
            
            return data
        
        except Exception as e:
            logging.error(f"Error fetching data for {yahoo_ticker}: {e}")
            # Return None to indicate error
            return None

# Initialize application state
state = AppState()

# Helper functions
def calculate_option_premium(spot, strike, days_to_expiry, volatility, risk_free_rate=0.05, option_type='put'):
    """Calculate option premium using Black-Scholes model"""
    days_to_expiry_years = days_to_expiry / 365
    
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * days_to_expiry_years) / (volatility * np.sqrt(days_to_expiry_years))
    d2 = d1 - volatility * np.sqrt(days_to_expiry_years)
    
    if option_type == 'call':
        premium = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * days_to_expiry_years) * norm.cdf(d2)
    else:  # put
        premium = strike * np.exp(-risk_free_rate * days_to_expiry_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    return premium

def get_current_price(ticker):
    """Get current price for the given ticker"""
    # Try to get the latest price from Yahoo Finance first
    try:
        yahoo_ticker = 'SI=F' if ticker == 'SI' else 'CC=F'
        data = yf.Ticker(yahoo_ticker).history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
    except Exception as e:
        logging.warning(f"Could not get real-time price for {ticker}: {e}")
    
    # Fall back to our stored price history
    if state.price_history[ticker]:
        return state.price_history[ticker][-1]
    else:
        # Last resort fallback
        return 20.5 if ticker == 'SI' else 3850

def update_prices():
    """Update prices with real market data"""
    for ticker in ['SI', 'CC']:
        try:
            # Fetch latest data
            data = state.fetch_market_data(ticker)
            
            if data is not None:
                # Get the latest price
                current_price = data['Close'].iloc[-1]
                
                # Add to price history if it's new
                if not state.price_history[ticker] or current_price != state.price_history[ticker][-1]:
                    state.price_history[ticker].append(current_price)
                    
                    # Keep only the latest 500 prices
                    if len(state.price_history[ticker]) > 500:
                        state.price_history[ticker] = state.price_history[ticker][-500:]
                    
                    # Update timestamp
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    state.timestamps.append(now_str)
                    if len(state.timestamps) > 500:
                        state.timestamps = state.timestamps[-500:]
                    
                    logging.info(f"Updated {ticker} price to {current_price}")
            else:
                # Fall back to simulated price updates
                last_price = state.price_history[ticker][-1] if state.price_history[ticker] else (20.5 if ticker == 'SI' else 3850)
                new_price = generate_simulated_price(ticker, last_price)
                state.price_history[ticker].append(new_price)
                
                # Keep only the latest 500 prices
                if len(state.price_history[ticker]) > 500:
                    state.price_history[ticker] = state.price_history[ticker][-500:]
                
                # Update timestamp
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                state.timestamps.append(now_str)
                if len(state.timestamps) > 500:
                    state.timestamps = state.timestamps[-500:]
                
                logging.info(f"Updated {ticker} price to {new_price} (simulated)")
                
        except Exception as e:
            logging.error(f"Error updating price for {ticker}: {e}")

def generate_simulated_price(ticker, last_price):
    """Generate a simulated price (used as fallback)"""
    if ticker == 'SI':
        # Silver has lower volatility
        change = random.uniform(-0.3, 0.3)
        # Add a slight bearish bias for the bear put strategy
        change -= 0.05
    else:  # CC
        # Cocoa has higher volatility
        change = random.uniform(-20, 20)
        # Add a slight bearish bias for the bear put strategy
        change -= 3
    
    return max(0.01, last_price + change)

def generate_order_book(ticker):
    """Generate a simulated order book for the given ticker"""
    current_price = get_current_price(ticker)
    
    # Clear existing order book
    state.order_book[ticker]['bids'] = []
    state.order_book[ticker]['asks'] = []
    
    # Number of orders on each side
    num_orders = random.randint(5, 15)
    
    # Generate bid orders (slightly below current price)
    for i in range(num_orders):
        price_offset = random.uniform(0.5, 5) if ticker == 'SI' else random.uniform(10, 100)
        price = current_price - price_offset * (i + 1) / num_orders
        size = random.randint(1, 10)
        state.order_book[ticker]['bids'].append({'price': price, 'size': size})
    
    # Generate ask orders (slightly above current price)
    for i in range(num_orders):
        price_offset = random.uniform(0.5, 5) if ticker == 'SI' else random.uniform(10, 100)
        price = current_price + price_offset * (i + 1) / num_orders
        size = random.randint(1, 10)
        state.order_book[ticker]['asks'].append({'price': price, 'size': size})
    
    # Sort the order book
    state.order_book[ticker]['bids'].sort(key=lambda x: x['price'], reverse=True)
    state.order_book[ticker]['asks'].sort(key=lambda x: x['price'])

def execute_bear_put_strategy(ticker):
    """Execute a bear put strategy for the given ticker"""
    current_price = get_current_price(ticker)
    
    # Set up strikes based on the ticker
    if ticker == 'SI':
        # Silver bear put
        short_strike = current_price * 0.98  # Short put slightly out of the money
        long_strike = current_price * 0.95   # Long put further out of the money
        days_to_expiry = 30
        volatility = config.get("si_volatility", 0.25)
    else:  # CC
        # Cocoa bear put
        short_strike = current_price * 0.97
        long_strike = current_price * 0.94
        days_to_expiry = 30
        volatility = config.get("cc_volatility", 0.30)
    
    # Calculate option premiums
    risk_free_rate = config.get("risk_free_rate", 0.05)
    short_put_premium = calculate_option_premium(current_price, short_strike, days_to_expiry, volatility, risk_free_rate, option_type='put')
    long_put_premium = calculate_option_premium(current_price, long_strike, days_to_expiry, volatility, risk_free_rate, option_type='put')
    
    # Net credit received from the bear put spread
    net_credit = short_put_premium - long_put_premium
    
    # Execute the trade
    trade = {
        'timestamp': datetime.now(),
        'type': 'bear_put_spread',
        'entry_price': current_price,
        'short_strike': short_strike,
        'long_strike': long_strike,
        'short_premium': short_put_premium,
        'long_premium': long_put_premium,
        'net_credit': net_credit,
        'days_to_expiry': days_to_expiry,
        'status': 'open'
    }
    
    # Add to positions
    state.positions[ticker].append(trade)
    
    # Add to trades history
    state.trades[ticker].append({
        'timestamp': datetime.now(),
        'action': 'entry',
        'price': current_price,
        'strategy': 'bear_put_spread',
        'details': f"Short {ticker} put @ {short_strike:.2f}, Long {ticker} put @ {long_strike:.2f}, Net credit: {net_credit:.2f}"
    })
    
    log_message = f"Executed {ticker} bear put spread at ${current_price:.2f}"
    print(log_message)
    logging.info(log_message)
    print(f"Short put strike: ${short_strike:.2f}, Long put strike: ${long_strike:.2f}")
    print(f"Net credit received: ${net_credit:.2f}")
    
    return trade

def calculate_portfolio_metrics():
    """Calculate current portfolio metrics including open positions value"""
    portfolio_value = state.portfolio_value
    open_positions_value = 0
    
    for ticker in ['SI', 'CC']:
        for position in state.positions[ticker]:
            if position['status'] == 'open':
                # Calculate current value of the position
                current_price = get_current_price(ticker)
                days_remaining = position['days_to_expiry'] - (datetime.now() - position['timestamp']).days
                days_remaining = max(0, days_remaining)
                
                if days_remaining == 0:
                    # Option expired
                    if current_price <= position['long_strike']:
                        # Maximum loss scenario
                        pnl = position['net_credit'] - (position['short_strike'] - position['long_strike'])
                    elif current_price <= position['short_strike']:
                        # Partial loss
                        pnl = position['net_credit'] - (position['short_strike'] - current_price)
                    else:
                        # Maximum profit scenario
                        pnl = position['net_credit']
                    
                    position_value = pnl
                else:
                    # Position still open, calculate current value
                    volatility = config.get("si_volatility", 0.25) if ticker == 'SI' else config.get("cc_volatility", 0.30)
                    risk_free_rate = config.get("risk_free_rate", 0.05)
                    
                    current_short_put = calculate_option_premium(
                        current_price, position['short_strike'], days_remaining, 
                        volatility, risk_free_rate, option_type='put'
                    )
                    current_long_put = calculate_option_premium(
                        current_price, position['long_strike'], days_remaining, 
                        volatility, risk_free_rate, option_type='put'
                    )
                    
                    current_value = current_short_put - current_long_put
                    position_value = position['net_credit'] - current_value
                
                open_positions_value += position_value
    
    return portfolio_value, open_positions_value

def evaluate_positions():
    """Evaluate all open positions and update P&L"""
    portfolio_pnl = state.portfolio_value
    
    for ticker in ['SI', 'CC']:
        current_price = get_current_price(ticker)
        ticker_pnl = 0
        
        for i, position in enumerate(state.positions[ticker]):
            if position['status'] == 'open':
                # Calculate current value of the position
                days_remaining = position['days_to_expiry'] - (datetime.now() - position['timestamp']).days
                days_remaining = max(0, days_remaining)
                
                if days_remaining == 0:
                    # Option expired
                    if current_price <= position['long_strike']:
                        # Maximum loss scenario
                        pnl = position['net_credit'] - (position['short_strike'] - position['long_strike'])
                    elif current_price <= position['short_strike']:
                        # Partial loss
                        pnl = position['net_credit'] - (position['short_strike'] - current_price)
                    else:
                        # Maximum profit scenario
                        pnl = position['net_credit']
                    
                    # Close the position
                    state.positions[ticker][i]['status'] = 'closed'
                    state.positions[ticker][i]['exit_price'] = current_price
                    state.positions[ticker][i]['pnl'] = pnl
                    
                    # Add to trades history
                    state.trades[ticker].append({
                        'timestamp': datetime.now(),
                        'action': 'exit',
                        'price': current_price,
                        'strategy': 'bear_put_spread',
                        'details': f"Closed position: PnL = {pnl:.2f}"
                    })
                    
                    log_message = f"\nCLOSED POSITION: {ticker} bear put spread"
                    print(log_message)
                    logging.info(log_message)
                    print(f"Exit price: ${current_price:.2f}")
                    print(f"P&L: ${pnl:.2f}")
                else:
                    # Position still open, calculate current value
                    volatility = config.get("si_volatility", 0.25) if ticker == 'SI' else config.get("cc_volatility", 0.30)
                    risk_free_rate = config.get("risk_free_rate", 0.05)
                    
                    current_short_put = calculate_option_premium(
                        current_price, position['short_strike'], days_remaining, 
                        volatility, risk_free_rate, option_type='put'
                    )
                    current_long_put = calculate_option_premium(
                        current_price, position['long_strike'], days_remaining, 
                        volatility, risk_free_rate, option_type='put'
                    )
                    
                    current_value = current_short_put - current_long_put
                    pnl = position['net_credit'] - current_value
                    
                    # Check if we should exit early
                    # Exit if we've reached 80% of max profit or if we're at 50% of max loss
                    max_profit = position['net_credit']
                    max_loss = (position['short_strike'] - position['long_strike']) - position['net_credit']
                    
                    if pnl >= 0.8 * max_profit or pnl <= -0.5 * max_loss:
                        # Close the position
                        state.positions[ticker][i]['status'] = 'closed'
                        state.positions[ticker][i]['exit_price'] = current_price
                        state.positions[ticker][i]['pnl'] = pnl
                        
                        # Add to trades history
                        state.trades[ticker].append({
                            'timestamp': datetime.now(),
                            'action': 'exit',
                            'price': current_price,
                            'strategy': 'bear_put_spread',
                            'details': f"Early exit: PnL = {pnl:.2f}"
                        })
                        
                        log_message = f"\nEARLY EXIT: {ticker} bear put spread"
                        print(log_message)
                        logging.info(log_message)
                        print(f"Exit price: ${current_price:.2f}")
                        print(f"P&L: ${pnl:.2f}")
                        print(f"Reason: {'Profit target reached' if pnl >= 0.8 * max_profit else 'Stop loss triggered'}")
                
                # Add to ticker P&L
                ticker_pnl += pnl
        
        # Update P&L history for this ticker
        state.pnl_history[ticker].append(ticker_pnl)
        if len(state.pnl_history[ticker]) > 500:
            state.pnl_history[ticker] = state.pnl_history[ticker][-500:]
        
        # Add to portfolio P&L
        portfolio_pnl += ticker_pnl
    
    # Update portfolio value
    state.portfolio_value = portfolio_pnl
    
    # Update portfolio P&L history
    state.pnl_history['portfolio'].append(portfolio_pnl)
    if len(state.pnl_history['portfolio']) > 500:
        state.pnl_history['portfolio'] = state.pnl_history['portfolio'][-500:]

def should_enter_position(ticker):
    """Determine if we should enter a new position based on market conditions"""
    # Check if we already have too many open positions
    max_positions = config.get("max_positions_per_instrument", 3)
    open_positions = sum(1 for pos in state.positions[ticker] if pos['status'] == 'open')
    if open_positions >= max_positions:
        return False
    
    # Check price trend - enter if we see a potential downtrend
    prices = state.price_history[ticker]
    if len(prices) < 5:
        return False
    
    # Simple trend check - look for a recent high followed by lower prices
    recent_prices = prices[-5:]
    if recent_prices[0] < recent_prices[1] and all(recent_prices[i] >= recent_prices[i+1] for i in range(1, 3)):
        # Price was going up, then started going down - potential reversal
        return True
    
    # Another condition - if price is significantly above a longer-term average
    if len(prices) >= 20:
        avg_20 = sum(prices[-20:]) / 20
        if prices[-1] > avg_20 * 1.05:  # Price is 5% above average
            return True
    
    # Random entry with low probability just to ensure we get some trades
    return random.random() < 0.05

def refresh_data():
    """Refresh all data - prices, order book, and positions"""
    update_prices()
    
    for ticker in ['SI', 'CC']:
        generate_order_book(ticker)
        
        # Check if we should enter a new position
        if should_enter_position(ticker):
            execute_bear_put_strategy(ticker)
    
    # Evaluate existing positions
    evaluate_positions()
    
    # Update last refresh time
    state.last_refresh = datetime.now()
    logging.info(f"Data refreshed at {state.last_refresh}")

# Console UI functions
def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print application header"""
    print("=" * 80)
    print("            TRADING SIMULATION PRICING ENGINE WITH REAL MARKET DATA")
    print("=" * 80)
    print(f"Last refresh: {state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Portfolio Value: ${state.portfolio_value:.2f} (Change: ${state.portfolio_value - config.get('initial_portfolio_value', 2500):.2f})")
    print("-" * 80)

def print_menu():
    """Print menu options"""
    print("\nMENU OPTIONS:")
    print("1. Market Overview")
    print("2. Position Details")
    print("3. Order Book")
    print("4. Trade History")
    print("5. Refresh Data")
    print("6. Change Refresh Rate (current: {} seconds)".format(state.refresh_rate))
    print("7. Toggle Auto-Refresh (current: {})".format("ON" if state.auto_refresh else "OFF"))
    print("0. Exit Application")
    print("-" * 80)

def display_market_overview():
    """Display market overview"""
    print("\nMARKET OVERVIEW")
    print("-" * 80)
    
    # Display current prices
    print("CURRENT PRICES (From Yahoo Finance):")
    for ticker in ['SI', 'CC']:
        yahoo_ticker = 'SI=F' if ticker == 'SI' else 'CC=F'
        current_price = get_current_price(ticker)
        print(f"{ticker} ({yahoo_ticker}): ${current_price:.2f}")
    
    # Display recent price trends
    print("\nRECENT PRICE MOVEMENTS:")
    
    for ticker in ['SI', 'CC']:
        prices = state.price_history[ticker]
        if len(prices) >= 5:
            recent_prices = prices[-5:]
            trend = "↑" if recent_prices[-1] > recent_prices[-2] else "↓"
            change = ((recent_prices[-1] / recent_prices[-5]) - 1) * 100
            print(f"{ticker}: {trend} {abs(change):.2f}% {'increase' if change > 0 else 'decrease'} over last 5 periods")
    
    # Display P&L summary
    print("\nPORTFOLIO P&L SUMMARY:")
    si_pnl = state.pnl_history['SI'][-1]
    cc_pnl = state.pnl_history['CC'][-1]
    total_pnl = si_pnl + cc_pnl
    
    print(f"Silver (SI) P&L: ${si_pnl:.2f}")
    print(f"Cocoa (CC) P&L: ${cc_pnl:.2f}")
    print(f"Total P&L: ${total_pnl:.2f}")
    
    # Display open positions count
    si_open = sum(1 for pos in state.positions['SI'] if pos['status'] == 'open')
    cc_open = sum(1 for pos in state.positions['CC'] if pos['status'] == 'open')
    
    print("\nOPEN POSITIONS:")
    print(f"Silver (SI): {si_open}")
    print(f"Cocoa (CC): {cc_open}")
    print(f"Total: {si_open + cc_open}")

def display_position_details():
    """Display position details"""
    print("\nPOSITION DETAILS")
    print("-" * 80)
    
    for ticker in ['SI', 'CC']:
        print(f"\n{ticker} POSITIONS:")
        
        # Check if there are any positions
        if not state.positions[ticker]:
            print(f"No positions for {ticker} yet.")
            continue
        
        # Display position details
        print(f"{'Status':<10} {'Entry Time':<20} {'Entry Price':<12} {'Short Strike':<14} {'Long Strike':<12} {'Net Credit':<12} {'Days to Expiry':<15} {'P&L':<10}")
        print("-" * 110)
        
        for pos in state.positions[ticker]:
            status = "OPEN" if pos['status'] == 'open' else "CLOSED"
            entry_time = pos['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            entry_price = f"${pos['entry_price']:.2f}"
            short_strike = f"${pos['short_strike']:.2f}"
            long_strike = f"${pos['long_strike']:.2f}"
            net_credit = f"${pos['net_credit']:.2f}"
            days_to_expiry = pos['days_to_expiry']
            
            pnl = pos.get('pnl', 'N/A')
            if pnl != 'N/A':
                pnl = f"${pnl:.2f}"
            
            print(f"{status:<10} {entry_time:<20} {entry_price:<12} {short_strike:<14} {long_strike:<12} {net_credit:<12} {days_to_expiry:<15} {pnl:<10}")
        
        # Display payoff info for open positions
        open_positions = [pos for pos in state.positions[ticker] if pos['status'] == 'open']
        if open_positions:
            latest_pos = open_positions[-1]
            current_price = get_current_price(ticker)
            
            print(f"\nLatest Open {ticker} Position Payoff Information:")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Maximum Profit: ${latest_pos['net_credit']:.2f}")
            
            max_loss = (latest_pos['short_strike'] - latest_pos['long_strike']) - latest_pos['net_credit']
            print(f"Maximum Loss: ${max_loss:.2f}")
            
            break_even = latest_pos['short_strike'] - latest_pos['net_credit']
            print(f"Break-Even Price: ${break_even:.2f}")
            
            if current_price <= latest_pos['long_strike']:
                print("Current Status: Maximum loss if expired today")
            elif current_price <= latest_pos['short_strike']:
                partial_loss = latest_pos['net_credit'] - (latest_pos['short_strike'] - current_price)
                print(f"Current Status: Partial loss of ${partial_loss:.2f} if expired today")
            else:
                print("Current Status: Maximum profit if expired today")

def display_order_book():
    """Display order book"""
    print("\nORDER BOOK")
    print("-" * 80)
    
    for ticker in ['SI', 'CC']:
        print(f"\n{ticker} ORDER BOOK:")
        
        # Get bids and asks
        bids = state.order_book[ticker]['bids']
        asks = state.order_book[ticker]['asks']
        
        if not bids or not asks:
            print(f"Order book for {ticker} is empty or being generated.")
            continue
        
        # Current price
        current_price = get_current_price(ticker)
        print(f"Current Price: ${current_price:.2f}")
        
        # Print bids
        print("\nBIDS (Buy Orders):")
        print(f"{'Price':<10} {'Size':<10}")
        print("-" * 20)
        
        for bid in bids[:5]:  # Display top 5 bids
            print(f"${bid['price']:<9.2f} {bid['size']:<10}")
        
        # Print asks
        print("\nASKS (Sell Orders):")
        print(f"{'Price':<10} {'Size':<10}")
        print("-" * 20)
        
        for ask in asks[:5]:  # Display top 5 asks
            print(f"${ask['price']:<9.2f} {ask['size']:<10}")
        
        # Calculate spread
        spread = asks[0]['price'] - bids[0]['price']
        print(f"\nBid-Ask Spread: ${spread:.2f}")

def display_trade_history():
    """Display trade history"""
    print("\nTRADE HISTORY")
    print("-" * 80)
    
    for ticker in ['SI', 'CC']:
        print(f"\n{ticker} TRADE HISTORY:")
        
        if not state.trades[ticker]:
            print(f"No trades for {ticker} yet.")
            continue
        
        print(f"{'Time':<20} {'Action':<10} {'Price':<12} {'Strategy':<20} {'Details':<50}")
        print("-" * 110)
        
        for trade in state.trades[ticker]:
            time_str = trade['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            action = trade['action'].upper()
            price = f"${trade['price']:.2f}"
            strategy = trade['strategy'].replace('_', ' ').title()
            details = trade['details']
            
            print(f"{time_str:<20} {action:<10} {price:<12} {strategy:<20} {details:<50}")
    
    # Display P&L summary
    print("\nP&L SUMMARY:")
    
    # Calculate closed trades P&L
    closed_pnl = {'SI': [], 'CC': []}
    
    for ticker in ['SI', 'CC']:
        for pos in state.positions[ticker]:
            if pos['status'] == 'closed' and 'pnl' in pos:
                closed_pnl[ticker].append(pos['pnl'])
    
    total_si_pnl = sum(closed_pnl['SI']) if closed_pnl['SI'] else 0
    total_cc_pnl = sum(closed_pnl['CC']) if closed_pnl['CC'] else 0
    
    print(f"Total Silver (SI) P&L: ${total_si_pnl:.2f}")
    print(f"Total Cocoa (CC) P&L: ${total_cc_pnl:.2f}")
    print(f"Combined P&L: ${(total_si_pnl + total_cc_pnl):.2f}")
    print(f"Current Portfolio Value: ${state.portfolio_value:.2f} (Change: ${state.portfolio_value - config.get('initial_portfolio_value', 2500):.2f})")

def auto_refresh_thread():
    """Thread for auto-refreshing data"""
    while True:
        try:
            if state.auto_refresh:
                time_since_refresh = (datetime.now() - state.last_refresh).total_seconds()
                if time_since_refresh >= state.refresh_rate:
                    refresh_data()
                    # We can't clear the screen here because it would disrupt user input
                    print(f"\n[AUTO REFRESH] Data refreshed at {state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
                    
            # Sleep for 1 second to prevent high CPU usage
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error in auto refresh thread: {e}")
            time.sleep(5)  # Sleep longer if there's an error

def headless_mode():
    """Run in headless mode without UI"""
    logging.info("Starting in headless mode")
    print("Running in headless mode. Check the log file for details.")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            refresh_data()
            time.sleep(state.refresh_rate)
    except KeyboardInterrupt:
        logging.info("Headless mode terminated by user")
        print("\nExiting headless mode.")

def main():
    """Main application entry point"""
    # Initialize order books
    for ticker in ['SI', 'CC']:
        generate_order_book(ticker)
    
    # Start auto-refresh thread
    refresh_thread = threading.Thread(target=auto_refresh_thread, daemon=True)
    refresh_thread.start()
    
    while True:
        clear_screen()
        print_header()
        
        # Display current view
        if state.current_view == "market_overview":
            display_market_overview()
        elif state.current_view == "position_details":
            display_position_details()
        elif state.current_view == "order_book":
            display_order_book()
        elif state.current_view == "trade_history":
            display_trade_history()
        
        print_menu()
        
        # Get user input
        try:
            choice = input("Enter your choice (0-7): ")
            
            if choice == '0':
                print("Exiting application...")
                break
            elif choice == '1':
                state.current_view = "market_overview"
            elif choice == '2':
                state.current_view = "position_details"
            elif choice == '3':
                state.current_view = "order_book"
            elif choice == '4':
                state.current_view = "trade_history"
            elif choice == '5':
                print("Refreshing data...")
                refresh_data()
                input("Press Enter to continue...")
            elif choice == '6':
                try:
                    new_rate = int(input("Enter new refresh rate in seconds (1-60): "))
                    if 1 <= new_rate <= 60:
                        state.refresh_rate = new_rate
                        print(f"Refresh rate updated to {state.refresh_rate} seconds.")
                    else:
                        print("Invalid input. Refresh rate must be between 1 and 60 seconds.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                input("Press Enter to continue...")
            elif choice == '7':
                state.auto_refresh = not state.auto_refresh
                print(f"Auto-refresh is now {'ON' if state.auto_refresh else 'OFF'}")
                input("Press Enter to continue...")
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")
        except KeyboardInterrupt:
            print("\nExiting application...")
            break

if __name__ == "__main__":
    # Check if running in headless mode
    if args.headless:
        headless_mode()
    else:
        main()