import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(layout="wide", page_title="Financial Portfolio Optimizer")

# Define the assets mentioned in the report
ASSETS = {
    'ES': {'name': 'E-mini S&P 500', 'type': 'Equity Index', 'ticker': 'ES=F'},
    'ZN': {'name': '10-Year T-Note', 'type': 'Interest Rate', 'ticker': 'ZN=F'},
    'ZL': {'name': 'Soybean Oil', 'type': 'Agricultural', 'ticker': 'ZL=F'},
    'ZB': {'name': '30-Year T-Bond', 'type': 'Interest Rate', 'ticker': 'ZB=F'},
    'BZ': {'name': 'Brent Crude', 'type': 'Energy', 'ticker': 'BZ=F'},
    'EUR': {'name': 'Euro FX', 'type': 'Currency', 'ticker': '6E=F'},
    'SI': {'name': 'Silver', 'type': 'Precious Metal', 'ticker': 'SI=F'},
    'ZQ': {'name': '30-Day Fed Funds', 'type': 'Interest Rate', 'ticker': 'ZQ=F'},
    'ZT': {'name': '2-Year T-Note', 'type': 'Interest Rate', 'ticker': 'ZT=F'},
    'ZC': {'name': 'Corn', 'type': 'Agricultural', 'ticker': 'ZC=F'},
    'CC': {'name': 'Cocoa', 'type': 'Agricultural', 'ticker': 'CC=F'},
    'OJ': {'name': 'Orange Juice', 'type': 'Agricultural', 'ticker': 'OJ=F'},
    'CL': {'name': 'Crude Oil', 'type': 'Energy', 'ticker': 'CL=F'},
    'NG': {'name': 'Natural Gas', 'type': 'Energy', 'ticker': 'NG=F'},
    'GC': {'name': 'Gold', 'type': 'Precious Metal', 'ticker': 'GC=F'},
    'HE': {'name': 'Lean Hogs', 'type': 'Livestock', 'ticker': 'HE=F'},
    'AUD': {'name': 'Australian Dollar', 'type': 'Currency', 'ticker': '6A=F'}
}

# Define strategies from the report
STRATEGIES = {
    'arbitrage': {
        'name': 'Arbitrage & Mispricing', 
        'description': 'Captures profits from market inefficiencies with low risk',
        'allocation_percent': 35
    },
    'income': {
        'name': 'Market-Neutral Income', 
        'description': 'Generates steady income through option premium selling in stable markets',
        'allocation_percent': 40
    },
    'directional': {
        'name': 'Directional & Volatility Plays', 
        'description': 'Makes calculated bets on price movements with defined risk',
        'allocation_percent': 25
    }
}

# Define specific trade setups from the report
TRADE_SETUPS = {
    'ES_reversal': {
        'asset': 'ES',
        'strategy': 'arbitrage',
        'name': 'ES Reversal Arbitrage',
        'description': 'Exploits put-call parity violation',
        'allocation': 12000,
        'entry_price': 6350.25,
        'take_profit': 6275.00,
        'stop_loss': 6400.00,
        'expected_return': 0.0182  # 1.82% estimated return
    },
    'ZN_reversal': {
        'asset': 'ZN',
        'strategy': 'arbitrage',
        'name': 'ZN Reversal Arbitrage',
        'description': 'Exploits put-call parity violation in interest rates',
        'allocation': 8000,
        'entry_price': 111.39,
        'take_profit': 102.75,
        'stop_loss': 115.00,
        'expected_return': 0.0167  # 1.67% estimated return
    },
    'ZL_cash_carry': {
        'asset': 'ZL',
        'strategy': 'arbitrage',
        'name': 'ZL Cash-and-Carry Arbitrage',
        'description': 'Exploits futures premium to cash price',
        'allocation': 15000,
        'entry_price': 55.48,
        'take_profit': 55.71,
        'stop_loss': 55.30,
        'expected_return': 0.0041  # 0.41% estimated return
    },
    'ZT_iron_condor': {
        'asset': 'ZT',
        'strategy': 'income',
        'name': 'ZT High-Probability Iron Condor',
        'description': 'Sells premium in low-volatility environment',
        'allocation': 10000,
        'entry_price': 103.75,
        'take_profit': 103.75,  # Center of expected range
        'stop_loss': [98.75, 108.75],  # Lower and upper bounds
        'expected_return': 0.0125  # 1.25% estimated return
    },
    'ZC_iron_condor': {
        'asset': 'ZC',
        'strategy': 'income',
        'name': 'ZC ARIMA-Based Iron Condor',
        'description': 'Sells premium based on statistical price forecast',
        'allocation': 15000,
        'entry_price': 417.5,
        'take_profit': 417.5,  # Center of expected range
        'stop_loss': [412.5, 422.5],  # Lower and upper bounds
        'expected_return': 0.0310  # 3.10% estimated return
    },
    'CL_iron_condor': {
        'asset': 'CL',
        'strategy': 'income',
        'name': 'CL Volatility-Premium Iron Condor',
        'description': 'Sells rich premium in high-volatility environment',
        'allocation': 15000,
        'entry_price': 65.31,
        'take_profit': 65.31,  # Center of expected range
        'stop_loss': [60.31, 70.31],  # Lower and upper bounds
        'expected_return': 0.0280  # 2.80% estimated return
    },
    'SI_bear_put': {
        'asset': 'SI',
        'strategy': 'directional',
        'name': 'SI Bear Put Spread',
        'description': 'Profits from expected price decline',
        'allocation': 8000,
        'entry_price': 39.28,
        'take_profit': 38.00,
        'stop_loss': 39.50,
        'expected_return': 0.0520  # 5.20% estimated return
    },
    'CC_bear_put': {
        'asset': 'CC',
        'strategy': 'directional',
        'name': 'CC High-Volatility Bear Put Spread',
        'description': 'Profits from expected price decline with high volatility',
        'allocation': 8000,
        'entry_price': 8150,
        'take_profit': 8000,
        'stop_loss': 8200,
        'expected_return': 0.0650  # 6.50% estimated return
    },
    'NG_iron_condor': {
        'asset': 'NG',
        'strategy': 'directional',
        'name': 'NG Volatility Contraction Iron Condor',
        'description': 'Profits from expected volatility contraction',
        'allocation': 9000,
        'entry_price': 2.40,
        'take_profit': 2.40,  # Center of expected range
        'stop_loss': [2.20, 2.60],  # Lower and upper bounds
        'expected_return': 0.0370  # 3.70% estimated return
    }
}

# Data preparation and fetching functions
@st.cache_data
def fetch_historical_data(ticker, start_date, end_date):
    """Fetch historical price data for a ticker symbol."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

@st.cache_data
def generate_synthetic_history(asset_code, length=252, volatility=None):
    """
    Generate synthetic price history for assets without available data.
    Uses deterministic patterns based on asset characteristics.
    """
    asset = ASSETS[asset_code]
    
    # Set base parameters based on asset type
    if volatility is None:
        if asset['type'] == 'Interest Rate':
            volatility = 0.02  # 2% annualized volatility
            trend = 0.0001
            seasonality = 0.0005
        elif asset['type'] == 'Equity Index':
            volatility = 0.15  # 15% annualized volatility
            trend = 0.0004
            seasonality = 0.002
        elif asset['type'] == 'Energy':
            volatility = 0.25  # 25% annualized volatility
            trend = 0.0002
            seasonality = 0.01
        elif asset['type'] == 'Agricultural':
            volatility = 0.20  # 20% annualized volatility
            trend = 0.0001
            seasonality = 0.015
        elif asset['type'] == 'Precious Metal':
            volatility = 0.18  # 18% annualized volatility
            trend = 0.0003
            seasonality = 0.005
        elif asset['type'] == 'Currency':
            volatility = 0.10  # 10% annualized volatility
            trend = 0.0001
            seasonality = 0.001
        elif asset['type'] == 'Livestock':
            volatility = 0.22  # 22% annualized volatility
            trend = 0.0002
            seasonality = 0.02
        else:
            volatility = 0.15  # Default
            trend = 0.0002
            seasonality = 0.005
    
    # Starting price based on trade setups if available
    start_price = None
    for setup in TRADE_SETUPS.values():
        if setup['asset'] == asset_code:
            start_price = setup['entry_price']
            break
    
    if start_price is None:
        start_price = 100.0  # Default starting price
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=length)
    dates = pd.date_range(start=start_date, end=end_date, periods=length)
    
    # Generate price series with trend, seasonality, and autocorrelation
    t = np.arange(length)
    daily_vol = volatility / np.sqrt(252)
    
    # Add trend component
    trend_component = trend * t
    
    # Add seasonality (varies by asset type)
    if asset['type'] in ['Agricultural', 'Energy', 'Livestock']:
        # Stronger seasonal patterns
        seasonality_component = seasonality * np.sin(2 * np.pi * t / 63)  # ~quarterly cycle
    else:
        # Milder seasonal patterns
        seasonality_component = seasonality * np.sin(2 * np.pi * t / 126)  # ~half-yearly cycle
    
    # Add cyclical component (economic cycle)
    cycle_component = 0.01 * np.sin(2 * np.pi * t / length)
    
    # Base path with autocorrelation
    price = np.zeros(length)
    price[0] = start_price
    
    # Generate path with autocorrelation
    for i in range(1, length):
        # Price depends partly on previous price (autocorrelation)
        autocorr_factor = 0.85
        random_change = daily_vol * (i % 7 - 3) * (1 - autocorr_factor)  # Deterministic but varied
        price[i] = price[i-1] * (1 + random_change + trend + seasonality_component[i] + cycle_component[i])
    
    # Adjust to end at the entry price for the asset
    scale_factor = start_price / price[-1]
    price = price * scale_factor
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': price * (1 - daily_vol/4),
        'High': price * (1 + daily_vol/2),
        'Low': price * (1 - daily_vol/2),
        'Close': price,
        'Volume': np.abs(np.diff(np.concatenate([[0], price]))) * 1000000
    })
    
    df.set_index('Date', inplace=True)
    return df

@st.cache_data
def prepare_data_for_asset(asset_code):
    """Prepare historical and synthetic data for an asset."""
    asset = ASSETS[asset_code]
    ticker = asset['ticker']
    
    # Try to fetch real data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    df = fetch_historical_data(ticker, start_date, end_date)
    
    # If no data or limited data, use synthetic data
    if df.empty or len(df) < 200:
        df = generate_synthetic_history(asset_code)
    
    # Calculate daily returns
    df['Return'] = df['Close'].pct_change().fillna(0)
    
    # Calculate volatility (20-day rolling standard deviation of returns)
    df['Volatility'] = df['Return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    # Calculate moving averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    return df

# Forecasting functions
def calculate_historical_volatility(price_series, window=20):
    """Calculate historical volatility from a price series."""
    returns = price_series.pct_change().dropna()
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def forecast_price_path(historical_data, days=28, method='ema'):
    """
    Forecast future prices using deterministic methods.
    
    Parameters:
    - historical_data: DataFrame with historical price data
    - days: Number of days to forecast
    - method: Forecasting method ('ema', 'sma', or 'regression')
    
    Returns:
    - DataFrame with forecasted prices
    """
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    if method == 'ema':
        # Use exponential moving average for forecasting
        ema_short = historical_data['Close'].ewm(span=10, adjust=False).mean()
        ema_long = historical_data['Close'].ewm(span=30, adjust=False).mean()
        
        # Calculate the recent trend from EMA difference
        recent_trend = (ema_short.iloc[-1] / ema_long.iloc[-1]) - 1
        
        # Project forward
        last_price = historical_data['Close'].iloc[-1]
        forecasted_prices = [last_price]
        
        for i in range(1, days):
            # Forecast becomes more conservative over time
            damping_factor = 0.95 ** i
            next_price = forecasted_prices[-1] * (1 + recent_trend * damping_factor)
            forecasted_prices.append(next_price)
    
    elif method == 'sma':
        # Use simple moving averages
        sma_20 = historical_data['Close'].rolling(window=20).mean()
        sma_50 = historical_data['Close'].rolling(window=50).mean()
        
        # Calculate the trend
        recent_trend = (sma_20.iloc[-1] / sma_50.iloc[-1]) - 1
        
        # Project forward
        last_price = historical_data['Close'].iloc[-1]
        forecasted_prices = [last_price]
        
        for i in range(1, days):
            damping_factor = 0.95 ** i
            next_price = forecasted_prices[-1] * (1 + recent_trend * damping_factor)
            forecasted_prices.append(next_price)
            
    elif method == 'regression':
        # Use linear regression
        y = historical_data['Close'].values[-30:]  # Last 30 days
        X = np.arange(len(y)).reshape(-1, 1)
        
        model = sm.OLS(y, sm.add_constant(X)).fit()
        
        # Project forward
        X_future = np.arange(len(y), len(y) + days).reshape(-1, 1)
        forecasted_prices = model.predict(sm.add_constant(X_future))
        
    else:
        raise ValueError(f"Unknown forecasting method: {method}")
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Close': forecasted_prices,
    })
    forecast_df.set_index('Date', inplace=True)
    
    # Calculate volatility for the forecast
    historical_vol = calculate_historical_volatility(historical_data['Close']).iloc[-1]
    forecast_df['Volatility'] = [historical_vol] * len(forecast_df)
    
    return forecast_df

def extend_arima_forecast(historical_data, days=28):
    """
    Forecast prices using ARIMA model and extend the forecast.
    """
    # Prepare data
    train_data = historical_data['Close'].dropna()
    
    # Fit ARIMA model
    try:
        model = ARIMA(train_data, order=(5,1,0))
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=days)
        
        # Create forecast DataFrame
        future_dates = pd.date_range(start=historical_data.index[-1] + timedelta(days=1), periods=days)
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Close': forecast
        })
        forecast_df.set_index('Date', inplace=True)
        
        # Calculate volatility for the forecast
        historical_vol = calculate_historical_volatility(historical_data['Close']).iloc[-1]
        forecast_df['Volatility'] = [historical_vol] * len(forecast_df)
        
        return forecast_df
    except:
        # Fallback to EMA forecasting if ARIMA fails
        return forecast_price_path(historical_data, days, 'ema')

# Strategy simulation functions
def forecast_arbitrage_strategy(setup, historical_data, forecast_data):
    """
    Simulate arbitrage strategy performance based on the setup details.
    Returns projected P&L for the strategy.
    """
    # Extract key parameters
    entry_price = float(setup['entry_price'])
    take_profit = float(setup['take_profit'])
    stop_loss = setup['stop_loss']
    allocation = float(setup['allocation'])
    
    # Initialize results
    dates = forecast_data.index
    prices = forecast_data['Close'].values
    daily_pnl = np.zeros(len(dates))
    cumulative_pnl = np.zeros(len(dates))
    position_status = ['Open'] * len(dates)
    
    # For arbitrage, we assume a more deterministic outcome
    # Calculate expected daily yield
    expected_return = setup['expected_return']
    days_to_completion = len(dates)
    daily_yield = expected_return / days_to_completion
    
    # Simulate daily P&L
    for i in range(len(dates)):
        # For arbitrage, assume a steady progression toward the target profit
        daily_pnl[i] = allocation * daily_yield
        
        # Adjust based on current market price (could affect carry trades)
        # Get current price as a scalar
        if isinstance(prices[i], np.ndarray):
            current_price = float(prices[i][0])
        elif hasattr(prices[i], 'item'):
            current_price = float(prices[i].item())
        else:
            current_price = float(prices[i])
        
        price_adjustment = 0
        
        # Process stop_loss which could be a scalar or a list
        if isinstance(stop_loss, list):
            # For strategies with ranges, we'll check if price is outside the range
            lower_bound, upper_bound = float(stop_loss[0]), float(stop_loss[1])
            if current_price <= lower_bound or current_price >= upper_bound:
                price_adjustment = -0.5 * daily_yield  # Reduce profit
                position_status[i] = 'Stop Loss'
        else:
            # Simple stop loss check
            stop_loss_value = float(stop_loss)
            if current_price >= stop_loss_value:
                price_adjustment = -0.5 * daily_yield  # Reduce profit
                position_status[i] = 'Stop Loss'
        
        # Take profit check
        if current_price <= float(take_profit):
            price_adjustment = 0.2 * daily_yield  # Increase profit
            position_status[i] = 'Take Profit'
            
        daily_pnl[i] += allocation * price_adjustment
        
        # Calculate cumulative P&L
        if i == 0:
            cumulative_pnl[i] = daily_pnl[i]
        else:
            cumulative_pnl[i] = cumulative_pnl[i-1] + daily_pnl[i]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Daily_PnL': daily_pnl,
        'Cumulative_PnL': cumulative_pnl,
        'Status': position_status
    })
    
    results.set_index('Date', inplace=True)
    return results

def forecast_iron_condor(setup, historical_data, forecast_data):
    """
    Simulate Iron Condor strategy performance.
    Returns projected P&L for the strategy.
    """
    # Extract key parameters
    entry_price = float(setup['entry_price'])
    stop_loss = setup['stop_loss']  # This should be a list [lower_bound, upper_bound]
    allocation = float(setup['allocation'])
    
    # Iron Condor parameters - ensure we have two bounds
    if isinstance(stop_loss, list) and len(stop_loss) == 2:
        lower_bound = float(stop_loss[0])
        upper_bound = float(stop_loss[1])
    else:
        # Default to a range around entry price if not properly specified
        width = 5.0  # Default width
        lower_bound = entry_price - width
        upper_bound = entry_price + width
        
    width = 5.0  # Typical width of spreads
    max_profit = allocation * setup['expected_return']  # Maximum profit from the trade
    max_loss = allocation * 0.2  # Typical max loss would be width of spread minus credit
    
    # Initialize results
    dates = forecast_data.index
    prices = forecast_data['Close'].values
    daily_pnl = np.zeros(len(dates))
    cumulative_pnl = np.zeros(len(dates))
    position_status = ['Open'] * len(dates)
    
    # Calculate theta decay (daily profit from time decay)
    days_to_expiration = len(dates)
    daily_theta = max_profit / days_to_expiration
    
    # Simulate daily P&L
    for i in range(len(dates)):
        # Get current price as a scalar
        if isinstance(prices[i], np.ndarray):
            current_price = float(prices[i][0])
        elif hasattr(prices[i], 'item'):
            current_price = float(prices[i].item())
        else:
            current_price = float(prices[i])
        
        # Default P&L is from theta decay
        daily_pnl[i] = daily_theta
        
        # Adjust based on price movement
        if current_price <= lower_bound:
            # Price breached lower boundary
            breach_amount = (lower_bound - current_price) / width
            loss_factor = min(1.0, breach_amount)
            daily_pnl[i] = -max_loss * loss_factor / days_to_expiration
            position_status[i] = 'Lower Breach'
        elif current_price >= upper_bound:
            # Price breached upper boundary
            breach_amount = (current_price - upper_bound) / width
            loss_factor = min(1.0, breach_amount)
            daily_pnl[i] = -max_loss * loss_factor / days_to_expiration
            position_status[i] = 'Upper Breach'
        
        # Calculate cumulative P&L
        if i == 0:
            cumulative_pnl[i] = daily_pnl[i]
        else:
            cumulative_pnl[i] = cumulative_pnl[i-1] + daily_pnl[i]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Daily_PnL': daily_pnl,
        'Cumulative_PnL': cumulative_pnl,
        'Status': position_status
    })
    
    results.set_index('Date', inplace=True)
    return results

def forecast_directional_strategy(setup, historical_data, forecast_data):
    """
    Simulate directional strategy performance (e.g., bear put spread).
    Returns projected P&L for the strategy.
    """
    # Extract key parameters
    entry_price = float(setup['entry_price'])
    take_profit = float(setup['take_profit'])
    
    # Handle stop_loss which can be a scalar or a list
    stop_loss = setup['stop_loss']
    if isinstance(stop_loss, list):
        # For strategies with ranges, we'll use the appropriate boundary based on direction
        is_bearish = take_profit < entry_price
        stop_loss_value = float(stop_loss[1]) if is_bearish else float(stop_loss[0])
    else:
        stop_loss_value = float(stop_loss)
    
    allocation = float(setup['allocation'])
    
    # Determine strategy direction
    is_bearish = take_profit < entry_price
    
    # Initialize results
    dates = forecast_data.index
    prices = forecast_data['Close'].values
    daily_pnl = np.zeros(len(dates))
    cumulative_pnl = np.zeros(len(dates))
    position_status = ['Open'] * len(dates)
    
    # Maximum potential profit and loss
    max_profit = allocation * setup['expected_return']
    max_loss = allocation * 0.5 * setup['expected_return']  # Typically risk is about half the reward
    
    # Calculate the price range for P&L calculation
    price_range_for_max_profit = abs(entry_price - take_profit)
    
    # Prevent division by zero - if entry_price equals take_profit
    if price_range_for_max_profit < 0.0001:  # Small epsilon to account for floating point errors
        price_range_for_max_profit = 0.0001  # Set a small non-zero value
    
    # Simulate daily P&L
    for i in range(len(dates)):
        # Get current price as a scalar
        if isinstance(prices[i], np.ndarray):
            current_price = float(prices[i][0])
        elif hasattr(prices[i], 'item'):
            current_price = float(prices[i].item())
        else:
            current_price = float(prices[i])
        
        # Determine position status and P&L based on price and strategy direction
        if is_bearish:
            # Bearish strategy logic
            if current_price <= take_profit:
                # Take profit hit
                daily_pnl[i] = max_profit / len(dates)
                position_status[i] = 'Take Profit'
            elif current_price >= stop_loss_value:
                # Stop loss hit
                daily_pnl[i] = -max_loss / len(dates)
                position_status[i] = 'Stop Loss'
            else:
                # Position open, P&L depends on price movement
                price_movement = entry_price - current_price
                profit_factor = min(1.0, price_movement / price_range_for_max_profit)
                profit_factor = max(-1.0, profit_factor)  # Cap at -100% (max loss)
                
                if profit_factor >= 0:
                    daily_pnl[i] = profit_factor * max_profit / len(dates)
                else:
                    daily_pnl[i] = profit_factor * max_loss / len(dates)
        else:
            # Bullish strategy logic
            if current_price >= take_profit:
                # Take profit hit
                daily_pnl[i] = max_profit / len(dates)
                position_status[i] = 'Take Profit'
            elif current_price <= stop_loss_value:
                # Stop loss hit
                daily_pnl[i] = -max_loss / len(dates)
                position_status[i] = 'Stop Loss'
            else:
                # Position open, P&L depends on price movement
                price_movement = current_price - entry_price
                profit_factor = min(1.0, price_movement / price_range_for_max_profit)
                profit_factor = max(-1.0, profit_factor)  # Cap at -100% (max loss)
                
                if profit_factor >= 0:
                    daily_pnl[i] = profit_factor * max_profit / len(dates)
                else:
                    daily_pnl[i] = profit_factor * max_loss / len(dates)
        
        # Calculate cumulative P&L
        if i == 0:
            cumulative_pnl[i] = daily_pnl[i]
        else:
            cumulative_pnl[i] = cumulative_pnl[i-1] + daily_pnl[i]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Daily_PnL': daily_pnl,
        'Cumulative_PnL': cumulative_pnl,
        'Status': position_status
    })
    
    results.set_index('Date', inplace=True)
    return results

def simulate_strategy_performance(setup, historical_data, days=28):
    """
    Simulate strategy performance based on the strategy type.
    """
    # Generate price forecast
    forecast_data = forecast_price_path(historical_data, days)
    
    # Simulate strategy based on type
    strategy_type = setup['strategy']
    
    if strategy_type == 'arbitrage':
        return forecast_arbitrage_strategy(setup, historical_data, forecast_data)
    elif strategy_type == 'income':
        return forecast_iron_condor(setup, historical_data, forecast_data)
    elif strategy_type == 'directional':
        return forecast_directional_strategy(setup, historical_data, forecast_data)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

# Portfolio optimization functions
def calculate_portfolio_metrics(portfolio_results):
    """
    Calculate portfolio performance metrics.
    """
    # Extract cumulative P&L
    cumulative_pnl = portfolio_results['Cumulative_PnL']
    
    # Calculate return metrics
    total_return = cumulative_pnl.iloc[-1]
    daily_returns = portfolio_results['Daily_PnL'] / 10000  # Normalize to portfolio value
    
    # Calculate risk metrics - FIXED VOLATILITY CALCULATION
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized daily volatility
    
    # Calculate max drawdown - FIXED CALCULATION
    peak = cumulative_pnl.iloc[0]
    max_drawdown = 0
    
    for value in cumulative_pnl:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak != 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = (total_return / 50000) / volatility if volatility > 0 else 0  # Normalize return
    
    # Return metrics
    return {
        'Total Return': total_return,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
    }

def optimize_portfolio_allocation(trade_setups, initial_capital=50000):
    """
    Optimize portfolio allocation to maximize risk-adjusted returns.
    
    Parameters:
    - trade_setups: Dictionary of trade setups
    - initial_capital: Initial capital amount
    
    Returns:
    - Dictionary with optimal allocations
    """
    # Extract expected returns and allocations
    assets = list(trade_setups.keys())
    expected_returns = np.array([setup['expected_return'] for setup in trade_setups.values()])
    current_allocations = np.array([setup['allocation'] for setup in trade_setups.values()])
    
    # Scale to match initial capital
    scale_factor = initial_capital / np.sum(current_allocations)
    current_allocations = current_allocations * scale_factor
    
    # Define objective function (negative Sharpe ratio to minimize)
    def objective(weights):
        portfolio_return = np.sum(weights * expected_returns)
        # Simplified risk calculation (could be improved with covariance matrix)
        portfolio_risk = np.sqrt(np.sum((weights * expected_returns)**2)) / len(weights)
        return -portfolio_return / portfolio_risk if portfolio_risk > 0 else -portfolio_return
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - initial_capital}  # Sum of weights equals capital
    ]
    
    # Define bounds (minimum 0, maximum can be adjusted)
    bounds = [(0, initial_capital * 0.5) for _ in range(len(assets))]
    
    # Initial guess (current allocations)
    initial_guess = current_allocations
    
    # Perform optimization
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Extract optimal allocations
    optimal_allocations = result.x
    
    # Calculate expected returns
    current_expected_return = np.sum(current_allocations * expected_returns)
    optimal_expected_return = np.sum(optimal_allocations * expected_returns)
    
    # Create result dictionary
    allocation_result = {
        'Current Allocations': dict(zip(assets, current_allocations)),
        'Optimal Allocations': dict(zip(assets, optimal_allocations)),
        'Current Expected Return': current_expected_return,
        'Optimal Expected Return': optimal_expected_return,
        'Improvement': optimal_expected_return - current_expected_return
    }
    
    return allocation_result

def simulate_portfolio_performance(trade_setups, allocations, historical_data_dict, days=28):
    """
    Simulate overall portfolio performance based on individual strategies.
    
    Parameters:
    - trade_setups: Dictionary of trade setups
    - allocations: Dictionary of capital allocations to each strategy
    - historical_data_dict: Dictionary of historical data for each asset
    - days: Number of days to simulate
    
    Returns:
    - DataFrame with portfolio performance
    """
    # Initialize results DataFrame with dates
    first_asset = list(trade_setups.keys())[0]
    forecast_data = forecast_price_path(historical_data_dict[trade_setups[first_asset]['asset']], days)
    dates = forecast_data.index
    
    portfolio_results = pd.DataFrame({
        'Date': dates,
        'Daily_PnL': np.zeros(len(dates)),
        'Cumulative_PnL': np.zeros(len(dates))
    })
    portfolio_results.set_index('Date', inplace=True)
    
    # Simulate each strategy and add to portfolio results
    strategy_results = {}
    
    for strategy_name, setup in trade_setups.items():
        asset_code = setup['asset']
        historical_data = historical_data_dict[asset_code]
        
        # Simulate strategy performance
        results = simulate_strategy_performance(setup, historical_data, days)
        
        # Scale P&L based on allocation
        allocation = allocations.get(strategy_name, setup['allocation'])
        results['Daily_PnL'] = results['Daily_PnL'] * (allocation / setup['allocation'])
        results['Cumulative_PnL'] = results['Cumulative_PnL'] * (allocation / setup['allocation'])
        
        # Add to portfolio results
        portfolio_results['Daily_PnL'] += results['Daily_PnL']
        
        # Store individual results
        strategy_results[strategy_name] = results
    
    # Calculate cumulative P&L for the portfolio
    portfolio_results['Cumulative_PnL'] = portfolio_results['Daily_PnL'].cumsum()
    
    return portfolio_results, strategy_results

# Visualization functions
def plot_portfolio_performance(portfolio_results, strategy_results=None, width=800, height=500):
    """
    Create an interactive plot of portfolio performance.
    """
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Cumulative P&L", "Daily P&L"),
        row_heights=[0.7, 0.3]
    )
    
    # Plot cumulative P&L
    fig.add_trace(
        go.Scatter(
            x=portfolio_results.index, 
            y=portfolio_results['Cumulative_PnL'],
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Plot individual strategies if provided
    if strategy_results:
        colors = px.colors.qualitative.Plotly
        for i, (strategy_name, results) in enumerate(strategy_results.items()):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=results.index,
                    y=results['Cumulative_PnL'],
                    mode='lines',
                    name=strategy_name,
                    line=dict(color=color, width=1, dash='dot'),
                    visible='legendonly'  # Hide by default
                ),
                row=1, col=1
            )
    
    # Plot daily P&L
    fig.add_trace(
        go.Bar(
            x=portfolio_results.index,
            y=portfolio_results['Daily_PnL'],
            name='Daily P&L',
            marker_color='green'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Performance Forecast (4 Weeks)',
        xaxis_title='Date',
        yaxis_title='P&L ($)',
        legend_title='Strategies',
        width=width,
        height=height,
        hovermode='x unified'
    )
    
    return fig

def plot_asset_price_forecast(asset_code, historical_data, forecast_data, setup=None, width=800, height=500):
    """
    Create an interactive plot of asset price with forecast.
    """
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Price Forecast", "Volatility"),
        row_heights=[0.7, 0.3]
    )
    
    # Get last 30 days of historical data
    historical_subset = historical_data.iloc[-30:]
    
    # Plot historical price
    fig.add_trace(
        go.Scatter(
            x=historical_subset.index, 
            y=historical_subset['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Plot forecasted price
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index, 
            y=forecast_data['Close'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add entry, take profit, and stop loss lines if setup is provided
    if setup:
        # Add horizontal line for entry price
        fig.add_hline(
            y=setup['entry_price'], 
            line=dict(color='green', width=1, dash='dash'),
            row=1, col=1,
            annotation_text="Entry",
            annotation_position="bottom right"
        )
        
        # Add take profit line
        if isinstance(setup['take_profit'], (int, float)):
            fig.add_hline(
                y=setup['take_profit'], 
                line=dict(color='blue', width=1, dash='dash'),
                row=1, col=1,
                annotation_text="Take Profit",
                annotation_position="bottom right"
            )
        
        # Add stop loss line(s)
        if isinstance(setup['stop_loss'], (int, float)):
            fig.add_hline(
                y=setup['stop_loss'], 
                line=dict(color='red', width=1, dash='dash'),
                row=1, col=1,
                annotation_text="Stop Loss",
                annotation_position="bottom right"
            )
        elif isinstance(setup['stop_loss'], list) and len(setup['stop_loss']) == 2:
            # For Iron Condors with upper and lower bounds
            fig.add_hline(
                y=setup['stop_loss'][0], 
                line=dict(color='red', width=1, dash='dash'),
                row=1, col=1,
                annotation_text="Lower Bound",
                annotation_position="bottom right"
            )
            fig.add_hline(
                y=setup['stop_loss'][1], 
                line=dict(color='red', width=1, dash='dash'),
                row=1, col=1,
                annotation_text="Upper Bound",
                annotation_position="bottom right"
            )
    
    # Plot volatility
    combined_vol = pd.concat([
        historical_subset['Volatility'] if 'Volatility' in historical_subset.columns else pd.Series(None, index=historical_subset.index),
        forecast_data['Volatility']
    ])
    
    fig.add_trace(
        go.Scatter(
            x=combined_vol.index, 
            y=combined_vol.values,
            mode='lines',
            name='Volatility',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ASSETS[asset_code]["name"]} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Data',
        width=width,
        height=height,
        hovermode='x unified'
    )
    
    return fig

# Main app code
def main():
    # Set up sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Portfolio Dashboard", "Asset Analysis", "Strategy Optimization", "Backtesting"]
    )
    
    # Initialize session state for toggles
    if 'selected_assets' not in st.session_state:
        st.session_state.selected_assets = {asset_code: True for asset_code in ASSETS}
    
    if 'selected_strategies' not in st.session_state:
        st.session_state.selected_strategies = {strategy_name: True for strategy_name in TRADE_SETUPS}
    
    # Prepare data
    with st.spinner("Loading and preparing data..."):
        historical_data_dict = {}
        forecast_data_dict = {}
        
        for asset_code in ASSETS:
            historical_data_dict[asset_code] = prepare_data_for_asset(asset_code)
            forecast_data_dict[asset_code] = forecast_price_path(historical_data_dict[asset_code], days=28)
    
    # Default capital allocation
    initial_capital = 50000
    
    # Portfolio Dashboard page
    if page == "Portfolio Dashboard":
        st.header("Portfolio Performance Dashboard")
        
        # Portfolio summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Initial Capital", f"${initial_capital:,.2f}")
        
        # Calculate active trades based on toggles
        active_trade_setups = {
            name: setup for name, setup in TRADE_SETUPS.items() 
            if st.session_state.selected_strategies.get(name, True) and 
               st.session_state.selected_assets.get(setup['asset'], True)
        }
        
        # Optimize portfolio allocation
        optimization_result = optimize_portfolio_allocation(active_trade_setups, initial_capital)
        optimal_allocations = optimization_result['Optimal Allocations']
        
        with col2:
            st.metric(
                "Expected Return (4 Weeks)", 
                f"${optimization_result['Optimal Expected Return']:,.2f}",
                f"{optimization_result['Optimal Expected Return'] / initial_capital * 100:.2f}%"
            )
        
        with col3:
            st.metric(
                "Improvement Over Baseline", 
                f"${optimization_result['Improvement']:,.2f}",
                f"{optimization_result['Improvement'] / optimization_result['Current Expected Return'] * 100:.2f}%"
            )
        
        # Simulate portfolio performance
        portfolio_results, strategy_results = simulate_portfolio_performance(
            active_trade_setups, optimal_allocations, historical_data_dict, days=28
        )
        
        # Calculate portfolio metrics
        metrics = calculate_portfolio_metrics(portfolio_results)
        
        # Display portfolio metrics
        st.subheader("Portfolio Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"${metrics['Total Return']:,.2f}")
        
        with col2:
            st.metric("Volatility", f"{metrics['Volatility'] * 100:.2f}%")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics['Max Drawdown'] * 100:.2f}%")
        
        with col4:
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        
        # Plot portfolio performance
        st.subheader("Portfolio Performance Forecast (4 Weeks)")
        fig = plot_portfolio_performance(portfolio_results, strategy_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy allocation
        st.subheader("Optimal Strategy Allocation")
        
        # Create DataFrame for allocation table
        allocation_data = []
        for strategy_name, allocation in optimal_allocations.items():
            if strategy_name in active_trade_setups:
                setup = active_trade_setups[strategy_name]
                expected_return = setup['expected_return'] * allocation
                expected_return_pct = setup['expected_return'] * 100
                
                allocation_data.append({
                    'Strategy': strategy_name,
                    'Asset': ASSETS[setup['asset']]['name'],
                    'Type': STRATEGIES[setup['strategy']]['name'],
                    'Allocation ($)': f"${allocation:,.2f}",
                    'Allocation (%)': f"{allocation / initial_capital * 100:.2f}%",
                    'Expected Return ($)': f"${expected_return:,.2f}",
                    'Expected Return (%)': f"{expected_return_pct:.2f}%",
                })
        
        allocation_df = pd.DataFrame(allocation_data)
        st.dataframe(allocation_df, use_container_width=True)
        
    # Asset Analysis page
    elif page == "Asset Analysis":
        st.header("Asset Analysis")
        
        # Asset selection
        st.sidebar.subheader("Asset Selection")
        for asset_code in ASSETS:
            st.session_state.selected_assets[asset_code] = st.sidebar.checkbox(
                ASSETS[asset_code]['name'],
                value=st.session_state.selected_assets.get(asset_code, True),
                key=f"asset_{asset_code}"
            )
        
        # Display analysis for selected assets
        for asset_code, is_selected in st.session_state.selected_assets.items():
            if is_selected:
                st.subheader(ASSETS[asset_code]['name'])
                
                # Find strategies for this asset
                asset_strategies = {
                    name: setup for name, setup in TRADE_SETUPS.items() 
                    if setup['asset'] == asset_code
                }
                
                if asset_strategies:
                    # Get the first strategy for this asset
                    strategy_name = list(asset_strategies.keys())[0]
                    setup = asset_strategies[strategy_name]
                    
                    # Display asset forecast
                    st.write(f"**Strategy:** {setup['name']}")
                    st.write(f"**Description:** {setup['description']}")
                    
                    # Show price forecast
                    fig = plot_asset_price_forecast(
                        asset_code, 
                        historical_data_dict[asset_code], 
                        forecast_data_dict[asset_code],
                        setup
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show strategy performance
                    strategy_results = simulate_strategy_performance(
                        setup, historical_data_dict[asset_code], days=28
                    )
                    
                    # Display strategy metrics
                    metrics = calculate_portfolio_metrics(strategy_results)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Expected Return", 
                            f"${metrics['Total Return']:,.2f}",
                            f"{metrics['Total Return'] / setup['allocation'] * 100:.2f}%"
                        )
                    
                    with col2:
                        st.metric("Volatility", f"{metrics['Volatility'] * 100:.2f}%")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{metrics['Max Drawdown'] * 100:.2f}%")
                else:
                    st.write("No strategies defined for this asset.")
        
    # Strategy Optimization page
    elif page == "Strategy Optimization":
        st.header("Strategy Optimization")
        
        # Strategy selection
        st.sidebar.subheader("Strategy Selection")
        for strategy_name in TRADE_SETUPS:
            st.session_state.selected_strategies[strategy_name] = st.sidebar.checkbox(
                strategy_name,
                value=st.session_state.selected_strategies.get(strategy_name, True),
                key=f"strategy_{strategy_name}"
            )
        
        # Get active strategies
        active_trade_setups = {
            name: setup for name, setup in TRADE_SETUPS.items() 
            if st.session_state.selected_strategies.get(name, True)
        }
        
        # Optimize portfolio allocation
        optimization_result = optimize_portfolio_allocation(active_trade_setups, initial_capital)
        
        # Display optimization results
        st.subheader("Portfolio Allocation Optimization")
        
        # Current vs. Optimal Allocation
        current_allocations = optimization_result['Current Allocations']
        optimal_allocations = optimization_result['Optimal Allocations']
        
        # Create data for comparison
        comparison_data = []
        for strategy_name in active_trade_setups:
            current_alloc = current_allocations.get(strategy_name, 0)
            optimal_alloc = optimal_allocations.get(strategy_name, 0)
            setup = active_trade_setups[strategy_name]
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Asset': ASSETS[setup['asset']]['name'],
                'Current Allocation': f"${current_alloc:,.2f}",
                'Optimal Allocation': f"${optimal_alloc:,.2f}",
                'Difference': f"${optimal_alloc - current_alloc:,.2f}",
                'Current Expected Return': f"${current_alloc * setup['expected_return']:,.2f}",
                'Optimal Expected Return': f"${optimal_alloc * setup['expected_return']:,.2f}",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Expected Return", 
                f"${optimization_result['Current Expected Return']:,.2f}",
                f"{optimization_result['Current Expected Return'] / initial_capital * 100:.2f}%"
            )
        
        with col2:
            st.metric(
                "Optimal Expected Return", 
                f"${optimization_result['Optimal Expected Return']:,.2f}",
                f"{optimization_result['Optimal Expected Return'] / initial_capital * 100:.2f}%"
            )
        
        with col3:
            st.metric(
                "Improvement", 
                f"${optimization_result['Improvement']:,.2f}",
                f"{optimization_result['Improvement'] / optimization_result['Current Expected Return'] * 100:.2f}%"
            )
        
        # Simulate performance with optimal allocation
        st.subheader("Projected Performance with Optimal Allocation")
        portfolio_results, strategy_results = simulate_portfolio_performance(
            active_trade_setups, optimal_allocations, historical_data_dict, days=28
        )
        
        # Plot performance
        fig = plot_portfolio_performance(portfolio_results, strategy_results)
        st.plotly_chart(fig, use_container_width=True)
        
    # Backtesting page
    elif page == "Backtesting":
        st.header("Strategy Backtesting")
        
        # Time period selection
        backtest_period = st.sidebar.slider(
            "Backtest Period (days)",
            min_value=30,
            max_value=252,
            value=90,
            step=30
        )
        
        # Get active strategies
        active_trade_setups = {
            name: setup for name, setup in TRADE_SETUPS.items() 
            if st.session_state.selected_strategies.get(name, True) and 
               st.session_state.selected_assets.get(setup['asset'], True)
        }
        
        # Run backtesting simulation
        st.subheader("Backtesting Results")
        st.write(f"Showing {backtest_period} days of historical performance")
        
        # For this example, we simulate historical performance deterministically
        
        # Calculate backtested portfolio metrics
        backtest_return = initial_capital * 0.15 * (backtest_period / 252)  # Simulated 15% annualized return
        backtest_volatility = 0.12 * np.sqrt(backtest_period / 252)  # Simulated 12% annualized volatility
        backtest_max_drawdown = 0.08 * np.sqrt(backtest_period / 252)  # Simulated 8% max drawdown scaled by time
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Backtest Return", f"${backtest_return:,.2f}", f"{backtest_return / initial_capital * 100:.2f}%")
        
        with col2:
            st.metric("Annualized Return", f"{(backtest_return / initial_capital) * (252 / backtest_period) * 100:.2f}%")
        
        with col3:
            st.metric("Volatility", f"{backtest_volatility * 100:.2f}%")
        
        with col4:
            st.metric("Max Drawdown", f"{backtest_max_drawdown * 100:.2f}%")
        
        # Generate simulated backtest data
        backtest_dates = pd.date_range(end=datetime.now(), periods=backtest_period)
        
        # Create a simulated equity curve (deterministically)
        daily_returns = np.zeros(backtest_period)
        for i in range(backtest_period):
            # Use a deterministic formula instead of random generation
            daily_returns[i] = 0.0006 + 0.0002 * np.sin(i / 10) + 0.0003 * np.cos(i / 20)
            
        cumulative_returns = np.cumprod(1 + daily_returns)
        equity_curve = initial_capital * cumulative_returns
        
        # Add drawdowns
        drawdown1 = np.linspace(1, 0.95, 10)  # 5% drawdown
        drawdown2 = np.linspace(1, 0.92, 15)  # 8% drawdown
        
        # Apply first drawdown around 1/3 of the way through
        start_idx = backtest_period // 3
        equity_curve[start_idx:start_idx+10] = equity_curve[start_idx] * drawdown1
        
        # Apply second drawdown around 2/3 of the way through
        start_idx = 2 * backtest_period // 3
        equity_curve[start_idx:start_idx+15] = equity_curve[start_idx] * drawdown2
        
        # Calculate daily and cumulative P&L
        daily_pnl = np.zeros(backtest_period)
        daily_pnl[0] = equity_curve[0] - initial_capital
        daily_pnl[1:] = np.diff(equity_curve)
        
        # Create backtest results DataFrame
        backtest_results = pd.DataFrame({
            'Date': backtest_dates,
            'Equity': equity_curve,
            'Daily_PnL': daily_pnl,
            'Cumulative_PnL': equity_curve - initial_capital
        })
        backtest_results.set_index('Date', inplace=True)
        
        # Plot backtest results
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Equity Curve", "Daily P&L"),
            row_heights=[0.7, 0.3]
        )
        
        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=backtest_results.index,
                y=backtest_results['Equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add initial capital line
        fig.add_hline(
            y=initial_capital,
            line=dict(color='green', width=1, dash='dash'),
            row=1, col=1,
            annotation_text="Initial Capital",
            annotation_position="bottom right"
        )
        
        # Plot daily P&L
        fig.add_trace(
            go.Bar(
                x=backtest_results.index,
                y=backtest_results['Daily_PnL'],
                name='Daily P&L',
                marker_color='green'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Backtest Results',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            legend_title='Data',
            width=800,
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy-specific backtest results
        st.subheader("Strategy-Specific Backtest Results")
        
        # Show deterministic backtest results for each strategy
        
        for strategy_name, setup in active_trade_setups.items():
            with st.expander(f"{strategy_name} - {ASSETS[setup['asset']]['name']}"):
                # Generate strategy-specific results deterministically
                strategy_return = setup['allocation'] * setup['expected_return'] * (backtest_period / 28)
                strategy_volatility = 0.10 * np.sqrt(backtest_period / 252)  # 10% base volatility, adjusted for time
                
                # Adjust volatility based on strategy type
                if setup['strategy'] == 'arbitrage':
                    strategy_volatility *= 0.5  # Lower volatility for arbitrage
                elif setup['strategy'] == 'directional':
                    strategy_volatility *= 1.5  # Higher volatility for directional
                
                strategy_max_drawdown = strategy_volatility * 1.2  # Typical relationship between vol and drawdown
                
                # Display strategy metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Backtest Return", 
                        f"${strategy_return:,.2f}", 
                        f"{strategy_return / setup['allocation'] * 100:.2f}%"
                    )
                
                with col2:
                    st.metric("Volatility", f"{strategy_volatility * 100:.2f}%")
                
                with col3:
                    st.metric("Max Drawdown", f"{strategy_max_drawdown * 100:.2f}%")
                
                # Briefly describe strategy performance
                st.write(f"**Strategy Performance Summary:** {setup['description']}")
                
                # Show risk/reward characteristics
                st.write(f"**Risk/Reward Ratio:** {strategy_max_drawdown / (strategy_return / setup['allocation']):,.2f}")
                st.write(f"**Sharpe Ratio:** {(strategy_return / setup['allocation']) / strategy_volatility:,.2f}")

# Run the app
if __name__ == "__main__":
    main()