import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import base64


def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, "rb") as bin_file:
        data = bin_file.read()
    return base64.b64encode(data).decode()  # Corrected this line

# Path to the locally stored QR code image
qr_code_path = "qrcode.png"  # Ensure the image is in your app directory

# Convert image to base64
qr_code_base64 = get_base64_of_bin_file(qr_code_path)

# Custom CSS to position the QR code close to the top-right corner under the "Deploy" area
st.markdown(
    f"""
    <style>
    .qr-code {{
        position: fixed;  /* Keeps the QR code fixed in the viewport */
        top: 10px;       /* Sets the distance from the top of the viewport */
        right: 10px;     /* Sets the distance from the right of the viewport */
        width: 200px;    /* Adjusts the width of the QR code */
        z-index: 100;    /* Ensures the QR code stays above other elements */
    }}
    </style>
    <img src="data:image/png;base64,{qr_code_base64}" class="qr-code">
    """,
    unsafe_allow_html=True
)



def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

# Calculate EMA
def calculate_ema(prices, period):
    alpha = 2 / (period + 1)
    ema = prices.ewm(span=period, adjust=False).mean()
    return ema

# Calculate RSI
def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_stock_data(stock_symbol, period):
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, period=period, interval='1d')
    
    # Ensure no NaN values in 'Close' column by dropping rows
    stock_data = stock_data.dropna(subset=['Close'])
    
    # Calculate VWAP
    stock_data['Typical_Price'] = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
    stock_data['Cumulative_Volume'] = stock_data['Volume'].cumsum()
    stock_data['Cumulative_TPV'] = (stock_data['Typical_Price'] * stock_data['Volume']).cumsum()
    stock_data['VWAP'] = stock_data['Cumulative_TPV'] / stock_data['Cumulative_Volume']
    
    # Calculate EMA 50
    stock_data['EMA_50'] = calculate_ema(stock_data['Close'], 50)
    
    # Calculate RSI
    stock_data['RSI'] = calculate_rsi(stock_data['Close'], 14)
    
    # Calculate ATR
    stock_data['ATR'] = calculate_atr(stock_data)
    
    # Ensure no NaN values in calculated columns
    stock_data = stock_data.dropna(subset=['VWAP', 'EMA_50', 'RSI', 'ATR'])
    
    # Generate Buy/Sell Signals
    stock_data['Signal'] = 0  # Default no signal
    
    # Buy when price is above VWAP, between VWAP and EMA 50, and RSI is below 40
    stock_data.loc[(stock_data['Close'] > stock_data['VWAP']) &
                   (stock_data['Close'] < stock_data['EMA_50']) &
                   (stock_data['RSI'] < 40), 'Signal'] = 1  # Buy Signal
    
    # Sell when price is below VWAP, between VWAP and EMA 50, and RSI is above 60
    stock_data.loc[(stock_data['Close'] < stock_data['VWAP']) &
                   (stock_data['Close'] > stock_data['EMA_50']) &
                   (stock_data['RSI'] > 60), 'Signal'] = -1  # Sell Signal
    
    return stock_data

def backtest_strategy(data, initial_capital=100000, transaction_fee_pct=0.001):
    """
    Backtest the trading strategy with transaction fees and ATR-based take profit/stop loss
    
    Parameters:
    data: DataFrame with price data and signals
    initial_capital: Starting capital for the simulation
    transaction_fee_pct: Transaction fee as a percentage of trade value
    """
    position = 0  # 0: no position, 1: long position
    capital = initial_capital
    shares = 0
    entry_price = 0
    trades = []
    
    # ATR multipliers for take profit and stop loss
    tp_multiplier = 2.0  # Take profit at 2 * ATR
    sl_multiplier = 1.0  # Stop loss at 1 * ATR
    
    for i in range(1, len(data)):
        current_price = data['Close'].iloc[i]
        current_atr = data['ATR'].iloc[i]
        
        if position == 0:  # No position
            if data['Signal'].iloc[i] == 1:  # Buy signal
                max_shares = int(capital * 0.95 / current_price)  # Use 95% of capital max
                shares = max_shares
                transaction_fee = shares * current_price * transaction_fee_pct
                capital -= (shares * current_price + transaction_fee)
                entry_price = current_price
                position = 1
                
                trades.append({
                    'date': data.index[i],
                    'type': 'buy',
                    'price': current_price,
                    'shares': shares,
                    'fee': transaction_fee,
                    'capital': capital
                })
        
        elif position == 1:  # Long position
            take_profit = entry_price + (current_atr * tp_multiplier)
            stop_loss = entry_price - (current_atr * sl_multiplier)
            
            # Check for exit conditions
            exit_signal = (
                data['Signal'].iloc[i] == -1 or  # Sell signal
                current_price >= take_profit or   # Take profit hit
                current_price <= stop_loss        # Stop loss hit
            )
            
            if exit_signal:
                transaction_fee = shares * current_price * transaction_fee_pct
                capital += (shares * current_price - transaction_fee)
                
                exit_type = 'sell_signal' if data['Signal'].iloc[i] == -1 else (
                    'take_profit' if current_price >= take_profit else 'stop_loss'
                )
                
                trades.append({
                    'date': data.index[i],
                    'type': exit_type,
                    'price': current_price,
                    'shares': shares,
                    'fee': transaction_fee,
                    'capital': capital
                })
                
                shares = 0
                position = 0
    
    # Calculate performance metrics
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        total_trades = len(trades_df) // 2  # Divide by 2 since each complete trade has buy and sell
        profitable_trades = sum(1 for i in range(0, len(trades_df)-1, 2) 
                              if trades_df.iloc[i+1]['capital'] > trades_df.iloc[i]['capital'])
        total_fees = trades_df['fee'].sum()
        final_return = ((capital - initial_capital) / initial_capital) * 100
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    else:
        total_trades = profitable_trades = total_fees = final_return = win_rate = 0
    
    return {
        'final_capital': capital,
        'total_return_pct': final_return,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'total_fees': total_fees,
        'trades': trades
    }

def main():
    st.title('Stock Analysis with VWAP, EMA, RSI, and Backtesting')
    
    # Dropdown for stock selection (Top 50 Stocks)
    stocks = {
    'Apple Inc. (AAPL)': 'AAPL',
    'Microsoft Corp (MSFT)': 'MSFT',
    'Alphabet Inc. (GOOGL)': 'GOOGL',
    'Amazon.com Inc. (AMZN)': 'AMZN',
    'Tesla Inc. (TSLA)': 'TSLA',
    'Meta Platforms Inc. (META)': 'META',
    'NVIDIA Corporation (NVDA)': 'NVDA',
    'Berkshire Hathaway Inc. (BRK.B)': 'BRK.B',
    'Visa Inc. (V)': 'V',
    'Johnson & Johnson (JNJ)': 'JNJ',
    'Procter & Gamble Co. (PG)': 'PG',
    'Walmart Inc. (WMT)': 'WMT',
    'UnitedHealth Group Incorporated (UNH)': 'UNH',
    'Home Depot Inc. (HD)': 'HD',
    'Mastercard Incorporated (MA)': 'MA',
    'Coca-Cola Company (KO)': 'KO',
    'PepsiCo, Inc. (PEP)': 'PEP',
    'Intel Corporation (INTC)': 'INTC',
    'Adobe Inc. (ADBE)': 'ADBE',
    'Salesforce Inc. (CRM)': 'CRM',
    'Qualcomm Inc. (QCOM)': 'QCOM',
    'Netflix Inc. (NFLX)': 'NFLX',
    'AbbVie Inc. (ABBV)': 'ABBV',
    'Texas Instruments Incorporated (TXN)': 'TXN',
    'Bristol-Myers Squibb Company (BMY)': 'BMY',
    'Chevron Corporation (CVX)': 'CVX',
    'Exxon Mobil Corporation (XOM)': 'XOM',
    'Starbucks Corporation (SBUX)': 'SBUX',
    'T-Mobile US Inc. (TMUS)': 'TMUS',
    'Advanced Micro Devices, Inc. (AMD)': 'AMD',
    'PayPal Holdings Inc. (PYPL)': 'PYPL',
}

    
    # Input parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_stock = st.selectbox('Select Stock', list(stocks.keys()))
    with col2:
        initial_capital = st.number_input('Initial Capital ($)', value=10000, step=10000)
    with col3:
        transaction_fee_pct = st.number_input('Transaction Fee (%)', value=0.1, step=0.05) / 100

    period = '1y'
    stock_symbol = stocks[selected_stock]
    
    # Fetch and process data
    stock_data = fetch_stock_data(stock_symbol, period)
    
    # Run backtest
    backtest_results = backtest_strategy(
        stock_data, 
        initial_capital=initial_capital,
        transaction_fee_pct=transaction_fee_pct
    )
    
    # Display backtest results
    st.subheader('Backtest Results')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Final Capital', f"${backtest_results['final_capital']:,.2f}")
    col2.metric('Total Return', f"{backtest_results['total_return_pct']:.2f}%")
    col3.metric('Win Rate', f"{backtest_results['win_rate']:.2f}%")
    col4.metric('Total Trades', backtest_results['total_trades'])
    
    st.metric('Total Fees Paid', f"${backtest_results['total_fees']:,.2f}")
    
    # Plotting
    plt.figure(figsize=(14, 12))
    
    # Price Plot
    plt.subplot(2, 1, 1)
    plt.plot(stock_data.index, stock_data['Close'], label=f'{selected_stock} Price', color='blue')
    plt.plot(stock_data.index, stock_data['VWAP'], label='VWAP', color='orange', linestyle='--')
    plt.plot(stock_data.index, stock_data['EMA_50'], label='EMA 50', color='purple', linestyle='-.')
    
    # Plot Buy and Sell points from backtest
    trades_df = pd.DataFrame(backtest_results['trades'])
    if len(trades_df) > 0:
        buy_points = trades_df[trades_df['type'] == 'buy']
        sell_points = trades_df[trades_df['type'].isin(['sell_signal', 'take_profit', 'stop_loss'])]
        
        plt.scatter(buy_points['date'], buy_points['price'], 
                   marker='^', color='g', s=100, label='Buy')
        plt.scatter(sell_points['date'], sell_points['price'], 
                   marker='v', color='r', s=100, label='Sell')
    
    plt.title(f'{selected_stock} Price with Signals (1 Year)')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.grid(True)
    
    # RSI Plot
    plt.subplot(2, 1, 2)
    plt.plot(stock_data.index, stock_data['RSI'], label='RSI', color='purple')
    plt.axhline(60, linestyle='--', alpha=0.5, color='red')
    plt.axhline(40, linestyle='--', alpha=0.5, color='green')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend()
    plt.grid(True)
    
    
    
    plt.tight_layout()
    st.pyplot(plt)
    
    # Display detailed trade history
    if len(trades_df) > 0:
        st.subheader('Trade History')
        trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
        trades_df['price'] = trades_df['price'].round(2)
        trades_df['fee'] = trades_df['fee'].round(2)
        trades_df['capital'] = trades_df['capital'].round(2)
        st.dataframe(trades_df)

if __name__ == "__main__":
    main()