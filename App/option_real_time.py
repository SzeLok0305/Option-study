import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import datetime
from pandas_datareader import data as pdr
import time

if 'sigma' not in st.session_state:
    st.session_state.sigma = None
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = None
if 'current_price' not in st.session_state:
    st.session_state.current_price = None
if 'expiration' not in st.session_state:
    st.session_state.expiration = None
if 'T_expiration' not in st.session_state:
    st.session_state.T_expiration = None
if 'available_strikes' not in st.session_state:
    st.session_state.available_strikes = None

def black_scholes(S, K, T, r, q, sigma, option_type='call'):
    #Pleas annualize everything before input:)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * q * np.exp(-q * T) * norm.cdf(d2) 
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    return option_price, delta, gamma, theta, vega

def Historical_volatility(df,window=20):
    rolling_vol = df['log return'].rolling(window=window).std() * np.sqrt(252)
    return rolling_vol.iloc[-1]

# Parkinson Volatility
def parkinson_volatility(df, window=20):
    high_low_ratio = np.log(df['High'] / df['Low'])
    vol = np.sqrt((1 / (4 * window * np.log(2))) * 
                  (high_low_ratio ** 2).rolling(window=window).sum() * 252)
    return vol.iloc[-1]

# Garman-Klass Volatility
def garman_klass_volatility(df, window=20):
    log_hl = np.log(df['High'] / df['Low']) ** 2
    log_co = np.log(df['Close'] / df['Open']) ** 2
    vol = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
                  .rolling(window=window).mean() * 252)
    return vol.iloc[-1]

# Rogers-Satchell Volatility
def rogers_satchell_volatility(df, window=20):
    log_ho = np.log(df['High'] / df['Open'])
    log_lo = np.log(df['Low'] / df['Open'])
    log_hc = np.log(df['High'] / df['Close'])
    log_lc = np.log(df['Low'] / df['Close'])
    vol = np.sqrt((log_ho * (log_ho - log_hc) + 
                   log_lo * (log_lo - log_lc))
                  .rolling(window=window).mean() * 252)
    return vol.iloc[-1]

def initialize_parameters(ticker, maturity_period, look_back_period=20):
    Start_date = datetime.date.today() - datetime.timedelta(days=2*look_back_period)
    df_Stock_price = yf.download(ticker, start=Start_date)

    df_Stock_price['log return'] = np.log(df_Stock_price[f'Close'] / df_Stock_price['Close'].shift(1))

    # Historical Volatility
    sigma = Historical_volatility(df_Stock_price,look_back_period)
    
    df_risk_free_rate = pdr.DataReader("DGS3MO", 'fred', Start_date).dropna()
    risk_free_rate = (df_risk_free_rate["DGS3MO"].iloc[-1])/100
    
    # Get expiration date and strikes
    symbol = yf.Ticker(ticker)
    expiration = symbol.options[maturity_period-1]
    current_price = symbol.info['currentPrice']
    #We need to track time down to second lets goooooooooooooooooooooo(beware of time zone tho)
    T_expiration = max(((pd.Timestamp(expiration) - pd.Timestamp(datetime.datetime.now())).total_seconds()) / (360 * 24 * 60 * 60), 1/(360 * 24 * 60 * 60))
    # Get available strikes
    option_chain = symbol.option_chain(expiration)
    available_strikes = sorted(option_chain.puts['strike'].unique())
    
    return sigma, risk_free_rate, current_price, expiration, T_expiration, available_strikes

def get_relevant_strikes(available_strikes, current_price, price_range_percent):
    price_range = current_price * (price_range_percent/100)
    min_strike = current_price - price_range
    max_strike = current_price + price_range
    
    relevant_strikes = [strike for strike in available_strikes 
                       if min_strike <= strike <= max_strike]
    return relevant_strikes

def update_price_table(ticker):
    symbol = yf.Ticker(ticker)
    current_price = symbol.info['currentPrice']
    dividend_yield = symbol.info.get('dividendYield', 0)
    
    # Get option chain for implied volatilities
    option_chain = symbol.option_chain(st.session_state.expiration)
    calls_iv = pd.Series(option_chain.calls.set_index('strike')['impliedVolatility'])
    calls_bid = pd.Series(option_chain.calls.set_index('strike')['bid'])
    calls_ask = pd.Series(option_chain.calls.set_index('strike')['ask'])

    puts_iv = pd.Series(option_chain.puts.set_index('strike')['impliedVolatility'])
    puts_bid = pd.Series(option_chain.puts.set_index('strike')['bid'])
    puts_ask = pd.Series(option_chain.puts.set_index('strike')['ask'])

    relevant_strikes = get_relevant_strikes(st.session_state.available_strikes, 
                                          current_price, 
                                          price_range_percent)
    
    call_data = []
    put_data = []
    for strike in relevant_strikes:
        # Calculate Call Greeks
        call_price, call_delta, call_gamma, call_theta, call_vega = black_scholes(
            current_price, strike, st.session_state.T_expiration, 
            st.session_state.risk_free_rate, dividend_yield, st.session_state.sigma, 'call')
        
        # Calculate Put Greeks
        put_price, put_delta, put_gamma, put_theta, put_vega = black_scholes(
            current_price, strike, st.session_state.T_expiration,
            st.session_state.risk_free_rate, dividend_yield, st.session_state.sigma, 'put')
        
        call_data.append({
            'Strike': strike,
            'Theoretical price': call_price,
            'Bids': calls_bid.get(strike, np.nan),
            'Ask': calls_ask.get(strike, np.nan),
            'Delta': call_delta,
            'Gamma': call_gamma,
            'Theta': call_theta,
            'Vega': call_vega,
            'IV': calls_iv.get(strike, np.nan)
        })
        
        put_data.append({
            'Strike': strike,
            'Theoretical price': put_price,
            'Bids': puts_bid.get(strike, np.nan),
            'Ask': puts_ask.get(strike, np.nan),
            'Delta': put_delta,
            'Gamma': put_gamma,
            'Theta': put_theta,
            'Vega': put_vega,
            'IV': puts_iv.get(strike, np.nan)
        })
    
    df_calls = pd.DataFrame(call_data)
    df_puts = pd.DataFrame(put_data)
    
    # Find ATM strike
    atm_strike = min(df_calls['Strike'], key=lambda x: abs(x - current_price))
    
    def highlight_atm(row):
        if row['Strike'] == atm_strike:
            return ['background-color: rgba(0, 255, 0, 0.5)'] * len(row)
        return [''] * len(row)
    
    # Style both dataframes
    styled_df_calls = df_calls.style\
        .apply(highlight_atm, axis=1)\
        .format({
            'Strike': '{:.2f}',
            'Theoretical price': '{:.2f}',
            'Bids': '{:.2f}',
            'Asks': '{:.2f}',
            'Delta': '{:.4f}',
            'Gamma': '{:.4f}',
            'Theta': '{:.4f}',
            'Vega': '{:.4f}',
            'IV': '{:.2%}'
        })
    
    styled_df_puts = df_puts.style\
        .apply(highlight_atm, axis=1)\
        .format({
            'Strike': '{:.2f}',
            'Theoretical price': '{:.2f}',
            'Bids': '{:.2f}',
            'Asks': '{:.2f}',
            'Delta': '{:.4f}',
            'Gamma': '{:.4f}',
            'Theta': '{:.4f}',
            'Vega': '{:.4f}',
            'IV': '{:.2%}'
        })
    
    return styled_df_calls, styled_df_puts, current_price

st.title('Option Price Update')

ticker = st.text_input('Enter ticker symbol', 'NVDA')

if 'previous_period' not in st.session_state:
    st.session_state.previous_period = 1
period_to_expiration = st.slider("Expiration period:", 1, 5, 1)

price_range_percent = st.slider('Price range (%)', 1, 50, 5)


# Initialize session states if they don't exist
if 'previous_ticker' not in st.session_state:
    st.session_state.previous_ticker = ticker
    
if 'previous_price' not in st.session_state:
    st.session_state.previous_price = None

if 'previous_expiration' not in st.session_state:
    st.session_state.previous_expiration = None

# Check if ticker has changed
if st.session_state.previous_ticker != ticker or period_to_expiration != st.session_state.previous_period:
    st.session_state.sigma, st.session_state.risk_free_rate, st.session_state.current_price, \
    st.session_state.expiration, st.session_state.T_expiration, st.session_state.available_strikes = initialize_parameters(ticker,maturity_period=period_to_expiration)
    # Reset previous price when ticker changes
    st.session_state.previous_price = st.session_state.current_price
    st.session_state.previous_ticker = ticker
    st.session_state.previous_period = period_to_expiration
# Initial initialization if parameters don't exist
elif (st.session_state.sigma is None or 
      st.session_state.risk_free_rate is None or 
      st.session_state.current_price is None):
    
    st.session_state.sigma, st.session_state.risk_free_rate, st.session_state.current_price, \
    st.session_state.expiration, st.session_state.T_expiration, st.session_state.available_strikes = initialize_parameters(ticker,maturity_period=period_to_expiration)
    st.session_state.previous_price = st.session_state.current_price


# Calculate price change
if st.session_state.previous_price is not None:
    price_change = st.session_state.current_price - st.session_state.previous_price
    price_change_pct = (price_change / st.session_state.previous_price) * 100
else:
    price_change = 0
    price_change_pct = 0
# Update previous price for next iteration
st.session_state.previous_price = st.session_state.current_price

#Table
Update_duration = 5
placeholder = st.empty()
while True:
    with placeholder.container():
        df_calls, df_puts, current_price = update_price_table(ticker)
        # Center align all content
        st.markdown(
            """
            <style>
            .centered {
                text-align: center;
                margin: 0 auto;
            }
            .stDataFrame {
                margin: 0 auto;
            }
            div[data-testid="stHorizontalBlock"] {
                text-align: center;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )

        st.markdown(f"<h4 class='centered'>Risk-Free Rate: {st.session_state.risk_free_rate:.2%}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 class='centered'>Expiration Date: {st.session_state.expiration}</h4>", unsafe_allow_html=True)
        st.markdown("""
                <hr style="height:2px;
               width:80%;
               border-width:0;
               color:gray;
               background-color:gray;
               margin: 20px auto;">
                """, unsafe_allow_html=True)
        if price_change > 0:
            color = "green"
            symbol = "↑"
        elif price_change < 0:
            color = "red"
            symbol = "↓"
        else:
            color ='grey'
            symbol = " "
        st.markdown(f"""
            <h3 style='text-align:center'>
            Current Price: <span style='color:{color}'>${current_price:.2f}</span> 
            <span style='color:{color}'>{symbol} ${abs(price_change):.2f} ({price_change_pct:.2f}%)</span>
            </h3>
            """, unsafe_allow_html=True)
        st.markdown(f"<h3 class='centered'>Historical Volatility (20 days)): {st.session_state.sigma:.2%}</h3>", unsafe_allow_html=True)

        
        st.markdown("<h4 class='centered'>Call Options</h4>", unsafe_allow_html=True)
        st.dataframe(df_calls, use_container_width=True, hide_index=True)
            
        st.markdown("<h4 class='centered'>Put Options</h4>", unsafe_allow_html=True)
        st.dataframe(df_puts, use_container_width=True, hide_index=True)
        
        st.markdown(f"<p class='centered'>Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}</p>", 
                   unsafe_allow_html=True)
    

    time.sleep(Update_duration)