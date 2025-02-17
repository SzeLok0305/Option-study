#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:08:43 2024

@author: arnold
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def black_scholes(S, K, T, r, q, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r - q  + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

st.title('Black-Scholes Option Price Calculator')

# Input parameters
S = st.number_input('Stock Price', min_value=0.01, value=100.0)
K = st.number_input('Strike Price', min_value=0.01, value=80.0)
T = st.slider('Time to Maturity (in years)', min_value=0.1, max_value=3.0, value=0.5, step = 0.01)
r = st.slider('Risk-free Rate(%)', min_value=0.0, max_value=20.0, value=5.0, step = 0.5)
q = st.slider("Dividend(%)", min_value=0.0, max_value=20.0, value=3.0, step = 0.5)
sigma = st.slider('Volatility(%)', min_value=0.0, max_value=100.0, value=20.0,step = 1.0)

# Calculate option prices
call_price = black_scholes(S, K, T, r/100, q/100, sigma/100, 'call')
put_price = black_scholes(S, K, T, r/100, q/100, sigma/100, 'put')

# Display results
st.subheader('Option Prices')
#st.write(f'Call Option Price: ${call_price:.2f}')
#st.write(f'Put Option Price: ${put_price:.2f}')
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-around;">
        <div style="background-color: green; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <p style="margin: 0; font-size: 20px;">Call</p>
            <p style="margin: 0; font-size: 24px;">${call_price:.2f}</p>
        </div>
        <div style="background-color: red; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <p style="margin: 0; font-size: 20px;">Put</p>
            <p style="margin: 0; font-size: 24px;">${put_price:.2f}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

r_SK = np.linspace(0,2,200)

data = pd.DataFrame({
    'S/K': r_SK,
    'C/K': black_scholes(r_SK, 1, 1.0, 1.0, 0.05, 0.2, 'call'),
    'P/K': black_scholes(r_SK, 1, 1.0, 1.0, 0.05, 0.2, 'put')
})

data['C/K'] = black_scholes(r_SK, 1, T, r/100, q/100, sigma/100, 'call')
data['P/K'] = black_scholes(r_SK, 1, T,  r/100, q/100, sigma/100, 'put')

fig_call = px.line(data, x='S/K', y='C/K', title='call price')
fig_call.update_layout(title={'text': '<b style="text-align: center;">Call Price/Stock Price </b>', 'x': 0.5})

fig_put = px.line(data, x='S/K', y='P/K', title='put price')
fig_put.update_layout(title={'text': '<b style="text-align: center;">Put Price/Stock Price</b>', 'x': 0.5})

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_call, use_container_width=True)
with col2:
    st.plotly_chart(fig_put, use_container_width=True)

####%%##################################################################################


#Heatmap plot:

T_range = np.linspace(0.1,5,5)
sigma_range = np.linspace(0, 1, 5)
T_mesh, sigma_mesh = np.meshgrid(T_range, sigma_range)

r_heat = st.number_input('risk-free rate(%)', min_value=0.0, value=5.0)
q_heat = st.number_input('dividend(%)', min_value=0.0, value=2.0)
#T_heat = st.slider('time to Maturity (in years)', min_value=0.1, max_value=3.0, value=0.5, step = 0.01)
TV_heat = st.number_input('Time value threshold relative to stock price (%)', min_value=0.0, max_value=10.0, value=5.0)
S_K_heat = st.slider("Moneyness (S/K)", min_value=0.0, max_value=2.0, value=0.80, step = 0.1)

call_prices_heat = black_scholes(S_K_heat, 1 , T_mesh, r_heat/100, q_heat/100, sigma_mesh, 'call')
put_prices_heat = black_scholes(S_K_heat, 1 , T_mesh, r_heat/100, q_heat/100, sigma_mesh, 'put')


def create_heatmap(prices, title, option_type):
    fig = go.Figure(data=go.Heatmap(
        z=prices,
        x=T_range,
        y=sigma_range,
        colorscale='Greys',
        showscale=False
    ))

    # Add text annotations
    for i in range(len(sigma_range)):
        for j in range(len(T_range)):
                if option_type == 'Call':
                    intrinsic_value = max(S_K_heat-1, 0)
                elif option_type == 'Put':
                    intrinsic_value = max(1-S_K_heat, 0)
                time_value = prices[j, i] - intrinsic_value
                if time_value > TV_heat/100 * S_K_heat:
                    text_color = 'Green'
                else:
                    text_color = 'red'
                fig.add_annotation(
                x=T_range[j],
                y=sigma_range[i],
                text=f"{prices[i, j]:.2f}",
                showarrow=False,
                font=dict(size=16, color=text_color)
            )

    fig.update_layout(
        title={
            'text': f'<b>{title}</b>',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Time to maturity',
        yaxis_title="Volatility",
        height=500,
        width=2000
    )

    return fig

st.markdown(f"""
<h3 style='text-align: center; color: #ffffff; margin-top: 20px; margin-bottom: 20px;'>
    Time value > {TV_heat}% of stock price: <span style='color: green;'>green</span> | 
    < {TV_heat}%: <span style='color: red;'>red</span>
</h3>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    fig_call_heat = create_heatmap(call_prices_heat, 'Call/Stock Price Heatmap', 'Call')
    st.plotly_chart(fig_call_heat, use_container_width=True)

with col4:
    fig_put_heat = create_heatmap(put_prices_heat, 'Put/Stock Price Heatmap', 'Put')
    st.plotly_chart(fig_put_heat, use_container_width=True)

