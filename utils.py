import yaml
import yfinance as yf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import re
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import time
import itertools
import random

CONFIG_FNAME = 'config.yaml'

def load_config():
    # Load configuration from YAML file
    config_dict = {}
    with open(CONFIG_FNAME, 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_dict['ticker'] = config['trading']['ticker']
        config_dict['start_date'] = config['trading']['start_date']
        config_dict['window_size'] = config['trading']['window_size']
        config_dict['start_balance'] = config['trading']['start_balance']
        config_dict['position_percent'] = config['trading']['position_percent']
        config_dict['min_slope_long'] = config['trading']['min_slope_long']
        config_dict['std_factor'] = config['trading']['std_factor']
        config_dict['optimise'] = config['trading']['optimise']
        
        config_dict['host'] = config['app']['host']
        config_dict['port'] = config['app']['port']
        config_dict['debug'] = config['app']['debug']

    return config_dict

def get_stock_data(ticker, start_date, end_date, frequency='1d'):
    """
    Get stock data from Yahoo Finance.
    
    :param ticker: Stock ticker symbol.
    :param start_date: Start date for the data in 'YYYY-MM-DD' format.
    :param end_date: End date for the data in 'YYYY-MM-DD' format.
    :param frequency: Data frequency ('1d' for daily, '1wk' for weekly, '1mo' for monthly). Default is daily.
    :return: DataFrame with stock data.
    """

    # Fetch data
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(start=start_date, end=end_date, interval=frequency)
    
    # Create DataFrame for Close prices
    raw_data = pd.DataFrame(hist_data['Close'])

    return raw_data

def clean_data(raw_data):
    """
    Clean data.
    """
    raw_data.dropna(inplace=True)
    raw_data.reset_index(drop=True, inplace=True)
    raw_data_np = raw_data.values.reshape((-1, 1))
    return raw_data_np

def get_training_data(arr, start_idx, train_size):
    """
    Get training data.
    """

    return arr[start_idx:start_idx + train_size, :]

def train_model(y):
    """
    Train the model.
    """
    x = np.arange(y.shape[0]).reshape((-1, 1))

    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]

    return model, slope

class TradeManager:
    def __init__(self, balance, position_percent=0.1):
        self.balance = balance
        self.position_percent = position_percent
        self.profit = 0.0
        self.trade_log = []
        self.reset()
        
    def reset(self):
        self.units_held = 0
        self.buy_price = None
        self.long = None
        self.in_trade = False

    def enter_trade(self, buy_price, long):
        self.buy_price = buy_price
        self.units_held = (self.balance * self.position_percent) / buy_price
        self.long = long
        self.in_trade = True

    def exit_trade(self, sell_price):
        if self.long:
            profit = (self.units_held * sell_price) - (self.units_held * self.buy_price)
        else:
            profit = -(self.units_held * sell_price) + (self.units_held * self.buy_price)
        self.balance += profit
        self.update_trade_log(sell_price, profit)
        self.reset()

    def update_trade_log(self, sell_price, profit):
        self.trade_log.append(f'type={self.long} entry price={self.buy_price} exit price={sell_price} profit={profit} balance={self.balance}')

def trade_test(data,
                window_size,
                start_balance,
                position_percent,
                min_slope_long,
                min_slope_short,
                std_factor):
    
    n = data.shape[0]
    tm = TradeManager(start_balance, position_percent)

    balance = start_balance
    balances = []
    preds = np.zeros_like(data)
    ubs = np.zeros_like(data)
    lbs = np.zeros_like(data)

    slope = [0.0]
    slopes = []
    long_entry = []
    short_entry = []
    trade_exit = []


    retrain = True

    for i in range(n):
        if i >= window_size:        
            y = get_training_data(data, i-window_size, window_size)
            if retrain:
                model, slope = train_model(y)
                slope = slope[0]
                y_std = y.std()
                retrain = False
                x_adj = 0

            x = np.arange(y.shape[0]).reshape((-1, 1)) + x_adj
            x_adj += 1
            
            y_preds = model.predict(x)

            y_t = data[i].item()
            y_pred_t = y_preds[-1].item()

            ubs_ = [y.item() + y_std * std_factor for y in y_preds]
            lbs_ = [y.item() - y_std * std_factor for y in y_preds]

            ub_t = ubs_[-1]
            lb_t = lbs_[-1]
            
            preds[i] = y_pred_t
            ubs[i] = ub_t
            lbs[i] = lb_t

            if tm.in_trade:
                if tm.long:
                    sl = max(sl, lb_t)
                else:
                    sl = min(sl, ub_t)

                if (tm.long and (y_t < sl or slope<0)) or (not tm.long and (y_t > sl or slope>0)) or i == (n - 1):
                    tm.exit_trade(y_t)
                    balance = tm.balance
                    retrain = True
                    trade_exit.append(y_t)
                else:
                    trade_exit.append(-1.0)
                long_entry.append(-1.0)
                short_entry.append(-1.0)
                
                    
            else:
                if slope > min_slope_long and y_t > lb_t and y_t < ub_t:
                    tm.enter_trade(y_t, True)
                    sl = lb_t
                    long_entry.append(y_t)
                    short_entry.append(-1.0)
                elif slope < min_slope_short and y_t < ub_t and y_t > lb_t:
                    tm.enter_trade(y_t, False)
                    sl = ub_t
                    short_entry.append(y_t)
                    long_entry.append(-1.0)
                else:
                    long_entry.append(-1.0)
                    short_entry.append(-1.0)
                trade_exit.append(-1.0)

            if y_t > ub_t or y_t < lb_t:
                retrain = True

            slopes.append(slope.item())
            balances.append(balance)

        else:
            slopes.append(0.0)
            balances.append(balance)
            long_entry.append(-1.0)
            short_entry.append(-1.0)
            trade_exit.append(-1.0)

    return balances, preds, ubs, lbs, slopes, long_entry, short_entry, trade_exit

def optimise(data, start_balance, position_percent, n_combinations=100, train_pct=0.8):
    # Define the ranges for a, b, and c
    window_range = range(5, 50, 1)  # Integer from 1 to 100
    slope_range = [round(i, 3) for i in itertools.takewhile(lambda x: x <= 3.0, itertools.count(0.001, 0.001))]  # Float from 0.1 to 1.0, step 0.001
    std_range = [round(i, 1) for i in itertools.takewhile(lambda x: x <= 2.0, itertools.count(0.5, 0.1))]  # Float from 1 to 2, step 0.1

    # Generate all possible combinations
    all_combinations = list(itertools.product(window_range, slope_range, std_range))

    # Randomly select N unique combinations
    if n_combinations <= len(all_combinations):
        random_combinations = random.sample(all_combinations, n_combinations)
    else:
        raise ValueError("N is larger than the total number of unique combinations")
    
    train_size =  int(len(data) * train_pct)
    train_data = data[:train_size, :]
    best_iteration = (-1, -1)
    for i, comb in enumerate(random_combinations):
        if i % 100 == 0:
            print(f'Iteration {i}')
        window_size = comb[0]
        min_slope_long = comb[1]
        min_slope_short = -comb[1]
        std_factor = comb[2]

        balances, _, _, _, _, _, _, _ = trade_test(train_data,
                window_size,
                start_balance,
                position_percent,
                min_slope_long,
                min_slope_short,
                std_factor
            )
        
        if balances[-1] > best_iteration[1]:
            print(i, balances[-1])
            best_iteration = (i, balances[-1])
    
    return random_combinations[best_iteration[0]]
    
# Create DataFrame with data
def create_df_for_plotting(raw_data_np,
                            preds,
                            lbs,
                            ubs,
                            long_entry,
                            short_entry,
                            trade_exit,
                            balances,
                            start_balance,
                            window_size
                            ):
    df = pd.DataFrame()
    df['x'] = np.arange(len(raw_data_np))
    df['y'] = raw_data_np
    df['y_preds'] = preds
    df['y_preds_lb'] = lbs
    df['y_preds_ub'] = ubs
    df['long_entry'] = long_entry
    df['short_entry'] = short_entry
    df['trade_exit'] = trade_exit
    df['balance'] = balances
    df['balance_bah'] = start_balance * np.cumprod(1.0 + pd.DataFrame(raw_data_np).pct_change().iloc[window_size:])
    df = df.iloc[window_size:]
    df['x'] = np.arange(len(raw_data_np)-window_size)
    df.reset_index(drop=True)

    return df

def create_plotly_plots(df):
        """
        Create plotly plots.
        """

        PAPER_BG_COLOR = '#010314'
        PLOT_BG_COLOR = '#C0BDC0' 
        FONT_COLOR = '#C0BDC0'
        GRID_COLOR = '#D9D7D9'
        
        min_x = df['x'].min()
        max_x = df['x'].max()
        min_y = df['y_preds_lb'].min()
        max_y = df['y_preds_ub'].max()

        # Plot the results using Plotly
        fig1 = go.Figure()

        # Prices
        fig1.add_trace(go.Scatter(x=df['x'], y=df['y'], name='Price', mode='lines', line=dict(width=1.0, color='magenta')))

        # Predictions
        fig1.add_trace(go.Scatter(x=df['x'], y=df['y_preds'], name='Predicted Price', mode='lines', line=dict(width=1.0, color='orange', dash='dot')))
        fig1.add_trace(go.Scatter(x=df['x'], y=df['y_preds_lb'], name='Lower Bound', mode='lines', line=dict(width=1.0, color='orange')))
        fig1.add_trace(go.Scatter(x=df['x'], y=df['y_preds_ub'], name='Upper Bound', mode='lines', line=dict(width=1.0, color='orange')))

        # Trade management
        fig1.add_trace(go.Scatter(x=df['x'], y=df['long_entry'], name='Long Entry', mode='markers', line=dict(width=1.0, color='green'), marker=dict(symbol='triangle-up')))
        fig1.add_trace(go.Scatter(x=df['x'], y=df['short_entry'], name='Short Entry', mode='markers', line=dict(width=1.0, color='red'), marker=dict(symbol='triangle-down')))
        fig1.add_trace(go.Scatter(x=df['x'], y=df['trade_exit'], name='Trade Exit', mode='markers', line=dict(width=1.0, color='blue'), marker=dict(symbol='cross')))

        fig1.update_layout(width=1000, 
                        height=600, 
                        paper_bgcolor=PAPER_BG_COLOR,
                        plot_bgcolor=PLOT_BG_COLOR,
                        font=dict(color=FONT_COLOR),
                        title={'text':'Price Predictions', 
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'}, 
                        yaxis_title='Price', 
                        xaxis=dict(range=[min_x, max_x], gridcolor=GRID_COLOR),
                        yaxis=dict(range=[min_y, max_y], gridcolor=GRID_COLOR),
                        legend ={'title_text': '',
                                'bgcolor': 'rgba(0,0,0,0)',
                                'orientation': 'h',
                                'yanchor': 'bottom',
                                'y': -0.2,
                                'xanchor': 'center',
                                'x': 0.5})

        # Plot the results using Plotly
        fig2 = go.Figure()

        # Balance
        fig2.add_trace(go.Scatter(x=df['x'], y=df['balance'], name='Balance (Active Trading)', mode='lines', line=dict(width=1.0, color='blue')))
        fig2.add_trace(go.Scatter(x=df['x'], y=df['balance_bah'], name='Balance (Buy and Hold)', mode='lines', line=dict(width=1.0, color='green')))
        fig2.update_layout(width=1000, 
                        height=400, 
                        paper_bgcolor=PAPER_BG_COLOR,
                        plot_bgcolor=PLOT_BG_COLOR,
                        font=dict(color=FONT_COLOR),
                        title={'text':'Balances', 
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},  
                        yaxis_title='Balance',
                        xaxis=dict(gridcolor=GRID_COLOR),
                        yaxis=dict(gridcolor=GRID_COLOR),
                        legend ={'title_text': '',
                                'bgcolor': 'rgba(0,0,0,0)',
                                'orientation': 'h',
                                'yanchor': 'bottom',
                                'y': -0.4,
                                'xanchor': 'center',
                                'x': 0.5}
                        )

        return fig1, fig2