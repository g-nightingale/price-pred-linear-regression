from flask import Flask, render_template, request
import pandas as pd
import plotly
import json
import utils as ut
import re

app = Flask(__name__)

def get_data_for_template(ticker, 
                        start_date,
                        end_date,
                        window_size,
                        start_balance,
                        position_percent,
                        min_slope_long,
                        std_factor,
                        optimise):
    

    raw_data = ut.get_stock_data(ticker, start_date, end_date)
    raw_data_np = ut.clean_data(raw_data)

    if optimise:
        best_parameters = ut.optimise(raw_data_np, 
                                    start_balance, 
                                    position_percent, 
                                    n_combinations=200, 
                                    train_pct=0.7)
        window_size = best_parameters[0]
        min_slope_long = best_parameters[1]
        std_factor = best_parameters[2]

    balances, \
    preds, \
    ubs, \
    lbs, \
    slopes, \
    long_entry, \
    short_entry, \
    trade_exit = ut.trade_test(raw_data_np,
                            window_size,
                            start_balance,
                            position_percent,
                            min_slope_long,
                            -min_slope_long,
                            std_factor)

    df = ut.create_df_for_plotting(raw_data_np,
                                    preds,
                                    lbs,
                                    ubs,
                                    long_entry,
                                    short_entry,
                                    trade_exit,
                                    balances,
                                    start_balance,
                                    window_size)
    
    trading_fig, balance_fig = ut.create_plotly_plots(df)
    
    # Convert the figures to JSON for rendering in the HTML template
    trading_fig_json = json.dumps(trading_fig, cls=plotly.utils.PlotlyJSONEncoder)
    balance_fig_json = json.dumps(balance_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return trading_fig_json, balance_fig_json, window_size, min_slope_long, std_factor

# Route for serving the dashboard
# http://127.0.0.1:5000/
@app.route('/', methods=['GET', 'POST'])
def dashboard():

    if request.method == 'POST':
        ticker = re.sub(r'[^a-zA-Z. ]', '', request.form['ticker'])
        window_size = int(request.form['window_size'])
        text_start_date = request.form['start_date']
        min_slope_long = float(request.form['min_slope_long'])
        std_factor = float(request.form['std_factor'])
        optimise_text = request.form['optimise']

        # Load config dict
        config = ut.load_config()
        start_balance = config['start_balance']
        position_percent = config['position_percent']

        # Ensure text_start_date is valid, otherwise set to config default
        try:
            pd.Timestamp(text_start_date)
        except ValueError:
            text_start_date = config['start_date']

    else:
        # Load config dict
        config = ut.load_config()
        ticker = config['ticker']
        text_start_date = config['start_date']
        window_size = config['window_size']
        start_balance = config['start_balance']
        position_percent = config['position_percent']
        min_slope_long = config['min_slope_long']
        std_factor = config['std_factor']
        optimise_text = config['optimise']

    # Time period
    start_date = pd.Timestamp(text_start_date)
    end_date = pd.Timestamp.now()

    optimise = True if optimise_text == 'y' else False 

    # Get figures
    trading_fig_json, \
    balance_fig_json, \
    window_size, \
    min_slope_long, \
    std_factor = get_data_for_template(ticker, 
                                start_date,
                                end_date,
                                window_size,
                                start_balance,
                                position_percent,
                                min_slope_long,
                                std_factor,
                                optimise)
       
    # Render the template, passing the data and JSON strings
    return render_template('trading_with_linear_regression.html',
                            trading_fig_json=trading_fig_json,
                            balance_fig_json=balance_fig_json,
                            ticker=ticker,
                            start_date=text_start_date,
                            window_size=window_size,
                            min_slope_long=min_slope_long,
                            std_factor=std_factor,
                            optimise=optimise_text)

if __name__ == '__main__':
    # Load config dict
    config = ut.load_config()
    app.run(host=config['host'], port=config['port'], debug=config['debug'])
