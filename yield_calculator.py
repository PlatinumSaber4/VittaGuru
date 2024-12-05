import numpy as np
import pandas as pd
from scipy.stats import linregress
import json
import scipy
from datetime import timedelta
import traceback

def retreive_date_interval(dohlcav_mpnxp_data, start_date, end_date):
  start_end_date_range_list=[]
  dncp_start_end_date_range_list=[]

  try:
      idx_1=dohlcav_mpnxp_data.index[dohlcav_mpnxp_data['DCP_date_current_period']==start_date].values[0]
      idx_2=dohlcav_mpnxp_data.index[dohlcav_mpnxp_data['DCP_date_current_period']==end_date].values[0]
      start_end_date_range_list=list(dohlcav_mpnxp_data.loc[idx_1:idx_2,'DCP_date_current_period'].values)
      dncp_start_end_date_range_list=list(dohlcav_mpnxp_data.loc[idx_1:idx_2,'DNCP_day_number_current_period'].values)
  except:
     pass
  return start_end_date_range_list,dncp_start_end_date_range_list

def plot_polynomial_3_regression(dates,yields,dates_num,yield_mode,pred_type):
  z=np.polyfit(dates_num,yields,3)
  p = np.poly1d(z)
  plt_title = "polynomial regression curve for "+yield_mode+" for "+pred_type
  return p(dates_num), plt_title

def curve_fit_exponential_regression(dates,yields,dates_num,yield_mode,pred_type):
  a,b= np.polyfit(dates_num,np.log(yields),1)
  a_guess=np.exp(b)
  b_guess=a
  popt, pcov =scipy.optimize.curve_fit(lambda t, a, b: a * np.exp(b * t),dates_num,yields,p0=(a_guess,b_guess),maxfev=1000000)
  factor=popt[0]
  exponent=popt[1]
  y_fitted=[]
  for i in dates_num:
    y_fitted.append(factor*np.exp(exponent* i))
  plt_title='exponential regression using curve fit function for '+yield_mode+" for "+pred_type
  day_number_cont=np.arange(dates_num[0],dates_num[-1]+1)
  y_thirty_days_yield=[]
  for i in day_number_cont:
    y_thirty_days_yield.append(factor*np.exp(exponent* i))
  thirty_days_yield=[]
  for i in range(len(day_number_cont)):
    if len(thirty_days_yield)<=30:
      thirty_days_yield.append(y_thirty_days_yield[i]/y_thirty_days_yield[0])
    else:
      thirty_days_yield.append(y_thirty_days_yield[i]/y_thirty_days_yield[i-30])
  delta = pd.to_datetime(dates[-1]) -pd.to_datetime(dates[0]) 
  dates_thirty_days=[]
  for i in range(delta.days + 1):
    day = pd.to_datetime(dates[0]) + timedelta(days=i)
  dates_thirty_days.append(day)
  p_dates_num_thirty_days,plt_title_thirty_days=plot_polynomial_3_regression(dates_thirty_days,thirty_days_yield,day_number_cont,yield_mode,pred_type)
  return np.mean(thirty_days_yield), y_fitted, plt_title, p_dates_num_thirty_days, plt_title_thirty_days

def plot_exponential_regression(dates,yields,dates_num):
  a,b= np.polyfit(dates_num,np.log(yields),1)
  ln_fit=linregress(dates_num,np.log(yields))
  factor=np.exp(b)
  exponent=np.exp(a)
  y_yield=factor*(exponent**dates_num)
  plt_title='exponential regression using old function'
  thirty_days_yield=[]
  for i in range(len(dates)):
    if len(thirty_days_yield)<=30:
      thirty_days_yield.append(y_yield[i]/y_yield[0])
    else:
      thirty_days_yield.append(y_yield[i]/y_yield[i-30])
  p_dates_num_thirty_days,plt_title_thirty_days=plot_polynomial_3_regression(dates,thirty_days_yield,dates_num,'forward','swing')
  return np.exp(b), a, (ln_fit.rvalue)**2, y_yield, plt_title, p_dates_num_thirty_days, plt_title_thirty_days

def calculate_yearly_average_monthly_yields(dates, dohlcav_mpnxp_data, sig_and_order_df, yield_mode, pred_type):

  dates = pd.to_datetime(dates)
  dohlcav_mpnxp_data['DCP_date_current_period'] = pd.to_datetime(dohlcav_mpnxp_data['DCP_date_current_period'])
  
  yearly_yield_data =[]
  for year in sorted(set(dates.year)):
      print(year)
      sig_and_order_df['dates'] = pd.to_datetime(sig_and_order_df['dates'])
      yearly_data = dohlcav_mpnxp_data[dohlcav_mpnxp_data['DCP_date_current_period'].dt.year == year]
      sig_and_order_year = sig_and_order_df[sig_and_order_df['dates'].dt.year == year]
      start_date = yearly_data['DCP_date_current_period'].min()
      end_date = yearly_data['DCP_date_current_period'].max()
      previous_trading_order='H'
      yearly_yield_initial = 1
      yearly_yields = []
      counter = 0

      if not sig_and_order_year.empty:
        first_prediction_date = sig_and_order_year['dates'].min()

        first_prediction_date = pd.to_datetime(first_prediction_date)

        yearly_data_filtered = yearly_data[yearly_data['DCP_date_current_period'] >= first_prediction_date]
        sig_and_order_year_filtered = sig_and_order_year[sig_and_order_year['dates'] >= first_prediction_date]

        yield_date_loop, day_number_loop = retreive_date_interval(yearly_data_filtered, first_prediction_date, end_date)
      sig_and_order_df['dates'] = pd.to_datetime(sig_and_order_df['dates']).dt.strftime("%Y-%m-%d")

      OPCP_index=yearly_data.index[yearly_data['DCP_date_current_period']==start_date].values[0]
      buying_price=selling_price=yearly_data.loc[OPCP_index,'OPCP_open_price_current_period']
      idx_current_date=None
      for date in yield_date_loop:
          date = pd.to_datetime(date).strftime("%Y-%m-%d")
          if not sig_and_order_year[sig_and_order_year['dates'] == date].empty:
            idx_current_date = sig_and_order_year.index[sig_and_order_year['dates'] == date].values[0]
          elif idx_current_date is not None:
            idx_current_date = idx_current_date
          else:
            continue
          if counter != 0:
            current_trading_order = sig_and_order_year.loc[idx_current_date - 1, 'orders']
          else:
            current_trading_order = 'H'
          check_list=['H','K',previous_trading_order]

          if current_trading_order not in check_list:
            if current_trading_order == 'B':
                if not yearly_data[yearly_data['DCP_date_current_period'] == date].empty:
                    buying_price = yearly_data[yearly_data['DCP_date_current_period'] == date]['OPCP_open_price_current_period'].values[0]
                previous_trading_order=current_trading_order
                check_list = ['H', 'K', previous_trading_order]

            elif current_trading_order == 'S':
                if not yearly_data[yearly_data['DCP_date_current_period'] == date].empty:
                    selling_price = yearly_data[yearly_data['DCP_date_current_period'] == date]['OPCP_open_price_current_period'].values[0]
                previous_trading_order=current_trading_order
                check_list = ['H', 'K', previous_trading_order]
                
            yearly_yield_initial = yearly_yield_initial * selling_price / buying_price
            yearly_yields.append(yearly_yield_initial)

          else:
              yearly_yields.append(yearly_yield_initial)
          counter = counter+1

      average_yearly_yield, _, _, _, _ = curve_fit_exponential_regression(yield_date_loop, yearly_yields, day_number_loop, yield_mode, pred_type)
      yearly_yield_data.append({
        'Year': year,
        'Average_Monthly_Yield': average_yearly_yield
      })
  yearly_yields_df = pd.DataFrame(yearly_yield_data)
  return yearly_yields_df

def TYC_calculate_yields(dohlcav_mpnxp_data, train_end_date, dates, signals, orders, yield_mode, pred_type):
  try:
    '''Reconstruct DataFrame dohlcav_mpnxp_data object'''
    dohlcav_mpnxp_data = pd.DataFrame(json.loads(dohlcav_mpnxp_data))
    dohlcav_mpnxp_data['DNCP_day_number_current_period'] = dohlcav_mpnxp_data['DNCP_day_number_current_period'].astype(float)
    dohlcav_mpnxp_data['OPCP_open_price_current_period'] = dohlcav_mpnxp_data['OPCP_open_price_current_period'].astype(float)
    sig_and_order_df=pd.DataFrame()
    sig_and_order_df['dates']=dates
    sig_and_order_df['signals']=signals
    sig_and_order_df['orders']=orders
    if yield_mode=='training':
      start_date=sig_and_order_df['dates'][0]
      end_date=train_end_date
    else:
      validation_start_index = dohlcav_mpnxp_data.index[dohlcav_mpnxp_data['DCP_date_current_period']==train_end_date].values[0]
      validation_end_index=dohlcav_mpnxp_data.tail(1).index.item()
      print('validation start index : ',validation_start_index)
      print(type(validation_start_index))
      print('validation end index : ',validation_end_index)
      print(type(validation_end_index))
      start_date=dohlcav_mpnxp_data['DCP_date_current_period'].iloc[int(validation_start_index)+1]
      end_date=dohlcav_mpnxp_data['DCP_date_current_period'].iloc[int(validation_end_index)]
    yield_initial=1
    yields=[]
    thirty_days_yields=[]
    trend_yields=[]
    yield_trend_offsets=[]
    yield_trend_offsets_moving_averages=[]
    in_out_orders=[]
    initial_period=start_date
    OPCP_index=dohlcav_mpnxp_data.index[dohlcav_mpnxp_data['DCP_date_current_period']==initial_period].values[0]
    buying_price=selling_price=dohlcav_mpnxp_data.loc[OPCP_index,'OPCP_open_price_current_period']
    previous_trading_order='H'
    if yield_mode=='training':
      yield_date_loop,day_number_loop=retreive_date_interval(dohlcav_mpnxp_data,sig_and_order_df['dates'][1],end_date)
    else:
      yield_date_loop,day_number_loop=retreive_date_interval(dohlcav_mpnxp_data,start_date,end_date)

    print(sig_and_order_df['dates'])
    for date in yield_date_loop:
      date=pd.to_datetime(date)
      date=date.strftime("%Y-%m-%d")

      matching_indices = sig_and_order_df.index[sig_and_order_df['dates'] == date].values
      print('len_matching_indices', len(matching_indices))
      if len(matching_indices) > 0:
          idx_current_date = matching_indices[0]
          if idx_current_date - 1 >= 0:
              current_trading_order = sig_and_order_df.loc[idx_current_date - 1, 'orders']  # trading_previous
          else:
              current_trading_order = None  
          check_list = ['H', 'K', previous_trading_order]
      else:
          print(f"Date {date} not found in 'dates' column.")
          current_trading_order = None
          check_list = ['H', 'K', previous_trading_order]

      if current_trading_order not in check_list:
        if current_trading_order=='B':
          buying_price=dohlcav_mpnxp_data[dohlcav_mpnxp_data['DCP_date_current_period']==date]['OPCP_open_price_current_period'].values[0]
          previous_trading_order=current_trading_order
          check_list.pop()
          check_list.append(previous_trading_order)
        if current_trading_order=='S':
          selling_price=dohlcav_mpnxp_data[dohlcav_mpnxp_data['DCP_date_current_period']==date]['OPCP_open_price_current_period'].values[0]
          previous_trading_order=current_trading_order
          check_list.pop()
          check_list.append(previous_trading_order)
        yield_initial=yield_initial*selling_price/buying_price#replace trading_yield
        print(yield_initial)
        yields.append(yield_initial)
      else:
        print(yield_initial)
        yields.append(yield_initial)

    a_guess,b_guess,ln_fit_rvalue_sqr,y_yield,plt_title,p_dates_num_thirty_days,plt_title_thirty_days=plot_exponential_regression(yield_date_loop,yields,day_number_loop)
    average_yield,y_fitted_curve_fit,plt_title_curve_fit,p_dates_num_thirty_days_curve_fit,plt_title_thirty_days_curve_fit=curve_fit_exponential_regression(yield_date_loop,yields,day_number_loop,yield_mode,pred_type)
    yearly_yields_df = calculate_yearly_average_monthly_yields(dates, dohlcav_mpnxp_data, sig_and_order_df, yield_mode, pred_type)
    overall_average_monthly_yield = yearly_yields_df['Average_Monthly_Yield'].mean()
    print(overall_average_monthly_yield)
    payload = {
      "success": True,
      "a_guess": a_guess,
      "b_guess": b_guess,
      "ln_fit_rvalue_sqr": ln_fit_rvalue_sqr,
      "y_yield": y_yield,
      "plt_title": plt_title,
      "p_dates_num_thirty_days": p_dates_num_thirty_days,
      "plt_title_thirty_days": plt_title_thirty_days,
      "average_yield": average_yield,
      "y_fitted_curve_fit": y_fitted_curve_fit,
      "plt_title_curve_fit": plt_title_curve_fit,
      "p_dates_num_thirty_days_curve_fit": p_dates_num_thirty_days_curve_fit,
      "plt_title_thirty_days_curve_fit": plt_title_thirty_days_curve_fit,
      "yearly_average_monthly_yield": yearly_yields_df.to_dict(orient='records'),
      "overall_average_monthly_yield": overall_average_monthly_yield
    }
  except Exception as errors:
    print(errors,traceback.format_exc())
    payload = {
      "success": False,
      "errors": traceback.format_exc()
    }
  return payload

def signals_and_recommendations_day(swing_targets, cpcp_cols, buy_threshold, sell_threshold):
  try:
    current_financial_asset_swing_corrected_targets=swing_targets
    close_price_current_period=cpcp_cols
    predicted_main_target_close_price_current_period_ratio = np.divide(current_financial_asset_swing_corrected_targets, close_price_current_period)
    predicted_main_target_close_price_current_period_ratio = np.array(predicted_main_target_close_price_current_period_ratio).flatten()
    buy_threshold_percentage =buy_threshold
    sell_threshold_percentage =sell_threshold
    buy_threshold = 1 + buy_threshold_percentage / 100
    sell_threshold = 1 + sell_threshold_percentage / 100
    current_financial_asset_signal = []
    for value in predicted_main_target_close_price_current_period_ratio:
        if value > buy_threshold:        
            current_financial_asset_signal.append(value / buy_threshold) 
        elif value < sell_threshold:
            current_financial_asset_signal.append(- sell_threshold / value ) 
        else:
            current_financial_asset_signal.append((2 * (value - sell_threshold ) / ( buy_threshold - sell_threshold ))-1) 
    current_financial_asset_recommendation = []
    for value in current_financial_asset_signal:
        if value > 0.01:
            current_financial_asset_recommendation.append('B')
        elif value < -0.01:
            current_financial_asset_recommendation.append('S')
        else:
            current_financial_asset_recommendation.append('H')

    payload = {
      "success": True,
      "current_financial_asset_signal": current_financial_asset_signal,
      "current_financial_asset_recommendation": current_financial_asset_recommendation
    }
  except Exception as errors:
    payload = {
      "success": False,
      "errors": str(errors)
    }
  return payload

def TYC_calculate_signals_orders_yields(swing_targets, 
                                        cpcp_cols,
                                        dohlcav_mpnxp_data,
                                        train_end_date,
                                        dates,
                                        buy_threshold = 1,
                                        sell_threshold = -1,
                                        yield_mode = 'training',
                                        pred_type = 'swing',
                                        ):
    dohlcav_mpnxp_data = dohlcav_mpnxp_data.to_json()
    
    calculate_signals_response = signals_and_recommendations_day(swing_targets,cpcp_cols,buy_threshold,sell_threshold)
    print(calculate_signals_response)
    signals = calculate_signals_response['current_financial_asset_signal']
    orders = calculate_signals_response['current_financial_asset_recommendation']
    calculate_yields_response = TYC_calculate_yields(dohlcav_mpnxp_data,train_end_date,dates,signals,orders,yield_mode,pred_type)
    print(calculate_yields_response)
    average_yield = calculate_yields_response['average_yield']
    return signals,orders,average_yield

def generate_signals_and_orders(dates, predictions):
    try:
        # Ensure predictions has the correct structure
        predictions = np.array(predictions)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 5)
        elif predictions.shape[1] != 5:
            raise ValueError(f"Expected 5 columns in predictions, but got {predictions.shape[1]}")

        # Create DataFrame
        predictions_df = pd.DataFrame(predictions, columns=['dates', 'Close', 'High', 'Low', 'Open'])
        predictions_df['dates'] = pd.to_datetime(predictions_df['dates'])

        # Ensure dates match
        dates = pd.to_datetime(dates)
        if len(dates) != len(predictions_df):
            raise ValueError(f"Dates and predictions length mismatch: {len(dates)} vs {len(predictions_df)}")

        # Generate signals and orders
        signals = []
        orders = []
        for i, date in enumerate(dates):
            if i < len(predictions_df):
                tomorrow_prediction = predictions_df.iloc[i]  # Use corresponding predictions
                predicted_close = tomorrow_prediction['Close']
                predicted_open = tomorrow_prediction['Open']

                # Example decision logic
                if predicted_close > predicted_open:
                    signals.append('Buy')
                    orders.append('B')
                else:
                    signals.append('Sell')
                    orders.append('S')
            else:
                # Handle missing predictions gracefully
                signals.append('Hold')
                orders.append('H')

        return signals, orders

    except Exception as e:
        print(f"Error in generate_signals_and_orders: {traceback.format_exc()}")
        return [], []


def calculate_intraday_yields(dohlcav_mpnxp_data, data, dates, predictions):
    """
    Calculate daily yields based on next-day predictions and also compute average monthly 
    yield and yearly average of monthly yields.
    """
    try:
        # Parse input DataFrame and prediction data
        signals, orders = generate_signals_and_orders(data, predictions)
        dohlcav_mpnxp_data = dohlcav_mpnxp_data
        dohlcav_mpnxp_data['DCP_date_current_period'] = pd.to_datetime(dohlcav_mpnxp_data['DCP_date_current_period'])
        dohlcav_mpnxp_data['OPCP_open_price_current_period'] = dohlcav_mpnxp_data['OPCP_open_price_current_period'].astype(float)
        
        predictions_df = pd.DataFrame(data = predictions, columns=['dates', 'Open', 'High', 'Low', 'Close'])
        predictions_df['dates'] = pd.to_datetime(predictions_df['dates'])
        
        sig_and_order_df = pd.DataFrame({'dates': pd.to_datetime(dates), 'signals': signals, 'orders': orders})
        
        # Check for empty inputs
        if dohlcav_mpnxp_data.empty or predictions_df.empty or sig_and_order_df.empty:
            raise ValueError("One or more input DataFrames are empty.")
        
        # Initialization
        yield_initial = 1.0
        yields = []
        previous_order = 'H'
        buying_price = None
        
        for i, row in sig_and_order_df.iterrows():
            date = row['dates']
            current_order = row['orders']
            
            # Ensure predictions exist for the next day
            next_day_prediction = predictions_df[predictions_df['dates'] == date]
            if next_day_prediction.empty:
                print(f"Warning: No predictions available for date: {date}. Yield remains unchanged.")
                yields.append(yield_initial)
                continue
            
            predicted_prices = next_day_prediction.iloc[0]
            open_price = predicted_prices['Open']
            close_price = predicted_prices['Close']
            
            # Decision logic based on orders
            if current_order == 'B' and previous_order != 'B':
                # Buy at next day's Open price
                buying_price = open_price
                previous_order = 'B'
            elif current_order == 'S' and previous_order == 'B':
                # Sell at next day's Close price
                if buying_price is not None:
                    yield_initial *= close_price / buying_price
                buying_price = None  # Reset buying price
                previous_order = 'S'
            
            # Append yield for the current day
            yields.append(yield_initial)
        
        # Create a DataFrame with daily yields
        yield_df = pd.DataFrame({'Date': sig_and_order_df['dates'], 'Yield': yields})
        yield_df['Month'] = yield_df['Date'].dt.to_period('M')  # Extract year-month
        yield_df['Year'] = yield_df['Date'].dt.year            # Extract year
        
        # Calculate average monthly yields
        monthly_yield = yield_df.groupby('Month')['Yield'].last().pct_change().mean()
        
        # Calculate yearly average of monthly yields
        yearly_monthly_yield = yield_df.groupby('Year')['Yield'].last().pct_change().mean()
        
        return {
            'daily_yields': yield_df,
            'average_monthly_yield': monthly_yield,
            'yearly_average_monthly_yield': yearly_monthly_yield
        }
    
    except Exception as e:
        print(f"Error in calculate_intraday_yields: {e}")
        print(traceback.format_exc())
        return {
            'daily_yields': pd.DataFrame(),
            'average_monthly_yield': None,
            'yearly_average_monthly_yield': None
        }