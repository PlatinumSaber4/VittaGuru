import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
from datetime import datetime
import tempfile

def sir_parameters(x,y):
  x=np.array(x)
  y=np.array(y)
  analytical_params = linregress(x, y)

  slope = analytical_params.slope
  intercept = analytical_params.intercept
  rvalue = analytical_params.rvalue

  y_trend_line = slope*x + intercept
  avg_trend_line_distance = np.mean(np.abs(y_trend_line - y)/y_trend_line)

  return slope, intercept, rvalue**2, avg_trend_line_distance

def comparison_plot(predicted_targets,actual_targets,dates,target, stock_name):
  print(type(predicted_targets))
  trend_slope,trend_intercept,trend_r2,dispersion=sir_parameters(actual_targets,predicted_targets)
  plt.figure(figsize=(20, 16))
  date_format = "%Y-%m-%d"
  date_objects = [datetime.strptime(date, date_format) for date in dates]
  plt.plot(date_objects, actual_targets, label='Actual')
  plt.plot(date_objects, predicted_targets, label='Predicted')
  plt.xlabel('DATE', fontsize=20)
  plt.ylabel(f'{target}', fontsize=20)
  dates = pd.Series(pd.to_datetime(dates))
  start_date = str(dates.min())
  end_date = str(dates.max())
  plt.title(f'Comparison graph - Predicted and Actual Targets versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=20,y=-0.07)
  plt.legend(fontsize=20)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  text_size = 'x-large'
  plt.figtext(.92,.85,['trend_slope:',round(trend_slope,4)], fontsize = text_size)
  plt.figtext(.92,.80,['trend_intercept:',round(trend_intercept,4)], fontsize=text_size)
  plt.figtext(.92,.75,['trend_r2:',round(trend_r2,4)], fontsize=text_size)
  plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
  plt.figtext(.92,.65,['trend_dispersion:',round(dispersion,4)], fontsize = text_size)
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
  plt.savefig(f'prediction_plots/{stock_name}_{target}_comparison_plot.png',bbox_inches='tight')
  result = {
    'trend_slope' : trend_slope,
    'trend_intercept': trend_intercept,
    'trend_r2': trend_r2,
    'dispersion': dispersion
  }
  return result,temp_file.name

def ratio_plot(predicted_targets,actual_targets,dates,target, stock_name):
  predicted_targets = np.array(predicted_targets)
  actual_targets = np.array(actual_targets)
  fig,ax = plt.subplots(1,figsize=(20,16))
  ratio = predicted_targets/actual_targets
  a=np.linspace(0,len(actual_targets),len(actual_targets),dtype=np.int32)
  z=np.polyfit(a,ratio,1)
  p=np.poly1d(z)
  plt.plot(ratio,label='Ratio') 
  plt.plot(a,p(a), alpha=0.75, label='Fitted Line')
  plt.xlabel('DATE',fontsize=20)
  plt.ylabel(f'{target}', fontsize=20)
  dates = pd.Series(pd.to_datetime(dates))
  start_date = str(dates.min())
  end_date = str(dates.max())
  plt.title(f' Ratio plot - Predicted / Actual Targets Ratios versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=20, y=-0.07)
  plt.legend(fontsize=20)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  trend_slope,trend_intercept,trend_r2,dispersion=sir_parameters(actual_targets,ratio)
  text_size = 'x-large'
  plt.figtext(.92,.85,['trend_slope:',round(trend_slope,4)], fontsize = text_size)
  plt.figtext(.92,.80,['trend_intercept:',round(trend_intercept,4)], fontsize=text_size)
  plt.figtext(.92,.75,['trend_r2:',round(trend_r2,4)], fontsize=text_size)
  plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
  plt.figtext(.92,.65,['trend_dispersion:',round(dispersion,4)], fontsize = text_size)
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
  plt.savefig(f'prediction_plots/{stock_name}_{target}_ratio_plot.png',bbox_inches='tight')
  result = {
    'trend_slope' : trend_slope,
    'trend_intercept': trend_intercept,
    'trend_r2': trend_r2,
    'dispersion': dispersion
  }
  return result,temp_file.name

def scatter_plot(predicted_targets,actual_targets,dates,target, stock_name):
  trend_slope,trend_intercept,trend_r2,dispersion=sir_parameters(actual_targets,predicted_targets)
  fig,ax = plt.subplots(1,figsize=(20,16))
  sns.regplot(x=actual_targets,y=predicted_targets,ax=ax)
  ax.set_xlabel(f'Actual Target {target}', fontsize=20)
  ax.set_ylabel(f'Predicted Target {target}', fontsize=20)
  dates = pd.Series(pd.to_datetime(dates))
  start_date = str(dates.min())
  end_date = str(dates.max())
  plt.title(f'Scatter graph - Predicted and Actual Targets versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=15, y=-0.07)
  text_size = 'x-large'
  plt.figtext(.92,.85,['trend_slope:',round(trend_slope,4)], fontsize = text_size)
  plt.figtext(.92,.80,['trend_intercept:',round(trend_intercept,4)], fontsize=text_size)
  plt.figtext(.92,.75,['trend_r2:',round(trend_r2,4)], fontsize=text_size)
  plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
  plt.figtext(.92,.65,['trend_dispersion:',round(dispersion,4)], fontsize = text_size)
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
  plt.savefig(f'prediction_plots/{stock_name}_{target}_scatter_plot.png',bbox_inches='tight')
  result = {
      'trend_slope' : trend_slope,
      'trend_intercept': trend_intercept,
      'trend_r2': trend_r2,
      'dispersion': dispersion
  }
  return result,temp_file.name

def generate_plot(predicted_targets,actual_targets,dates,target, stock_name):
  print("Started")
  predicted_targets = np.array(predicted_targets)
  actual_targets = np.array(actual_targets)
  plot_funcs = {
    'Comparision': comparison_plot(predicted_targets,actual_targets,dates,target, stock_name)[0],
    'Comparisio_plot_address':comparison_plot(predicted_targets,actual_targets,dates,target, stock_name)[1],
    'Ratio': ratio_plot(predicted_targets,actual_targets,dates,target, stock_name)[0],
    'Ratio_Plot_address': ratio_plot(predicted_targets,actual_targets,dates,target, stock_name)[1],
    'Scatter': scatter_plot(predicted_targets,actual_targets,dates,target, stock_name)[0],
    'Scatter_plot_address':scatter_plot(predicted_targets,actual_targets,dates,target, stock_name)[1]
  }
  return plot_funcs