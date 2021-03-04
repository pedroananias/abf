#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################################
# ### Miscellaneous Functions
# ### Module responsible for storing extra data processing functions, accuracy measures and others.
##################################################################################################################

# Módulos dependentes
import gc
import scipy
import warnings
import math
import pandas as pd
import numpy as np

# Machine Learning
from sklearn import metrics

# Disable warnings
warnings.filterwarnings('ignore') 


# Statistics methods
def concordance_measures(cm, y_true, y_pred):

  # initial attributes
  y_true = [] if len(y_true) == 0 else y_true
  y_pred = [] if len(y_pred) == 0 else y_pred
  total = float(len(y_true))
  m = 1.0 if len(cm) == 0 else len(cm[0])
  marg = 0.0

  # measures
  acc           = metrics.accuracy_score(y_true, y_pred)
  bacc          = metrics.balanced_accuracy_score(y_true, y_pred)
  f1score       = metrics.f1_score(y_true, y_pred, average='weighted')
  kappa         = metrics.cohen_kappa_score(y_true, y_pred)
  mcc           = metrics.matthews_corrcoef(y_true, y_pred)
  try:
    rmse = math.sqrt(metrics.mean_squared_error(y_true, y_pred))
  except:
    rmse = 0
  try:
    mae = metrics.mean_absolute_error(y_true, y_pred)
  except:
    mae = 0
  tau, p_value  = scipy.stats.kendalltau(y_true, y_pred)
  
  # marg
  vtau = 0.0
  vkappa = 0.0
  if m > 1:
    for i in range(0,m):
      marg += sum(cm[i,])*sum(cm[:,i])

    # Others
    vtau = (1/total)*((acc*(1-acc))/((1-(1/m))**2)) if (1 - (1/m)) > 0 else 0
    t1 = acc
    t2 = marg/((total)**2)
    t3 = 0.0
    t4 = 0.0
    for i in range(0,m):
      t3 += cm[i,i]*(sum(cm[i,])+sum(cm[:,i]))
    for i in range(0,m):
      for j in range(0,m):
        t4 += cm[i,j]*((sum(cm[j,])+sum(cm[:,i]))**2)
    t3 = t3/((total)**2)
    t4 = t4/((total)**3)

    # Kappa Variance
    vkappa = (1/total)*( ((t1*(1-t1))/((1-t2)**2)) + ((2*(1-t1)*(2*t1*t2-t3))/((1-t2)**3)) + ((((1-t1)**2)*(t4-4*(t2**2)))/((1-t2)**4)) )

  # TP, FP, TN, FN
  try:
    tn, fp, fn, tp = cm.ravel()
  except:
    tn, fp, fn, tp = [0,0,0,0]

  # fix values
  acc           = float(acc if not math.isnan(acc) else 0.0)
  bacc          = float(bacc if not math.isnan(bacc) else 0.0)
  f1score       = float(f1score if not math.isnan(f1score) else 0.0)
  kappa, vkappa = float(kappa if not math.isnan(kappa) else 0.0), float(vkappa if not math.isnan(vkappa) else 0.0)
  mcc           = float(mcc if not math.isnan(mcc) else 0.0)
  rmse          = float(rmse if not math.isnan(rmse) else 0.0)
  mae           = float(mae if not math.isnan(mae) else 0.0)
  tau, vtau     = float(tau if not math.isnan(tau) else 0.0), float(vtau if not math.isnan(vtau) else 0.0)
  p_value       = float(p_value if not math.isnan(p_value) else 0.0)

  # String
  string = 'Acc:'+str(round(acc,4))+', BAcc:'+str(round(bacc,4))+', F1-Score:'+str(round(f1score,4))+', Kappa:'+str(round(kappa,4))+', vKappa:'+str(round(vkappa,4))+', Tau:'+str(round(tau,4))+', vTau:'+str(round(vtau,4))+', p-value:'+str(round(p_value,4))+', Mcc:'+str(round(mcc,4))+', RMSE:'+str(round(rmse,4))+', MAE:'+str(round(mae,4))+', TP:'+str(tp)+', FP:'+str(fp)+', TN:'+str(tn)+', FN:'+str(fn)

  # Response
  return {'total':total,'acc':acc,'bacc':bacc,'f1score':f1score,'tau':tau,'vtau':vtau,'p_value':p_value,'kappa':kappa,'vkappa':vkappa,'mcc':mcc,'rmse':rmse,'mae':mae,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'string':string}


# Remove duplicated dates
def remove_duplicated_dates(dates: list):
  visited = []
  for i,date in enumerate(dates):
    if date.strftime("%Y-%m-%d") in visited:
      del dates[i]
    else:
      visited.append(date.strftime("%Y-%m-%d"))
  return dates


# Frame a time series as a supervised learning dataset.
def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):

  # configuration
  n_vars          = 1 if type(df) is list else df.shape[1]
  df              = pd.DataFrame(df)
  cols, names     = list(), list()
  
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
      
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
      
  # put it all together
  df = pd.DataFrame(np.concatenate(cols, axis=1),columns=names)

  # clear memory
  del cols, names
  gc.collect()

  # drop rows with NaN values
  if dropnan:
    df.dropna(inplace=True)
      
  # return value
  return df