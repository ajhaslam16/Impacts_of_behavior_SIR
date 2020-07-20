#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For exploring bifurcations in the parameters.

Created on Wed Jul 15 19:25:13 2020

"""

import numpy as np


'''
Given the solution to the model over a time period and the time discritization info. 

Returns an estimation of the total size of the epidemic by the end of it.

'''
def final_epi_size(state_sim, time):
    
    Sn_fin = state_sim[-1,0]
    Sa_fin = state_sim[-1,1]
    
    fin_S = Sn_fin + Sa_fin
    
    return 1 - fin_S




'''
Given the solution to the model over a time period and the time discritization info.
Optional argument to specify which data you want the peak in...

    (I_S, M, I_tot, I+E)

optional argument to specific which data you are giving (ie. SIRan, SEIRan, or
SEIRDan, etc..). Default is SIR

    (SIR, SEIR)


Returns both which day is the peak and what the peak is.
'''
def daily_peak(state_sim, time, peak_type='I_S', data_type='SIR'):
    
    code=2
    if data_type == 'SIR':
        code = 2
    elif data_type == 'SEIR':
        code = 4
    
    if peak_type == 'I_S':
        peak_dat = state_sim[:,code]
    elif peak_type == 'M':
        peak_dat = state_sim[:,-1]
    else:
        peak_dat = state_sim[:,code]+state_sim[:,code+1]+state_sim[:,code+2]
        
    if data_type == 'SEIR' and peak_type == 'I+E':
        peak_dat += state_sim[:,2]+state_sim[:,3]
        
    
    peak_amount = max(peak_dat)
    day_peak = time[np.argmax(peak_dat)]
    
    return day_peak, peak_amount

