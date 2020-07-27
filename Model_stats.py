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


'''
Only designed for the SIR version where we are looking for peaks in I_S

A peak is considered to be the middle of a four day period in which the first 
two days show an increase in I_S and the second two days show a decrease in I_S.
'''
def peak_data(state_sim, time):
    
    
    IS_data = state_sim[:,2]
    is_inc_data = []
    
    for i in range(len(IS_data)-1):
        is_inc_data.append(int(IS_data[i]<IS_data[i+1]))
        
    is_inc_data = np.array(is_inc_data)
        
    desired_sequence = np.concatenate((np.ones(200),np.zeros(200)))
    peak_days = []
    peak_heights = []
    
    for j in range(len(IS_data)-400):
        
        seq = is_inc_data[j:j+400]
        if seq==desired_sequence:
            peak_days.append(time[j+200])
            peak_heights.append(IS_data[j+200])
    
    return peak_days, peak_heights
