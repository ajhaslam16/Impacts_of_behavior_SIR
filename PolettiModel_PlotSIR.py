#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Poletti Model
SIR with social behavioral changes
adapted to plot all three compartment types

AIM-MCRN Dynamics and Data 

Created on Tue Jul 14 12:42:31 2020

@author: Alanna J. Haslam
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


## Global parameters ##

num_compartments = 8 # the number of compartments in the epi part of the model

recovery_days = 2.8 # average length of infection (current baseline from 2012)
gamma = 1/recovery_days # rate of recovery

beta_S = 0.5 # infection rate for symptomatic infectives
beta_A = 0.5 # infection rate for asymptomatic infectives

q = 0.85  # reduction factor for those with altered/adjusted behavior
p = 0.6   # probability of developing symptoms


avg_memory = 10   # average length of memory regarding active cases
nu  = 1/avg_memory  # rate of forgetting past "new case" counts

#m_n = # percieved risk of infection without adjusting behavior
#m_a = # percieved risk of infection while using adjusted behavior
#k =  # constant cost of having adjusted behavior
M_thresh = 0.01 # risk threshold ... k/(m_n-m_a)
m = 1/M_thresh

#omega_tilde = # rate at which individuals meet
#phi =         # proportionality constant for how risk perception linearly relates to switching
#omega = omega_tilde*phi
#alpha =       # timescale factor ... t (slow) / tau(fast)
rho = 10 #k*omega/alpha   # speed of behavioral changes (1/days)

#mu_tilde = # irrational exploration (randomness)
mu = 10**(-8)  #mu_tilde/(omega*k) # irrational exploration with fixed units


'''
This is the version of the Poletti model from equation (3) on page 83 of the
2012 paper. This model is different from the model that Poletti et. al. use
in their analysis because it does not assume that x:(1-x) gives the same ratio
for S, I_A, and R. 
'''
def ODE_system(State_vector,t):
    
#    epi_compartments = State_vector[:num_compartments]
#    behavior_variables = State_vector[num_compartments:]
    
    epi_compartments = State_vector[:-1]
    behavior_variables = State_vector[-1]
    
    S_n, S_a, I_S, I_An, I_Aa, R_S, R_An, R_Aa = epi_compartments
    M = behavior_variables
    
    lambda_t = force_of_infection(I_S, I_An, I_Aa)
    
    ### first establish the transmission dynamics ###
    Sn_dot = -lambda_t*S_n
    Sa_dot = - q*lambda_t*S_a
    
    IS_dot  = p*(lambda_t*S_n + q*lambda_t*S_a) - gamma*I_S
    IAn_dot = (1-p)*lambda_t*S_n - gamma*I_An
    IAa_dot = (1-p)*q*lambda_t*S_a - gamma*I_Aa
    
    RS_dot  = gamma*I_S
    RAn_dot = gamma*I_An
    RAa_dot = gamma*I_Aa
    
    ### second add the imitation and behavior switching dynamics ###
    M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu*I_S
    
    Delta_P = payoff_difference(M) #note: this still has extra k divided 
    
    ## intra-compartment imitation:
    
    Sn_dot += rho*(S_n*S_a*Delta_P + mu*S_a - mu*S_n)
    Sa_dot += rho*(S_n*S_a*Delta_P - mu*S_a + mu*S_n)
    
    IAn_dot += rho*(I_An*I_Aa*Delta_P + mu*I_Aa - mu*I_An)
    IAa_dot += rho*(I_An*I_Aa*Delta_P - mu*I_Aa + mu*I_An)
    
    RAn_dot += rho*(R_An*R_Aa*Delta_P + mu*R_Aa - mu*R_An)
    RAa_dot += rho*(R_An*R_Aa*Delta_P - mu*R_Aa + mu*R_An)
    
    
    ## inter-compartment imitation:
    
    if Delta_P > 0: # note that sign should still be the same since k>0
        
        Sn_dot +=  rho*S_a*(I_An+R_An)*Delta_P
        Sa_dot += -rho*S_a*(I_An+R_An)*Delta_P
        
        IAn_dot +=  rho*I_Aa*(S_n+R_An)*Delta_P
        IAa_dot += -rho*I_Aa*(S_n+R_An)*Delta_P
        
        RAn_dot +=  rho*R_Aa*(I_An+S_n)*Delta_P
        RAa_dot += -rho*R_Aa*(I_An+S_n)*Delta_P
        
    elif Delta_P <0: # is the sign correct here? Should it be abs[Delta_P]?
        
        Sn_dot += -rho*S_n*(I_Aa+R_Aa)*Delta_P
        Sa_dot +=  rho*S_n*(I_Aa+R_Aa)*Delta_P
        
        IAn_dot += -rho*I_An*(S_a+R_Aa)*Delta_P
        IAa_dot +=  rho*I_An*(S_a+R_Aa)*Delta_P
        
        RAn_dot += -rho*R_An*(I_Aa+S_a)*Delta_P
        RAa_dot +=  rho*R_An*(I_Aa+S_a)*Delta_P
        
    deriv = np.array([Sn_dot,Sa_dot,IS_dot, IAn_dot, IAa_dot,\
                      RS_dot, RAn_dot, RAa_dot, M_dot])
#    deriv = np.zeros(9)
    
    return deriv


'''
Gives the force of infection (or lambda) given the current number of infected
individuals in each respective infected compartments.
'''
def force_of_infection(I_S, I_An, I_Aa):

    return beta_S*I_S+beta_A*I_An+q*beta_A*I_Aa


'''
Gives P_n-P_a as guage of which behavior is more advantageous as a function 
of percieved number of cases (ie. perceived risk).
'''
def payoff_difference(M):
    
#    P_n = -m_n*M
#    P_a = -k - m_a*M
#    Delta_P = P_n-P_a   # most intuitive
    
    Delta_P  = 1-m*M    # note: divided by extra k that will be fixed with rho
    
    return Delta_P

def plot_SIR(time, solution):
    
    fig, ax = plt.subplots()  

    S = solution[:,0]+solution[:,1]
    I = solution[:,2]+solution[:,3]+solution[:,4]
    R = solution[:,5]+solution[:,6]+solution[:,7]

    ax.plot(time, S, '-b')
    ax.plot(time, I, '-r')
    ax.plot(time, R, '-g')
    
    ax.set(xlabel='time (days)', ylabel='Level of Symptomatic Infective Individuals')
    ax.grid()
    
    return

def main():
    
    Sn_0  = 0.99
    Sa_0  = 0.005
    IS_0  = 0.001
    IAn_0 = 0.004
    IAa_0 = 0
    RS_0  = 0
    RAn_0 = 0
    RAa_0 = 0
    
    M_0 = 0.0001  # prior belief of risk (or "overestimation" of risk)
    
    initial_state = np.array([Sn_0,Sa_0,IS_0,IAn_0,IAa_0,RS_0,RAn_0,RAa_0,M_0])
    
    if np.sum(initial_state)- M_0 != 1:
        print("Error: make sure population sums to 1")
        return
    
    t = np.arange(0,10,0.001)
    
    solution = odeint(ODE_system,initial_state,t)
    
    plot_SIR(t, solution)
    
    
    return

main()

















    

