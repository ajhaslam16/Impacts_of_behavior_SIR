#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poletti.py

This Python file contains the functions which compute the derivative of the 
state variables at time t and current state y0. 

These functions can be imported and used with odeint to solve the system of
differential equations associated with Poletti's model and the varients of it
that we are exploring.


Note that certain parameters are required. These are assigned globally in this
code. However there is also a function (set_params) that allows you to 
externally set any combination of the parameters. 


Created on Fri Jul 17 12:07:27 2020

@author: ajhaslam
"""

import numpy as np



## Global Parameters##

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



params = [gamma,beta_S, beta_A, q, p, nu, m, rho, mu]


'''
Set the parameters to be different.
'''
def get_params(gamma0=gamma, beta_S0=beta_S, beta_A0=beta_A,
              q0=q,p0=p, nu0=nu, m0=m, rho0=rho, mu0=mu):
    
    params = [gamma,beta_S, beta_A, q, p, nu, m, rho, mu]
    
    return params



'''
Classic SIR model

params is an optional argument which should be a list of 8 parameter values.
By default it is the hardwired global parameters defined in this file. 

Note that if you want to include this additional argument when calling odeint,
you must write:
    
    odeint(SIR_system, State_vector, t, args=(params,))
    
    * Note the comma.

'''
def SIR_system(State_vector, t, params=params):
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu = params
    
    S, I, R = State_vector
    
    S_dot = -beta_S*S*I
    I_dot = beta_S*S*I - gamma*I
    R_dot = gamma*I
    
    deriv = np.array([S_dot, I_dot, R_dot])
    
    return deriv


'''
SEIR COVID-19 model without behavior dynamics.
'''


'''
Helper function for Poletti model

Gives the force of infection (or lambda) given the current number of infected
individuals in each respective infected compartments.
'''
def force_of_infection(I_S, I_An, I_Aa):

    return beta_S*I_S+beta_A*I_An+q*beta_A*I_Aa


'''
Helper function for Poletti model


Gives P_n-P_a as guage of which behavior is more advantageous as a function 
of percieved number of cases (ie. perceived risk).
'''
def payoff_difference(M):
    
#    P_n = -m_n*M
#    P_a = -k - m_a*M
#    Delta_P = P_n-P_a # k - (m_n-m_a)*M  # most intuitive
    
    Delta_P  = 1-m*M  # k*(1-m*M) = the real Delta_P  # note: divided by extra k that will be fixed with rho
    
    return Delta_P



'''
SIR  with behavior dynamics (Poletti -- compartment version)

This is the version of the Poletti model from equation (3) on page 83 of the
2012 paper. This model is different from the model that Poletti et. al. use
in their analysis because it does not assume that x:(1-x) gives the same ratio
for S, I_A, and R. 

params is an optional argument which should be a list of 8 parameter values.
By default it is the hardwired global parameters defined in this file. 

Note that if you want to include this additional argument when calling odeint,
you must write:
    
    odeint(SIR_system, State_vector, t, args=(params,))
    
    * Note the comma.

'''
def ODE_system(State_vector,t, params=params):
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu = params
    
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
    Sa_dot += -rho*(S_n*S_a*Delta_P + mu*S_a - mu*S_n)
    
    IAn_dot += rho*(I_An*I_Aa*Delta_P + mu*I_Aa - mu*I_An)
    IAa_dot += -rho*(I_An*I_Aa*Delta_P + mu*I_Aa - mu*I_An)
    
    RAn_dot += rho*(R_An*R_Aa*Delta_P + mu*R_Aa - mu*R_An)
    RAa_dot += -rho*(R_An*R_Aa*Delta_P + mu*R_Aa - mu*R_An)
    
    
     ## inter-compartment imitation:
    
    # if Delta_P >0:
    
    Sn_dot +=  rho*S_a*(I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    Sa_dot += -rho*S_a*(I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    
    IAn_dot +=  rho*I_Aa*(S_n+R_An)*Delta_P*np.heaviside(Delta_P,0)
    IAa_dot += -rho*I_Aa*(S_n+R_An)*Delta_P*np.heaviside(Delta_P,0)
    
    RAn_dot +=  rho*R_Aa*(I_An+S_n)*Delta_P*np.heaviside(Delta_P,0)
    RAa_dot += -rho*R_Aa*(I_An+S_n)*Delta_P*np.heaviside(Delta_P,0)
    
    # if Delta_P <0
    
    Sn_dot +=  rho*S_n*(I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    Sa_dot += -rho*S_n*(I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    
    IAn_dot +=  rho*I_An*(S_a+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    IAa_dot += -rho*I_An*(S_a+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    
    RAn_dot +=  rho*R_An*(I_Aa+S_a)*Delta_P*np.heaviside(-Delta_P,0)
    RAa_dot += -rho*R_An*(I_Aa+S_a)*Delta_P*np.heaviside(-Delta_P,0)


    deriv = np.array([Sn_dot,Sa_dot,IS_dot, IAn_dot, IAa_dot,\
                      RS_dot, RAn_dot, RAa_dot, M_dot])
    
    return deriv