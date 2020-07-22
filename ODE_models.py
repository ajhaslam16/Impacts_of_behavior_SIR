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
"""

import numpy as np



'''
Set the parameters. Default for all is currently the baseline parameters from
Poletti 2012.
'''
def get_params(gamma=(1/2.8), beta_S=0.5, beta_A=0.5,
              q=0.85, p=1, nu=(1/2.8), m=(1/0.01), rho=10,
              mu=(10**(-8)), xi=(1/3), dealthdelt=0.02):
    
    params = [gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, dealthdelt]
    
    return params



'''
Classic SIR model

params is a list of 8 parameter values.
By default it is the hardwired global parameters defined in this file. 

Note that if you want to include this additional argument when calling odeint,
you must write:
    
    odeint(SIR_system, State_vector, t, args=(params,))
    
    * Note the comma.

'''
def SIR_system(State_vector, t, params):
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, dealthdelt = params
    
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
def force_of_infection(I_S, I_An, I_Aa, params):
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, dealthdelt = params

    return beta_S*I_S+beta_A*I_An+q*beta_A*I_Aa


'''
Helper function for Poletti model


Gives P_n-P_a as guage of which behavior is more advantageous as a function 
of percieved number of cases (ie. perceived risk).
'''
def payoff_difference(M, params):
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, dealthdelt = params
    
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
    
    odeint(SIRan_system, State_vector, t, args=(params,))
    
    * Note the comma.

'''
def SIRan_system(State_vector,t, params):
    
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, dealthdelt = params
    
    epi_compartments = State_vector[:-1]
    behavior_variables = State_vector[-1]
    
    S_n, S_a, I_S, I_An, I_Aa, R_S, R_An, R_Aa = epi_compartments
    M = behavior_variables
    
    lambda_t = force_of_infection(I_S, I_An, I_Aa, params)
    
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
    M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu*M
    
    Delta_P = payoff_difference(M, params=params) #note: this still has extra k divided 
    
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


'''
This is another updated version of the Poletti model from equation (3) on page 83 of the
2012 paper, now including an "Exposed" category. Again, this model is different from the model that Poletti et. al. use
in their analysis because it does not assume that x:(1-x) gives the same ratio
for S, I_A, and R. Additionally, we have a compartment for exposed individuals (E), before
they become infectious and (a)symptomatic.
'''
def SEIRan_system(State_vector,t, params):
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, dealthdelt = params
    
    epi_compartments = State_vector[:-1]
    behavior_variables = State_vector[-1]
    
    S_n, S_a, E_n, E_a, I_S, I_An, I_Aa, R_S, R_An, R_Aa = epi_compartments
    M = behavior_variables
    
    lambda_t = force_of_infection(I_S, I_An, I_Aa, params)
    
    ### first establish the transmission dynamics ###
    Sn_dot = -lambda_t*S_n
    Sa_dot = - q*lambda_t*S_a
    
    #additional Exposed compartment
    En_dot = lambda_t*S_n - xi*E_n
    Ea_dot = q*lambda_t*S_a - xi*E_a
    
    IS_dot  = p*xi*(E_n + E_a) - gamma*I_S
    IAn_dot = (1-p)*xi*E_n - gamma*I_An
    IAa_dot = (1-p)*q*xi*E_a - gamma*I_Aa
    
    RS_dot  = gamma*I_S
    RAn_dot = gamma*I_An
    RAa_dot = gamma*I_Aa
    
    ### second add the imitation and behavior switching dynamics ###
    M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu*M
    
    Delta_P = payoff_difference(M, params = params) #note: this still has extra k divided 
    
    ## intra-compartment imitation:
    
    Sn_dot += rho*(S_n*S_a*Delta_P + mu*S_a - mu*S_n)
    Sa_dot += -rho*(S_n*S_a*Delta_P + mu*S_a - mu*S_n)
    
    #exposed dynamics added in
    En_dot += rho*(E_n*E_a*Delta_P + mu*E_a - mu*E_n)
    Ea_dot += -rho*(E_n*E_a*Delta_P + mu*E_a - mu*E_n)
    
    IAn_dot += rho*(I_An*I_Aa*Delta_P + mu*I_Aa - mu*I_An)
    IAa_dot += -rho*(I_An*I_Aa*Delta_P + mu*I_Aa - mu*I_An)
    
    RAn_dot += rho*(R_An*R_Aa*Delta_P + mu*R_Aa - mu*R_An)
    RAa_dot += -rho*(R_An*R_Aa*Delta_P + mu*R_Aa - mu*R_An)
    
    
     ## inter-compartment imitation:
    
    # if Delta_P >0:
    
    Sn_dot +=  rho*S_a*(E_n+I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    Sa_dot += -rho*S_a*(E_n+I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    
    En_dot += rho*E_a*(S_n+I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    Ea_dot += -rho*E_a*(S_n+I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    
    IAn_dot +=  rho*I_Aa*(S_n+E_n+R_An)*Delta_P*np.heaviside(Delta_P,0)
    IAa_dot += -rho*I_Aa*(S_n+E_n+R_An)*Delta_P*np.heaviside(Delta_P,0)
    
    RAn_dot +=  rho*R_Aa*(E_n+I_An+S_n)*Delta_P*np.heaviside(Delta_P,0)
    RAa_dot += -rho*R_Aa*(E_n+I_An+S_n)*Delta_P*np.heaviside(Delta_P,0)
    
    # if Delta_P <0
    
    Sn_dot +=  rho*S_n*(E_a+I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    Sa_dot += -rho*S_n*(E_a+I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    
    En_dot += rho*E_n*(S_a+I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    Ea_dot += -rho*E_n*(S_a+I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    
    IAn_dot +=  rho*I_An*(S_a+E_a+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    IAa_dot += -rho*I_An*(S_a+E_a+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    
    RAn_dot +=  rho*R_An*(I_Aa+E_a+S_a)*Delta_P*np.heaviside(-Delta_P,0)
    RAa_dot += -rho*R_An*(I_Aa+E_a+S_a)*Delta_P*np.heaviside(-Delta_P,0)
    
        
    deriv = np.array([Sn_dot,Sa_dot,En_dot,Ea_dot,IS_dot, IAn_dot, IAa_dot,\
                      RS_dot, RAn_dot, RAa_dot, M_dot])

    return deriv


'''
This is another updated version of the Poletti model from equation (3) on page 83 of the
2012 paper. As before, we do not assume the same ratio for x:(1-x), as compared with S, I_A, and R.
We again have a compartment for exposed individuals (E), before they become infectious and (a)symptomatic.  
Additionally, this includes a death category, 
when symptomatic infectious do not recover.
'''
def SEIRDan_system(State_vector,t,params):
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, deathdelt = params
    
    
    epi_compartments = State_vector[:-1]
    behavior_variables = State_vector[-1]
    
    S_n, S_a, E_n, E_a, I_S, I_An, I_Aa, R_S, R_An, R_Aa, D = epi_compartments
    M = behavior_variables
    
    lambda_t = force_of_infection(I_S, I_An, I_Aa,params)
    
    ### first establish the transmission dynamics ###
    Sn_dot = -lambda_t*S_n
    Sa_dot = - q*lambda_t*S_a
    
    #includes exposed category
    En_dot = lambda_t*S_n - xi*E_n
    Ea_dot = q*lambda_t*S_a - xi*E_a
    
    IS_dot  = p*xi*(E_n + E_a) - gamma*I_S
    IAn_dot = (1-p)*xi*E_n - gamma*I_An
    IAa_dot = (1-p)*q*xi*E_a - gamma*I_Aa
    
    #currently, D_dot does not incorporate imitation effects
    D_dot = deathdelt*I_S
    
    RS_dot  = (gamma - deathdelt)*I_S
    RAn_dot = gamma*I_An
    RAa_dot = gamma*I_Aa
    
    ### second add the imitation and behavior switching dynamics ###
    M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu*I_S
    
    Delta_P = payoff_difference(M,params=params) #note: this still has extra k divided 
    
    ## intra-compartment imitation:
    
    Sn_dot += rho*(S_n*S_a*Delta_P + mu*S_a - mu*S_n)
    Sa_dot += -rho*(S_n*S_a*Delta_P + mu*S_a - mu*S_n)
    
    En_dot += rho*(E_n*E_a*Delta_P + mu*E_a - mu*E_n)
    Ea_dot += -rho*(E_n*E_a*Delta_P + mu*E_a - mu*E_n)
    
    IAn_dot += rho*(I_An*I_Aa*Delta_P + mu*I_Aa - mu*I_An)
    IAa_dot += -rho*(I_An*I_Aa*Delta_P + mu*I_Aa - mu*I_An)
    
    RAn_dot += rho*(R_An*R_Aa*Delta_P + mu*R_Aa - mu*R_An)
    RAa_dot += -rho*(R_An*R_Aa*Delta_P + mu*R_Aa - mu*R_An)
    
    
     ## inter-compartment imitation:
    
    # if Delta_P >0:
    
    Sn_dot +=  rho*S_a*(E_n+I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    Sa_dot += -rho*S_a*(E_n+I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    
    En_dot += rho*E_a*(S_n+I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    Ea_dot += -rho*E_a*(S_n+I_An+R_An)*Delta_P*np.heaviside(Delta_P,0)
    
    IAn_dot +=  rho*I_Aa*(S_n+E_n+R_An)*Delta_P*np.heaviside(Delta_P,0)
    IAa_dot += -rho*I_Aa*(S_n+E_n+R_An)*Delta_P*np.heaviside(Delta_P,0)
    
    RAn_dot +=  rho*R_Aa*(E_n+I_An+S_n)*Delta_P*np.heaviside(Delta_P,0)
    RAa_dot += -rho*R_Aa*(E_n+I_An+S_n)*Delta_P*np.heaviside(Delta_P,0)
    
    # if Delta_P <0
    
    Sn_dot +=  rho*S_n*(E_a+I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    Sa_dot += -rho*S_n*(E_a+I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    
    En_dot += rho*E_n*(S_a+I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    Ea_dot += -rho*E_n*(S_a+I_Aa+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    
    IAn_dot +=  rho*I_An*(S_a+E_a+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    IAa_dot += -rho*I_An*(S_a+E_a+R_Aa)*Delta_P*np.heaviside(-Delta_P,0)
    
    RAn_dot +=  rho*R_An*(I_Aa+E_a+S_a)*Delta_P*np.heaviside(-Delta_P,0)
    RAa_dot += -rho*R_An*(I_Aa+E_a+S_a)*Delta_P*np.heaviside(-Delta_P,0)
    
    
        
    deriv = np.array([Sn_dot,Sa_dot,En_dot,Ea_dot,IS_dot, IAn_dot, IAa_dot,\
                      RS_dot, RAn_dot, RAa_dot, D_dot, M_dot])

    return deriv



'''
SIR  with behavior dynamics (Poletti -- compartment version)

Identical to SIRan_system except that the payoff function has been adjusted.

payoffB refers to our second proposed payoff change. This one involves a feedback
between how many people are currently behaving in a certain way and the payoff.
If many people are already adopting safer behavior, then my risk is already 
decreased without my having to change behavior and inconvienice myself. 
Once enough people have adopted the safer behavior, Delta_P will flip again since
the risk of contracting the virus with normal behavior no longer outweighs the
inconvience of adopting changed behavior.

params is an optional argument which should be a list of 8 parameter values.
By default it is the hardwired global parameters defined in this file. 

Note that if you want to include this additional argument when calling odeint,
you must write:
    
    odeint(SIRan_system, State_vector, t, args=(params,))
    
    * Note the comma.

'''
def SIRan_system_payoffB(State_vector,t, params):
    
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, dealthdelt = params
    
    epi_compartments = State_vector[:-1]
    behavior_variables = State_vector[-1]
    
    S_n, S_a, I_S, I_An, I_Aa, R_S, R_An, R_Aa = epi_compartments
    M = behavior_variables
    
    lambda_t = force_of_infection(I_S, I_An, I_Aa, params)
    
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
    M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu*M
    
    Delta_P = payoffB_difference(M, S_a, I_Aa, R_Aa, params)
    
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

'''
Helper function for Poletti model


Gives P_n-P_a as guage of which behavior is more advantageous as a function 
of percieved number of cases (ie. perceived risk).

* Note: this is the version B altered payoff function. 

'''
def payoffB_difference(M, S_a, I_Aa, R_Aa, params):
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, dealthdelt = params
    
#    P_n = -m_n*((1-q*(S_a, I_Aa, R_Aa))*(1-p)*(1/p) +1)*M
#    P_a = -k - m_a**((1-q*(S_a, I_Aa, R_Aa))*(1-p)*(1/p) +1)*M
#    Delta_P = P_n-P_a # k - (m_n-m_a)*((1-q*(S_a, I_Aa, R_Aa))*(1-p)*(1/p) +1)*M
    
    Delta_P  = 1-m*((1-q*(S_a + I_Aa + R_Aa))*(1-p)*(1/p) +1)*M  # note: divided by extra k that will be fixed with rho
    
    return Delta_P
