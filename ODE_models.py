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
import math


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
Helper function for Poletti model


Gives P_n-P_a as guage of which behavior is more advantageous as a function 
of percieved number of cases (ie. perceived risk).
'''
def payoff_difference_asym(M, params):
    
    gamma,beta_S, beta_A, q, p, nu, m, rho, mu, xi, dealthdelt = params
    
#    P_n = -m_n*M
#    P_a = -k - m_a*M
#    Delta_P = P_n-P_a # k - (m_n-m_a)*M  # most intuitive
    
    Delta_P  = 1-m*(1+(1-p)/p)*M  # k*(1-m*M) = the real Delta_P  # note: divided by extra k that will be fixed with rho
    
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
def SIRan_system(State_vector,t, params, asymp_payoff=False):
    
    
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
    
    if asymp_payoff:
        Delta_P = payoff_difference_asym(M, params=params)
    else:
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
    
~~~~~~~~~~~~~
The difference here is that I am using a different relationship to mark the 
rate of switching. Instead of having an activation function and strict linear 
relationship, I have used a switching function that depends on Delta_P in the 
following way:
    
    f(Delta_P) = (1/pi) * arctan(10(Delta_P - 0.33)) + 0.5
    
~~~~~~~~~~~~~~

'''
def SIRan_system_switchingB(State_vector,t, params):
    
    
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
    
    
    f_an = ((1/np.pi)*np.arctan(10*(Delta_P-0.33))+0.5)
    f_na = ((1/np.pi)*np.arctan(10*(-Delta_P-0.33))+0.5)
    
    
    ## random switching:
    
    Sn_dot += rho*(mu*S_a - mu*S_n)
    Sa_dot += rho*(mu*S_n - mu*S_a)
    
    IAn_dot += rho*(mu*I_Aa - mu*I_An)
    IAa_dot += rho*(mu*I_An - mu*I_Aa)
    
    RAn_dot += rho*(mu*R_Aa - mu*R_An)
    RAa_dot += rho*(mu*R_An - mu*R_Aa)
    
    
     ## inter-compartment imitation:
    
    # if Delta_P >0:
    
    Sn_dot +=  rho*(S_a*(S_n+I_An+R_An))*f_an
    Sa_dot += -rho*(S_a*(S_n+I_An+R_An))*f_an
    
    IAn_dot +=  rho*(I_Aa*(S_n+I_An+R_An))*f_an
    IAa_dot += -rho*(I_Aa*(S_n+I_An+R_An))*f_an
    
    RAn_dot +=  rho*(R_Aa*(S_n+I_An+R_An))*f_an
    RAa_dot += -rho*(R_Aa*(S_n+I_An+R_An))*f_an
    
    # if Delta_P <0
    
    Sn_dot += -rho*(S_n*(S_a+I_Aa+R_Aa))*f_na
    Sa_dot +=  rho*(S_n*(S_a+I_Aa+R_Aa))*f_na
    
    IAn_dot += -rho*(I_An*(S_a+I_Aa+R_Aa))*f_na
    IAa_dot +=  rho*(I_An*(S_a+I_Aa+R_Aa))*f_na
    
    RAn_dot += -rho*(R_An*(S_a+I_Aa+R_Aa))*f_na
    RAa_dot +=  rho*(R_An*(S_a+I_Aa+R_Aa))*f_na


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
    IAa_dot = (1-p)*xi*E_a - gamma*I_Aa
    
    RS_dot  = gamma*I_S
    RAn_dot = gamma*I_An
    RAa_dot = gamma*I_Aa
    
    ### second add the imitation and behavior switching dynamics ###
    M_dot = p*(xi*(E_n + E_a)) - nu*M
    
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
    IAa_dot = (1-p)*xi*E_a - gamma*I_Aa
    
    #currently, D_dot does not incorporate imitation effects
    D_dot = deathdelt*I_S
    
    RS_dot  = (gamma - deathdelt)*I_S
    RAn_dot = gamma*I_An
    RAa_dot = gamma*I_Aa
    
    ### second add the imitation and behavior switching dynamics ###
    M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu*M
    
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
    
    Delta_P  = 1-m*((1-(1-q)*(S_a + I_Aa + R_Aa))*(1-p)*(1/p) +1)*M  # note: divided by extra k that will be fixed with rho
    
    return Delta_P


'''

This is another updated version of the Poletti model from equation (3) on page 83 of the
2012 paper, now including an "Exposed" category. Again, this model is different from the model that Poletti et. al. use
in their analysis because it does not assume that x:(1-x) gives the same ratio
for S, I_A, and R. Additionally, we have a compartment for exposed individuals (E), before
they become infectious and (a)symptomatic.

Now, this SEIR model uses arctan as opposed to Heaviside.

'''
def SEIRant_system(State_vector,t, params):
    
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
    IAa_dot = (1-p)*xi*E_a - gamma*I_Aa
    
    RS_dot  = gamma*I_S
    RAn_dot = gamma*I_An
    RAa_dot = gamma*I_Aa
    
    ### second add the imitation and behavior switching dynamics ###
    M_dot = p*(xi*(E_n + E_a)) - nu*M
    
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
        ## includes arctan as opposed to heaviside
    
    # if Delta_P >0:
    
    Sn_dot +=  rho*S_a*(E_n+I_An+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    Sa_dot += -rho*S_a*(E_n+I_An+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    
    En_dot += rho*E_a*(S_n+I_An+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    Ea_dot += -rho*E_a*(S_n+I_An+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    
    IAn_dot +=  rho*I_Aa*(S_n+E_n+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    IAa_dot += -rho*I_Aa*(S_n+E_n+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    
    RAn_dot +=  rho*R_Aa*(E_n+I_An+S_n)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    RAa_dot += -rho*R_Aa*(E_n+I_An+S_n)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    
    # if Delta_P <0
    
    Sn_dot +=  rho*S_n*(E_a+I_Aa+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    Sa_dot += -rho*S_n*(E_a+I_Aa+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    
    En_dot += rho*E_n*(S_a+I_Aa+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    Ea_dot += -rho*E_n*(S_a+I_Aa+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    
    IAn_dot +=  rho*I_An*(S_a+E_a+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    IAa_dot += -rho*I_An*(S_a+E_a+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    
    RAn_dot +=  rho*R_An*(I_Aa+E_a+S_a)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    RAa_dot += -rho*R_An*(I_Aa+E_a+S_a)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    
        
    deriv = np.array([Sn_dot,Sa_dot,En_dot,Ea_dot,IS_dot, IAn_dot, IAa_dot,\
                      RS_dot, RAn_dot, RAa_dot, M_dot])

    return deriv
  
'''
  SIR model with heaviside
  for this and the above model, could introduce parameter that determines steepness of arctan.
'''
  
def SIRant_system(State_vector,t, params):
    
    
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
    
    Sn_dot +=  rho*S_a*(I_An+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    Sa_dot += -rho*S_a*(I_An+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    
    IAn_dot +=  rho*I_Aa*(S_n+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    IAa_dot += -rho*I_Aa*(S_n+R_An)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    
    RAn_dot +=  rho*R_Aa*(I_An+S_n)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    RAa_dot += -rho*R_Aa*(I_An+S_n)*Delta_P*(np.arctan(90*Delta_P)*(1/math.pi)+0.5)
    
    # if Delta_P <0
    
    Sn_dot +=  rho*S_n*(I_Aa+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    Sa_dot += -rho*S_n*(I_Aa+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    
    IAn_dot +=  rho*I_An*(S_a+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    IAa_dot += -rho*I_An*(S_a+R_Aa)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    
    RAn_dot +=  rho*R_An*(I_Aa+S_a)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)
    RAa_dot += -rho*R_An*(I_Aa+S_a)*Delta_P*(np.arctan(-90*Delta_P)*(1/math.pi)+0.5)


    deriv = np.array([Sn_dot,Sa_dot,IS_dot, IAn_dot, IAa_dot,\
                      RS_dot, RAn_dot, RAa_dot, M_dot])
    
    return deriv

'''
accessory version allows you to specify which kind of switching and which 
kind of payoff function you want to have. 

Note that we still consider the switching mechanism to be driven by 
immitation dynamics. The difference now is that you have the option to 
specify how the switching rate per iteraction depends on Delta_P.

Payoff options:
    - original (linear in M) = 0
    - linear in M with consideration for asymptomatics = 1
    - linear in M with consideration for asymptomatics & behavior = 2
    
Switching options:
    - original (linear with heaviside activtaion) = 0
    - linear with arctan activation = 1
    - sigmoid switching (switchingB) = 2

'''
def SIRan_system_accessory(State_vector,t, params, switching=0, payoff=0):
    
    
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
    
    
    if payoff == 1:
        Delta_P = payoff_difference_asym(M, params=params)
    elif payoff == 2:
        Delta_P = payoffB_difference(M, S_a, I_Aa, R_Aa, params=params)
    else:
        Delta_P = payoff_difference(M, params=params)
    
    if switching == 1:
        f_an = np.sqrt(Delta_P**2)*(np.arctan(90*(Delta_P))+0.5)
        f_na = np.sqrt(Delta_P**2)*(np.arctan(90*(-Delta_P))+0.5)
    elif switching == 2:
        f_an = ((1/np.pi)*np.arctan(10*(Delta_P-0.33))+0.5)
        f_na = ((1/np.pi)*np.arctan(10*(-Delta_P-0.33))+0.5)
    else:
        f_an = Delta_P*np.heaviside(Delta_P,0)
        f_na = -Delta_P*np.heaviside(-Delta_P,0)
    
    ## random switching:
    
    Sn_dot += rho*(mu*S_a - mu*S_n)
    Sa_dot += rho*(mu*S_n - mu*S_a)
    
    IAn_dot += rho*(mu*I_Aa - mu*I_An)
    IAa_dot += rho*(mu*I_An - mu*I_Aa)
    
    RAn_dot += rho*(mu*R_Aa - mu*R_An)
    RAa_dot += rho*(mu*R_An - mu*R_Aa)
    
    
     ## inter-compartment imitation:
    
    # if Delta_P >0:
    
    Sn_dot +=  rho*(S_a*(S_n+I_An+R_An))*f_an
    Sa_dot += -rho*(S_a*(S_n+I_An+R_An))*f_an
    
    IAn_dot +=  rho*(I_Aa*(S_n+I_An+R_An))*f_an
    IAa_dot += -rho*(I_Aa*(S_n+I_An+R_An))*f_an
    
    RAn_dot +=  rho*(R_Aa*(S_n+I_An+R_An))*f_an
    RAa_dot += -rho*(R_Aa*(S_n+I_An+R_An))*f_an
    
    # if Delta_P <0
    
    Sn_dot += -rho*(S_n*(S_a+I_Aa+R_Aa))*f_na
    Sa_dot +=  rho*(S_n*(S_a+I_Aa+R_Aa))*f_na
    
    IAn_dot += -rho*(I_An*(S_a+I_Aa+R_Aa))*f_na
    IAa_dot +=  rho*(I_An*(S_a+I_Aa+R_Aa))*f_na
    
    RAn_dot += -rho*(R_An*(S_a+I_Aa+R_Aa))*f_na
    RAa_dot +=  rho*(R_An*(S_a+I_Aa+R_Aa))*f_na


    deriv = np.array([Sn_dot,Sa_dot,IS_dot, IAn_dot, IAa_dot,\
                      RS_dot, RAn_dot, RAa_dot, M_dot])
    
    return deriv

  '''
  New equation for force_of_infections (for the age structuring model) - to incorporate both compartments.
  '''
  
def force_of_infectioney(Ie_S, Ie_An, Ie_Aa, Iy_S, Iy_An, Iy_Aa):

    return beta_S*(Ie_S+Iy_S)+beta_A*(Ie_An+Iy_An)+qe*beta_A*Ie_Aa+qy*beta_A*Iy_Aa


  '''
  New model - with SEIRD compartments, and variation in the parameters across age compartments.  Right now, the payoff functions
  are the same in both categories - however, it would make sense to generalize in the future.  Switching dynamics determined in 
  terms of age compartments.
  '''
def ODE_systembig(State_vector,t,params):
    
#    epi_compartments = State_vector[:num_compartments]
#    behavior_variables = State_vector[num_compartments:]
    
    epi_compartments = State_vector[:-2]
    behavior_variables = State_vector[-2:]
    
    Se_n, Se_a, Sy_n, Sy_a, Ee_n, Ee_a, Ey_n, Ey_a, Ie_S, Ie_An, Ie_Aa, Iy_S, Iy_An, Iy_Aa, De, Dy, Re_S, Re_An, Re_Aa, Ry_S, Ry_An, Ry_Aa = epi_compartments
    Me, My = behavior_variables
    
    lambda_t = force_of_infectioney(Ie_S, Ie_An, Ie_Aa, Iy_S, Iy_An, Iy_Aa)
    
    ### first establish the transmission dynamics ###
    Sen_dot = -lambda_t*Se_n
    Sea_dot = - qe*lambda_t*Se_a #different rate of distancing among people who are in e
    
    Syn_dot = -lambda_t*Sy_n
    Sya_dot = - qy*lambda_t*Sy_a #different rate of distancing among people who are in y
    
    Een_dot = lambda_t*Se_n - xi*Ee_n #currently, assume xi the same
    Eea_dot = qe*lambda_t*Se_a - xi*Ee_a
    
    Eyn_dot = lambda_t*Sy_n - xi*Ey_n
    Eya_dot = qy*lambda_t*Sy_a - xi*Ey_a
    
    IeS_dot  = pe*xi*(Ee_n + qe*Ee_a) - gammae*Ie_S
    IeAn_dot = (1-pe)*xi*Ee_n - gammae*Ie_An
    IeAa_dot = (1-pe)*xi*Ee_a - gammae*Ie_Aa   
    
    IyS_dot  = py*xi*(Ey_n + qy*Ey_a) - gammay*Iy_S
    IyAn_dot = (1-py)*xi*Ey_n - gammay*Iy_An
    IyAa_dot = (1-py)*xi*Ey_a - gammay*Iy_Aa
    
    '''
     IeS_dot  = pe*xi*(Ee_n + Ee_a) - gammae*Ie_S
    IeAn_dot = (1-pe)*xi*Ee_n - gammae*Ie_An
    IeAa_dot = (1-pe)*xi*Ee_a - gammae*Ie_Aa   
    
    IyS_dot  = py*xi*(Ey_n + Ey_a) - gammay*Iy_S
    IyAn_dot = (1-py)*xi*Ey_n - gammay*Iy_An
    IyAa_dot = (1-py)*qy*xi*Ey_a - gammay*Iy_Aa
    '''
    
    ReS_dot  = (gammae - deathe)*Ie_S
    ReAn_dot = gammae*Ie_An
    ReAa_dot = gammae*Ie_Aa
    
    RyS_dot  = (gammay - deathy)*Iy_S
    RyAn_dot = gammay*Iy_An
    RyAa_dot = gammay*Iy_Aa
    
    
    De_dot = deathe*Ie_S
    Dy_dot = deathy*Iy_S
    
    
    Deltae_P = payoff_difference(Me) #note: this still has extra k divided 
    Deltay_P = payoff_difference(My)
    
    ## intra-compartment imitation:
    
    Sen_dot += rho*(Se_n*Se_a*Deltae_P + mue*Se_a - mue*Se_n)
    Sea_dot += -rho*(Se_n*Se_a*Deltae_P + mue*Se_a - mue*Se_n)
    
    Syn_dot += rho*(Sy_n*Sy_a*Deltay_P + muy*Sy_a - muy*Sy_n)
    Sya_dot += -rho*(Sy_n*Sy_a*Deltay_P + muy*Sy_a - muy*Sy_n)
    
    Een_dot += rho*(Ee_n*Ee_a*Deltae_P + mue*Ee_a - mue*Ee_n)
    Eea_dot += -rho*(Ee_n*Ee_a*Deltae_P + mue*Ee_a - mue*Ee_n)
    
    Eyn_dot += rho*(Ey_n*Ey_a*Deltay_P + muy*Ey_a - muy*Ey_n)
    Eya_dot += -rho*(Ey_n*Ey_a*Deltay_P + muy*Ey_a - muy*Ey_n)
    
    IeAn_dot += rho*(Ie_An*Ie_Aa*Deltae_P + mue*Ie_Aa - mue*Ie_An)
    IeAa_dot += -rho*(Ie_An*Ie_Aa*Deltae_P + mue*Ie_Aa - mue*Ie_An)
    
    IyAn_dot += rho*(Iy_An*Iy_Aa*Deltay_P + muy*Iy_Aa - muy*Iy_An)
    IyAa_dot += -rho*(Iy_An*Iy_Aa*Deltay_P + muy*Iy_Aa - muy*Iy_An)
    
    ReAn_dot += rho*(Re_An*Re_Aa*Deltae_P + mue*Re_Aa - mue*Re_An)
    ReAa_dot += -rho*(Re_An*Re_Aa*Deltae_P + mue*Re_Aa - mue*Re_An)
    
    RyAn_dot += rho*(Ry_An*Ry_Aa*Deltay_P + muy*Ry_Aa - muy*Ry_An)
    RyAa_dot += -rho*(Ry_An*Ry_Aa*Deltay_P + muy*Ry_Aa - muy*Ry_An)
    
    
     ## inter-compartment imitation:
        
        ## right now, am assuming that only comparing with people within compartment
    
    # if Delta_P >0:
    
    Sen_dot +=  rho*Se_a*(Ee_n+Ie_An+Re_An)*Deltae_P*np.heaviside(Deltae_P,0)
    Sea_dot += -rho*Se_a*(Ee_n+Ie_An+Re_An)*Deltae_P*np.heaviside(Deltae_P,0)
    
    Syn_dot +=  rho*Sy_a*(Ey_n+Iy_An+Ry_An)*Deltay_P*np.heaviside(Deltay_P,0)
    Sya_dot += -rho*Sy_a*(Ey_n+Iy_An+Ry_An)*Deltay_P*np.heaviside(Deltay_P,0)
    
    Een_dot += rho*Ee_a*(Se_n+Ie_An+Re_An)*Deltae_P*np.heaviside(Deltae_P,0)
    Eea_dot += -rho*Ee_a*(Se_n+Ie_An+Re_An)*Deltae_P*np.heaviside(Deltae_P,0)
    
    Eyn_dot += rho*Ey_a*(Sy_n+Iy_An+Ry_An)*Deltay_P*np.heaviside(Deltay_P,0)
    Eya_dot += -rho*Ey_a*(Sy_n+Iy_An+Ry_An)*Deltay_P*np.heaviside(Deltay_P,0)
    
    IeAn_dot +=  rho*Ie_Aa*(Se_n+Ee_n+Re_An)*Deltae_P*np.heaviside(Deltae_P,0)
    IeAa_dot += -rho*Ie_Aa*(Se_n+Ee_n+Re_An)*Deltae_P*np.heaviside(Deltae_P,0)
    
    IyAn_dot +=  rho*Iy_Aa*(Sy_n+Ey_n+Ry_An)*Deltay_P*np.heaviside(Deltay_P,0)
    IyAa_dot += -rho*Iy_Aa*(Sy_n+Ey_n+Ry_An)*Deltay_P*np.heaviside(Deltay_P,0)
    
    ReAn_dot +=  rho*Re_Aa*(Ee_n+Ie_An+Se_n)*Deltae_P*np.heaviside(Deltae_P,0)
    ReAa_dot += -rho*Re_Aa*(Ee_n+Ie_An+Se_n)*Deltae_P*np.heaviside(Deltae_P,0)
    
    RyAn_dot +=  rho*Ry_Aa*(Ey_n+Iy_An+Sy_n)*Deltay_P*np.heaviside(Deltay_P,0)
    RyAa_dot += -rho*Ry_Aa*(Ey_n+Iy_An+Sy_n)*Deltay_P*np.heaviside(Deltay_P,0)
    
    # if Delta_P <0
    
    Sen_dot +=  rho*Se_n*(Ee_a+Ie_Aa+Re_Aa)*Deltae_P*np.heaviside(-Deltae_P,0)
    Sea_dot += -rho*Se_n*(Ee_a+Ie_Aa+Re_Aa)*Deltae_P*np.heaviside(-Deltae_P,0)
    
    Een_dot += rho*Ee_n*(Se_a+Ie_Aa+Re_Aa)*Deltae_P*np.heaviside(-Deltae_P,0)
    Eea_dot += -rho*Ee_n*(Se_a+Ie_Aa+Re_Aa)*Deltae_P*np.heaviside(-Deltae_P,0)
    
    IeAn_dot +=  rho*Ie_An*(Se_a+Ee_a+Re_Aa)*Deltae_P*np.heaviside(-Deltae_P,0)
    IeAa_dot += -rho*Ie_An*(Se_a+Ee_a+Re_Aa)*Deltae_P*np.heaviside(-Deltae_P,0)
    
    ReAn_dot +=  rho*Re_An*(Ie_Aa+Ee_a+Se_a)*Deltae_P*np.heaviside(-Deltae_P,0)
    ReAa_dot += -rho*Re_An*(Ie_Aa+Ee_a+Se_a)*Deltae_P*np.heaviside(-Deltae_P,0)
    
    Syn_dot +=  rho*Sy_n*(Ey_a+Iy_Aa+Ry_Aa)*Deltay_P*np.heaviside(-Deltay_P,0)
    Sya_dot += -rho*Sy_n*(Ey_a+Iy_Aa+Ry_Aa)*Deltay_P*np.heaviside(-Deltay_P,0)
    
    Eyn_dot += rho*Ey_n*(Sy_a+Iy_Aa+Ry_Aa)*Deltay_P*np.heaviside(-Deltay_P,0)
    Eya_dot += -rho*Ey_n*(Sy_a+Iy_Aa+Ry_Aa)*Deltay_P*np.heaviside(-Deltay_P,0)
    
    IyAn_dot +=  rho*Iy_An*(Sy_a+Ey_a+Ry_Aa)*Deltay_P*np.heaviside(-Deltay_P,0)
    IyAa_dot += -rho*Iy_An*(Sy_a+Ey_a+Ry_Aa)*Deltay_P*np.heaviside(-Deltay_P,0)
    
    RyAn_dot +=  rho*Ry_An*(Iy_Aa+Ey_a+Sy_a)*Deltay_P*np.heaviside(-Deltay_P,0)
    RyAa_dot += -rho*Ry_An*(Iy_Aa+Ey_a+Sy_a)*Deltay_P*np.heaviside(-Deltay_P,0)
    
    Me_dot = pe*(xi*(Ee_n + Ee_a+Ey_n + Ey_a)) - nu*Me #currently, determined by all exposed
    My_dot = py*(xi*(Ee_n + Ee_a+Ey_n + Ey_a)) - nu*My
    
        
    deriv = np.array([Sen_dot,Sea_dot,Syn_dot,Sya_dot,Een_dot,Eea_dot,Eyn_dot,Eya_dot,IeS_dot, IeAn_dot, IeAa_dot,IyS_dot, IyAn_dot, IyAa_dot, De_dot, Dy_dot, ReS_dot, ReAn_dot, ReAa_dot, RyS_dot, RyAn_dot, RyAa_dot, Me_dot, My_dot])

    return deriv






