
# coding: utf-8

# In[9]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# In[10]:


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
SIR model
'''
def SIR_system(State_vector, t):
    
    S, I, R = State_vector
    
    S_dot = -beta_S*S*I
    I_dot = beta_S*S*I - gamma*I
    R_dot = gamma*I
    
    deriv = np.array([S_dot, I_dot, R_dot])
    
    return deriv


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
    
    
    
    ### Giving up on the if statements for continuity issues. ###
    ## inter-compartment imitation:
    
#    if Delta_P > 0: # note that sign should still be the same since k>0
#        
#        Sn_dot +=  rho*S_a*(I_An+R_An)*Delta_P
#        Sa_dot += -rho*S_a*(I_An+R_An)*Delta_P
#        
#        IAn_dot +=  rho*I_Aa*(S_n+R_An)*Delta_P
#        IAa_dot += -rho*I_Aa*(S_n+R_An)*Delta_P
#        
#        RAn_dot +=  rho*R_Aa*(I_An+S_n)*Delta_P
#        RAa_dot += -rho*R_Aa*(I_An+S_n)*Delta_P
#        
#    elif Delta_P <0: # is the sign correct here? Should it be abs[Delta_P]?
#        
#        Sn_dot += -rho*S_n*(I_Aa+R_Aa)*Delta_P
#        Sa_dot +=  rho*S_n*(I_Aa+R_Aa)*Delta_P
#        
#        IAn_dot += -rho*I_An*(S_a+R_Aa)*Delta_P
#        IAa_dot +=  rho*I_An*(S_a+R_Aa)*Delta_P
#        
#        RAn_dot += -rho*R_An*(I_Aa+S_a)*Delta_P
#        RAa_dot +=  rho*R_An*(I_Aa+S_a)*Delta_P
        
    deriv = np.array([Sn_dot,Sa_dot,IS_dot, IAn_dot, IAa_dot,                      RS_dot, RAn_dot, RAa_dot, M_dot])
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
#    Delta_P = P_n-P_a # k - (m_n-m_a)*M  # most intuitive
    
    Delta_P  = 1-m*M  # k*(1-m*M) = the real Delta_P  # note: divided by extra k that will be fixed with rho
    
    return Delta_P

def plot_SIR(time, solution, reg_SIR=False):
    
    fig, ax = plt.subplots()  
    
    if reg_SIR == False:
        S = solution[:,0]+solution[:,1]
        I = solution[:,2]+solution[:,3]+solution[:,4]
        R = solution[:,5]+solution[:,6]+solution[:,7]
    else:
        S = solution[:,0]
        I = solution[:,1]
        R = solution[:,2]

    ax.plot(time, S, '-b')
    ax.plot(time, I, '-r')
    ax.plot(time, R, '-g')
    
    ax.set(xlabel='time (days)', ylabel='SIR Dynamics')
    ax.set_ylim((0,1))
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
    
    t = np.arange(0,100,0.0005)
    
    solution = odeint(ODE_system,initial_state,t)
#    solution = odeint(SIR_system, np.array([0.99,0.01,0]),t)
    
    plot_SIR(t, solution)
    
    
    return

main()


# In[20]:


## Global parameters ##

num_compartments = 10 # the number of compartments in the epi part of the model

recovery_days = 2.8 # average length of infection (current baseline from 2012)
gamma = 1/recovery_days # rate of recovery

exposed_days = 3 # length of time in exposed before becoming infectious; set to 3 currently
xi = 1/exposed_days # rate of movement from E to I

beta_S = 0.7 # infection rate for symptomatic infectives
beta_A = 0.7 # infection rate for asymptomatic infectives

q = 0.99  # reduction factor for those with altered/adjusted behavior
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
SEIR model
'''
#updated SEIR model
def SEIR_system(State_vector, t):
    
    S, E, I, R = State_vector
    
    S_dot = -beta_S*S*I
    E_dot = beta_S*S*I - delta*E
    I_dot = delta*E - gamma*I
    R_dot = gamma*I
    
    deriv = np.array([S_dot, E_dot, I_dot, R_dot])
    
    return deriv


'''
This is an updated version of the Poletti model from equation (3) on page 83 of the
2012 paper. This model is different from the model that Poletti et. al. use
in their analysis because it does not assume that x:(1-x) gives the same ratio
for S, I_A, and R. And now, we have added a compartment for exposed individuals (E), before
they become infectious and (a)symptomatic.
'''
def ODE_system(State_vector,t):
    
#    epi_compartments = State_vector[:num_compartments]
#    behavior_variables = State_vector[num_compartments:]
    
    epi_compartments = State_vector[:-1]
    behavior_variables = State_vector[-1]
    
    S_n, S_a, E_n, E_a, I_S, I_An, I_Aa, R_S, R_An, R_Aa = epi_compartments
    M = behavior_variables
    
    lambda_t = force_of_infection(I_S, I_An, I_Aa)
    
    ### first establish the transmission dynamics ###
    Sn_dot = -lambda_t*S_n
    Sa_dot = - q*lambda_t*S_a
    
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
    
    Delta_P = payoff_difference(M) #note: this still has extra k divided 
    
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
    
    
    
    ### Giving up on the if statements for continuity issues. ###
    ## inter-compartment imitation:
    
#    if Delta_P > 0: # note that sign should still be the same since k>0
#        
#        Sn_dot +=  rho*S_a*(I_An+R_An)*Delta_P
#        Sa_dot += -rho*S_a*(I_An+R_An)*Delta_P
#        
#        IAn_dot +=  rho*I_Aa*(S_n+R_An)*Delta_P
#        IAa_dot += -rho*I_Aa*(S_n+R_An)*Delta_P
#        
#        RAn_dot +=  rho*R_Aa*(I_An+S_n)*Delta_P
#        RAa_dot += -rho*R_Aa*(I_An+S_n)*Delta_P
#        
#    elif Delta_P <0: # is the sign correct here? Should it be abs[Delta_P]?
#        
#        Sn_dot += -rho*S_n*(I_Aa+R_Aa)*Delta_P
#        Sa_dot +=  rho*S_n*(I_Aa+R_Aa)*Delta_P
#        
#        IAn_dot += -rho*I_An*(S_a+R_Aa)*Delta_P
#        IAa_dot +=  rho*I_An*(S_a+R_Aa)*Delta_P
#        
#        RAn_dot += -rho*R_An*(I_Aa+S_a)*Delta_P
#        RAa_dot +=  rho*R_An*(I_Aa+S_a)*Delta_P
        
    deriv = np.array([Sn_dot,Sa_dot,En_dot,Ea_dot,IS_dot, IAn_dot, IAa_dot,                      RS_dot, RAn_dot, RAa_dot, M_dot])
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
#    Delta_P = P_n-P_a # k - (m_n-m_a)*M  # most intuitive
    
    Delta_P  = 1-m*M  # k*(1-m*M) = the real Delta_P  # note: divided by extra k that will be fixed with rho
    
    return Delta_P

def plot_SIR(time, solution, reg_SIR=False):
    
    fig, ax = plt.subplots()  
    
    if reg_SIR == False:
        S = solution[:,0]+solution[:,1]
        E = solution[:,2]+solution[:,3]
        I = solution[:,4]+solution[:,5]+solution[:,6]
        R = solution[:,7]+solution[:,8]+solution[:,9]
    else:
        S = solution[:,0]
        E = solution[:,1]
        I = solution[:,2]
        R = solution[:,3]

    ax.plot(time, S, '-b')
    ax.plot(time, E, '-y')
    ax.plot(time, I, '-r')
    ax.plot(time, R, '-g')
    
    ax.set(xlabel='time (days)', ylabel='SEIR Dynamics')
    ax.set_ylim((0,1))
    ax.grid()
    
    return

def main():
    
    Sn_0  = 0.99
    Sa_0  = 0.002
    En_0 = 0.002
    Ea_0 = 0.003
    IS_0  = 0.001
    IAn_0 = 0.002
    IAa_0 = 0
    RS_0  = 0
    RAn_0 = 0
    RAa_0 = 0
    
    M_0 = 0.0001  # prior belief of risk (or "overestimation" of risk)
    
    initial_state = np.array([Sn_0,Sa_0,En_0,Ea_0,IS_0,IAn_0,IAa_0,RS_0,RAn_0,RAa_0,M_0])
    
    if np.sum(initial_state)- M_0 != 1:
        print("Error: make sure population sums to 1")
        return
    
    t = np.arange(0,100,0.0005)
    
    solution = odeint(ODE_system,initial_state,t)
#    solution = odeint(SIR_system, np.array([0.99,0.01,0]),t)
    
    plot_SIR(t, solution)
    
    
    return

main()



# In[23]:


## Global parameters ##

num_compartments = 10 # the number of compartments in the epi part of the model

recovery_days = 2.8 # average length of infection (current baseline from 2012)
gamma = 1/recovery_days # rate of recovery

exposed_days = 3 # length of time in exposed before becoming infectious; set to 3 currently
xi = 1/exposed_days # rate of movement from E to I

beta_S = 0.7 # infection rate for symptomatic infectives
beta_A = 0.7 # infection rate for asymptomatic infectives

q = 0.99  # reduction factor for those with altered/adjusted behavior
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
SEIR model
'''
#updated SEIR model
def SEIR_system(State_vector, t):
    
    S, E, I, R = State_vector
    
    S_dot = -beta_S*S*I
    E_dot = beta_S*S*I - delta*E
    I_dot = delta*E - gamma*I
    R_dot = gamma*I
    
    deriv = np.array([S_dot, E_dot, I_dot, R_dot])
    
    return deriv


'''
This is an updated version of the Poletti model from equation (3) on page 83 of the
2012 paper. This model is different from the model that Poletti et. al. use
in their analysis because it does not assume that x:(1-x) gives the same ratio
for S, I_A, and R. And now, we have added a compartment for exposed individuals (E), before
they become infectious and (a)symptomatic.
'''
def ODE_system(State_vector,t):
    
#    epi_compartments = State_vector[:num_compartments]
#    behavior_variables = State_vector[num_compartments:]
    
    epi_compartments = State_vector[:-1]
    behavior_variables = State_vector[-1]
    
    S_n, S_a, E_n, E_a, I_S, I_An, I_Aa, R_S, R_An, R_Aa = epi_compartments
    M = behavior_variables
    
    lambda_t = force_of_infection(I_S, I_An, I_Aa)
    
    ### first establish the transmission dynamics ###
    Sn_dot = -lambda_t*S_n
    Sa_dot = - q*lambda_t*S_a
    
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
    ### no matter what M_dot is changed to, it does not change.
    
    Delta_P = payoff_difference(M) #note: this still has extra k divided 
    
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
    
    
    
    ### Giving up on the if statements for continuity issues. ###
    ## inter-compartment imitation:
    
#    if Delta_P > 0: # note that sign should still be the same since k>0
#        
#        Sn_dot +=  rho*S_a*(I_An+R_An)*Delta_P
#        Sa_dot += -rho*S_a*(I_An+R_An)*Delta_P
#        
#        IAn_dot +=  rho*I_Aa*(S_n+R_An)*Delta_P
#        IAa_dot += -rho*I_Aa*(S_n+R_An)*Delta_P
#        
#        RAn_dot +=  rho*R_Aa*(I_An+S_n)*Delta_P
#        RAa_dot += -rho*R_Aa*(I_An+S_n)*Delta_P
#        
#    elif Delta_P <0: # is the sign correct here? Should it be abs[Delta_P]?
#        
#        Sn_dot += -rho*S_n*(I_Aa+R_Aa)*Delta_P
#        Sa_dot +=  rho*S_n*(I_Aa+R_Aa)*Delta_P
#        
#        IAn_dot += -rho*I_An*(S_a+R_Aa)*Delta_P
#        IAa_dot +=  rho*I_An*(S_a+R_Aa)*Delta_P
#        
#        RAn_dot += -rho*R_An*(I_Aa+S_a)*Delta_P
#        RAa_dot +=  rho*R_An*(I_Aa+S_a)*Delta_P
        
    deriv = np.array([Sn_dot,Sa_dot,En_dot,Ea_dot,IS_dot, IAn_dot, IAa_dot,                      RS_dot, RAn_dot, RAa_dot, M_dot])
#    deriv = np.zeros(9)
    print(M)
    print(S_n)
    print(Delta_P)
    
    ### currently, Delta_P is always negative
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
#    Delta_P = P_n-P_a # k - (m_n-m_a)*M  # most intuitive
    
    Delta_P  = 1-m*M  # k*(1-m*M) = the real Delta_P  # note: divided by extra k that will be fixed with rho
    
    return Delta_P

def plot_SIR(time, solution, reg_SIR=False):
    
    fig, ax = plt.subplots()  
    
    if reg_SIR == False:
        S = solution[:,0]+solution[:,1]
        E = solution[:,2]+solution[:,3]
        I = solution[:,4]+solution[:,5]+solution[:,6]
        R = solution[:,7]+solution[:,8]+solution[:,9]
    else:
        S = solution[:,0]
        E = solution[:,1]
        I = solution[:,2]
        R = solution[:,3]

    ax.plot(time, S, '-b')
    ax.plot(time, E, '-y')
    ax.plot(time, I, '-r')
    ax.plot(time, R, '-g')
    
    ax.set(xlabel='time (days)', ylabel='SEIR Dynamics')
    ax.set_ylim((0,1))
    ax.grid()
    
    return

def main():
    
    Sn_0  = 0.99
    Sa_0  = 0.002
    En_0 = 0.002
    Ea_0 = 0.003
    IS_0  = 0.001
    IAn_0 = 0.002
    IAa_0 = 0
    RS_0  = 0
    RAn_0 = 0
    RAa_0 = 0
    
    M_0 = 0.3  # prior belief of risk (or "overestimation" of risk)
    
    initial_state = np.array([Sn_0,Sa_0,En_0,Ea_0,IS_0,IAn_0,IAa_0,RS_0,RAn_0,RAa_0,M_0])
    
    if np.sum(initial_state)- M_0 != 1:
        print("Error: make sure population sums to 1")
        return
    
    t = np.arange(0,100,0.0005)
    
    solution = odeint(ODE_system,initial_state,t)
#    solution = odeint(SIR_system, np.array([0.99,0.01,0]),t)
    
    plot_SIR(t, solution)
    
    
    return

main()




# In[19]:


## Global parameters ##

num_compartments = 11 # the number of compartments in the epi part of the model

recovery_days = 2.8 # average length of infection (current baseline from 2012)
gamma = 1/recovery_days # rate of recovery

exposed_days = 3 # length of time in exposed before becoming infectious; set to 3 currently
xi = 1/exposed_days # rate of movement from E to I

delta =  0.2 #death rate given symptomatic
                #should probably name something else

beta_S = 0.7 # infection rate for symptomatic infectives
beta_A = 0.7 # infection rate for asymptomatic infectives

q = 0.99  # reduction factor for those with altered/adjusted behavior
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
SEIR model
'''
#updated SEIR model
def SEIR_system(State_vector, t):
    
    S, E, I, R = State_vector
    
    S_dot = -beta_S*S*I
    E_dot = beta_S*S*I - delta*E
    I_dot = delta*E - gamma*I
    R_dot = gamma*I
    
    deriv = np.array([S_dot, E_dot, I_dot, R_dot])
    
    return deriv


'''
This is an updated version of the Poletti model from equation (3) on page 83 of the
2012 paper. This model is different from the model that Poletti et. al. use
in their analysis because it does not assume that x:(1-x) gives the same ratio
for S, I_A, and R. And now, we have added a compartment for exposed individuals (E), before
they become infectious and (a)symptomatic.  Additionally, this includes a death category, 
when symptomatic infectious do not recover.
'''
def ODE_system(State_vector,t):
    
#    epi_compartments = State_vector[:num_compartments]
#    behavior_variables = State_vector[num_compartments:]
    
    epi_compartments = State_vector[:-1]
    behavior_variables = State_vector[-1]
    
    S_n, S_a, E_n, E_a, I_S, I_An, I_Aa, R_S, R_An, R_Aa, D = epi_compartments
    M = behavior_variables
    
    lambda_t = force_of_infection(I_S, I_An, I_Aa)
    
    ### first establish the transmission dynamics ###
    Sn_dot = -lambda_t*S_n
    Sa_dot = - q*lambda_t*S_a
    
    En_dot = lambda_t*S_n - xi*E_n
    Ea_dot = q*lambda_t*S_a - xi*E_a
    
    IS_dot  = p*xi*(E_n + E_a) - gamma*I_S
    IAn_dot = (1-p)*xi*E_n - gamma*I_An
    IAa_dot = (1-p)*q*xi*E_a - gamma*I_Aa
    
    D_dot = delta*I_S
    
    RS_dot  = (gamma - delta)*I_S
    RAn_dot = gamma*I_An
    RAa_dot = gamma*I_Aa
    
    ### second add the imitation and behavior switching dynamics ###
    M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu*M
    ### no matter what M_dot is changed to, the output does not appear to change
    ### this is some issue with the code
    ### M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu1*IS_dot
    ### M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu2*D
    ### M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu2*D_dot
    ### 
    
    Delta_P = payoff_difference(M) #note: this still has extra k divided 
    
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
    
    
    
    ### Giving up on the if statements for continuity issues. ###
    ## inter-compartment imitation:
    
#    if Delta_P > 0: # note that sign should still be the same since k>0
#        
#        Sn_dot +=  rho*S_a*(I_An+R_An)*Delta_P
#        Sa_dot += -rho*S_a*(I_An+R_An)*Delta_P
#        
#        IAn_dot +=  rho*I_Aa*(S_n+R_An)*Delta_P
#        IAa_dot += -rho*I_Aa*(S_n+R_An)*Delta_P
#        
#        RAn_dot +=  rho*R_Aa*(I_An+S_n)*Delta_P
#        RAa_dot += -rho*R_Aa*(I_An+S_n)*Delta_P
#        
#    elif Delta_P <0: # is the sign correct here? Should it be abs[Delta_P]?
#        
#        Sn_dot += -rho*S_n*(I_Aa+R_Aa)*Delta_P
#        Sa_dot +=  rho*S_n*(I_Aa+R_Aa)*Delta_P
#        
#        IAn_dot += -rho*I_An*(S_a+R_Aa)*Delta_P
#        IAa_dot +=  rho*I_An*(S_a+R_Aa)*Delta_P
#        
#        RAn_dot += -rho*R_An*(I_Aa+S_a)*Delta_P
#        RAa_dot +=  rho*R_An*(I_Aa+S_a)*Delta_P
        
    deriv = np.array([Sn_dot,Sa_dot,En_dot,Ea_dot,IS_dot, IAn_dot, IAa_dot,                      RS_dot, RAn_dot, RAa_dot, D_dot, M_dot])
#    deriv = np.zeros(9)
    print(M)
    print(S_n)
    print(Delta_P)
    
    ### currently, Delta_P is always negative
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
#    Delta_P = P_n-P_a # k - (m_n-m_a)*M  # most intuitive
    
    Delta_P  = 1-m*M  # k*(1-m*M) = the real Delta_P  # note: divided by extra k that will be fixed with rho
    
    return Delta_P

def plot_SIR(time, solution, reg_SIR=False):
    
    fig, ax = plt.subplots()  
    
    if reg_SIR == False:
        S = solution[:,0]+solution[:,1]
        E = solution[:,2]+solution[:,3]
        I = solution[:,4]+solution[:,5]+solution[:,6]
        R = solution[:,7]+solution[:,8]+solution[:,9]
        D = solution[:,10]
    else:
        S = solution[:,0]
        E = solution[:,1]
        I = solution[:,2]
        R = solution[:,3]
        D = solution[:,4]

    ax.plot(time, S, '-o')
    ax.plot(time, E, '-y')
    ax.plot(time, I, '-r')
    ax.plot(time, R, '-g')
    ax.plot(time, D, '-b')
    
    ax.set(xlabel='time (days)', ylabel='SEIRD Dynamics')
    ax.set_ylim((0,1))
    ax.grid()
    
    return

def main():
    
    Sn_0  = 0.99
    Sa_0  = 0.002
    En_0 = 0.002
    Ea_0 = 0.003
    IS_0  = 0.001
    IAn_0 = 0.002
    IAa_0 = 0
    RS_0  = 0
    RAn_0 = 0
    RAa_0 = 0
    D_0 = 0
    
    M_0 = 0.3  # prior belief of risk (or "overestimation" of risk)
    
    initial_state = np.array([Sn_0,Sa_0,En_0,Ea_0,IS_0,IAn_0,IAa_0,RS_0,RAn_0,RAa_0,D_0,M_0])
    
    if np.sum(initial_state)- M_0 != 1:
        print("Error: make sure population sums to 1")
        return
    
    t = np.arange(0,100,0.0005)
    
    solution = odeint(ODE_system,initial_state,t)
#    solution = odeint(SIR_system, np.array([0.99,0.01,0]),t)
    
    plot_SIR(t, solution)
    
    
    return

main()





# In[21]:


## Global parameters ##

num_compartments = 11 # the number of compartments in the epi part of the model

recovery_days = 2.8 # average length of infection (current baseline from 2012)
gamma = 1/recovery_days # rate of recovery

exposed_days = 3 # length of time in exposed before becoming infectious; set to 3 currently
xi = 1/exposed_days # rate of movement from E to I

delta =  0.2 #death rate given symptomatic
                #should probably name something else

beta_S = 0.7 # infection rate for symptomatic infectives
beta_A = 0.7 # infection rate for asymptomatic infectives

q = 0.99  # reduction factor for those with altered/adjusted behavior
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
SEIR model
'''
#updated SEIR model
def SEIR_system(State_vector, t):
    
    S, E, I, R = State_vector
    
    S_dot = -beta_S*S*I
    E_dot = beta_S*S*I - delta*E
    I_dot = delta*E - gamma*I
    R_dot = gamma*I
    
    deriv = np.array([S_dot, E_dot, I_dot, R_dot])
    
    return deriv


'''
This is an updated version of the Poletti model from equation (3) on page 83 of the
2012 paper. This model is different from the model that Poletti et. al. use
in their analysis because it does not assume that x:(1-x) gives the same ratio
for S, I_A, and R. And now, we have added a compartment for exposed individuals (E), before
they become infectious and (a)symptomatic.  Additionally, this includes a death category, 
when symptomatic infectious do not recover.
'''
def ODE_system(State_vector,t):
    
#    epi_compartments = State_vector[:num_compartments]
#    behavior_variables = State_vector[num_compartments:]
    
    epi_compartments = State_vector[:-1]
    behavior_variables = State_vector[-1]
    
    S_n, S_a, E_n, E_a, I_S, I_An, I_Aa, R_S, R_An, R_Aa, D = epi_compartments
    M = behavior_variables
    
    lambda_t = force_of_infection(I_S, I_An, I_Aa)
    
    ### first establish the transmission dynamics ###
    Sn_dot = -lambda_t*S_n
    Sa_dot = - q*lambda_t*S_a
    
    En_dot = lambda_t*S_n - xi*E_n
    Ea_dot = q*lambda_t*S_a - xi*E_a
    
    IS_dot  = p*xi*(E_n + E_a) - gamma*I_S
    IAn_dot = (1-p)*xi*E_n - gamma*I_An
    IAa_dot = (1-p)*q*xi*E_a - gamma*I_Aa
    
    D_dot = delta*I_S
    
    RS_dot  = (gamma - delta)*I_S
    RAn_dot = gamma*I_An
    RAa_dot = gamma*I_Aa
    
    ### second add the imitation and behavior switching dynamics ###
    M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu*I_S
    ### no matter what M_dot is changed to, the output does not appear to change
    ### this is some issue with the code
    ### M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu1*IS_dot
    ### M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu2*D
    ### M_dot = p*(lambda_t*S_n + q*lambda_t*S_a) - nu2*D_dot
    ### 
    
    Delta_P = payoff_difference(M) #note: this still has extra k divided 
    
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
    
    
    
    ### Giving up on the if statements for continuity issues. ###
    ## inter-compartment imitation:
    
#    if Delta_P > 0: # note that sign should still be the same since k>0
#        
#        Sn_dot +=  rho*S_a*(I_An+R_An)*Delta_P
#        Sa_dot += -rho*S_a*(I_An+R_An)*Delta_P
#        
#        IAn_dot +=  rho*I_Aa*(S_n+R_An)*Delta_P
#        IAa_dot += -rho*I_Aa*(S_n+R_An)*Delta_P
#        
#        RAn_dot +=  rho*R_Aa*(I_An+S_n)*Delta_P
#        RAa_dot += -rho*R_Aa*(I_An+S_n)*Delta_P
#        
#    elif Delta_P <0: # is the sign correct here? Should it be abs[Delta_P]?
#        
#        Sn_dot += -rho*S_n*(I_Aa+R_Aa)*Delta_P
#        Sa_dot +=  rho*S_n*(I_Aa+R_Aa)*Delta_P
#        
#        IAn_dot += -rho*I_An*(S_a+R_Aa)*Delta_P
#        IAa_dot +=  rho*I_An*(S_a+R_Aa)*Delta_P
#        
#        RAn_dot += -rho*R_An*(I_Aa+S_a)*Delta_P
#        RAa_dot +=  rho*R_An*(I_Aa+S_a)*Delta_P
        
    deriv = np.array([Sn_dot,Sa_dot,En_dot,Ea_dot,IS_dot, IAn_dot, IAa_dot,                      RS_dot, RAn_dot, RAa_dot, D_dot, M_dot])
#    deriv = np.zeros(9)
    print(M)
    print(S_n)
    print(Delta_P)
    
    ### currently, Delta_P is always negative
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
#    Delta_P = P_n-P_a # k - (m_n-m_a)*M  # most intuitive
    
    Delta_P  = 1-m*M  # k*(1-m*M) = the real Delta_P  # note: divided by extra k that will be fixed with rho
    
    return Delta_P

def plot_SIR(time, solution, reg_SIR=False):
    
    fig, ax = plt.subplots()  
    
    if reg_SIR == False:
        S = solution[:,0]+solution[:,1]
        E = solution[:,2]+solution[:,3]
        I = solution[:,4]+solution[:,5]+solution[:,6]
        R = solution[:,7]+solution[:,8]+solution[:,9]
        D = solution[:,10]
    else:
        S = solution[:,0]
        E = solution[:,1]
        I = solution[:,2]
        R = solution[:,3]
        D = solution[:,4]

    ax.plot(time, S, '-o')
    ax.plot(time, E, '-y')
    ax.plot(time, I, '-r')
    ax.plot(time, R, '-g')
    ax.plot(time, D, '-b')
    
    ax.set(xlabel='time (days)', ylabel='SEIRD Dynamics')
    ax.set_ylim((0,1))
    ax.grid()
    
    return

def main():
    
    Sn_0  = 0.99
    Sa_0  = 0.002
    En_0 = 0.002
    Ea_0 = 0.003
    IS_0  = 0.001
    IAn_0 = 0.002
    IAa_0 = 0
    RS_0  = 0
    RAn_0 = 0
    RAa_0 = 0
    D_0 = 0
    
    M_0 = 0.3  # prior belief of risk (or "overestimation" of risk)
    
    initial_state = np.array([Sn_0,Sa_0,En_0,Ea_0,IS_0,IAn_0,IAa_0,RS_0,RAn_0,RAa_0,D_0,M_0])
    
    if np.sum(initial_state)- M_0 != 1:
        print("Error: make sure population sums to 1")
        return
    
    t = np.arange(0,100,0.0005)
    
    solution = odeint(ODE_system,initial_state,t)
#    solution = odeint(SIR_system, np.array([0.99,0.01,0]),t)
    
    plot_SIR(t, solution)
    
    
    return

main()





