import networkx as nx
import numpy as np
import nolds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.random import choice
from numpy.random import uniform
from doepy import build
import pandas as pd
import time

NUM_FLOOR=1e-4

def logistic(x,r):
    return r*x*(1-x)

def random_attributes(G,seed_set_p=0.05):
    nodes=G.nodes()
    n_nodes=G.number_of_nodes()
    #Nodes have three attributes: Threshold, Budget and a State Vector.
    thresholds = uniform(0,1,n_nodes)
    budgets    = uniform(0,1,n_nodes)

    states=np.array([0,1])
    state_vectors=choice(states,size=n_nodes,p=[1-seed_set_p,seed_set_p])
    
    attributes=dict()
    for node,th,bg,state in zip(nodes,thresholds,budgets,state_vectors):
        attributes[node]={
            'threshold':th,
            'budget':bg,
            'states':[state]
            }
        
    return attributes

def ngm_sum(node, G, prod_qual):
    neighbors=G.neighbors(node)
    state_vectors=[]
    for neighbor in neighbors:
        neighbor_attributes=G.nodes()[neighbor]
        states=neighbor_attributes['states']
        state_vectors.append(states)
    state_matrix=np.array(state_vectors)
    return state_matrix.dot(prod_qual).sum()

def influence(node,G,prod,place,promo):
    influence=ngm_sum(node,G,prod)*place*promo
    return influence

def activation(node,G,prod,price,place,promo):
    na=G.nodes()[node]
    threshold,budget=na['threshold'],na['budget']
    
    inf=influence(node,G,prod,place,promo)
    if inf>=threshold and budget>=price:
        active = 1
    else:
        active = 0
    return active

def random_mkt_mix(budget,periods,granularity=1000):
    x= np.linspace(0,1,periods)
    mkt_mix=[]
    expenditure=float('inf')
    while(expenditure>budget):
        expenditure=0
        for i in range(4):
            m=uniform(-1,1)
            b=uniform(0,1)
            y=m*x+b
            y[y<0]=0
            y[y>1]=1
            #y = MinMaxScaler(feature_range=(0,1)).fit_transform(y.reshape(-1, 1)).reshape(-1)
            mkt_mix.append(y)
            if i!=3 and i!=1: #price and promo don't count for budget
                expenditure+=np.sum(y)
        #print('exp='+str(expenditure)+" budget="+str(budget))
    mkt_mix=np.array(mkt_mix)+NUM_FLOOR
    return mkt_mix

def simulate_control(n,p,seed_set_p,mkt_mix,promo_budget=300,r_promo=None,r_price=None,last=25):
    periods=mkt_mix.shape[1]
    G=nx.gnp_random_graph(n,p)
    attributes=random_attributes(G,seed_set_p)
    nx.set_node_attributes(G,attributes)
    prods,prices,places,promos=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
    
    nodes=[n for n in G.nodes]
    promo_cost=0
    previous_state=np.sum([attr['states'][0] for attr in attributes.values()])
    for t in range(periods):
        #prod_control=logistic(previous_state/n,r_promo)+NUM_FLOOR
        price_control=logistic(1-(previous_state/n),r_price)+NUM_FLOOR
        #place_control=logistic(previous_state/n,r_promo)+NUM_FLOOR
        if promo_cost<promo_budget:
            promo_control=logistic(price_control*(previous_state/n),r_promo)+NUM_FLOOR
        else:
            print("ran out of promo budget")
            promo_control=0
        promo_cost+=promo_control
        if t%10==0:
            print("t="+str(t+1)+" adoption="+str(round(previous_state/n,2))+", curr_ad_exp="+str(round(promo_control,2)))#+" curr_price="+str(round(price_control,2))+" curr_ad_exp="+str(round(promo_control,2))+" cum_ad_exp="+str(round(promo_cost,2)))
        prod,price,place,promo=prods[t],price_control,places[t],promo_control
        #prod,price,place,promo=prod_control,price_control,place_control,promo_control
        states=dict()
        for node in nodes:
            curr_state=[s for s in G.nodes()[node]['states']]
            curr_state.append(activation(node,G,prod,price,place,promo))
            states[node]=curr_state
        nx.set_node_attributes(G, states, "states")   
        previous_state=np.sum([st[-1] for st in nx.get_node_attributes(G,'states').values()])
        #if previous_state==0 and t<periods-last:
            #print("t="+str(t+1)+" adoption="+str(previous_state)+" cum_ad_exp="+str(round(promo_cost,2)))
            #raise ValueError
    state_matrix=np.array(list(nx.get_node_attributes(G,"states").values()))
    state_matrix=np.array(state_matrix)/n
    #print("t="+str(t+1)+" adoption="+str(previous_state)+" cum_ad_exp="+str(round(promo_cost,2)))
    return state_matrix

def simulate_system(doe,budget,promo_budget,n,seed_set_p,iterations,sims,last):
    print("Starting simulation with "+str(len(doe))+" exp. units each with "+str(sims)+" Monte Carlo simulations: a total of "+str(len(doe)*sims)+" runs...")
    steadystates=[]
    lyapunovs=[]
    for unit in doe:
        start = time.time()
        #r_promo=unit[0]
        #r_price=unit[1]
        p=unit[2]
        r_promo=unit[0]
        r_price=unit[1]
        max_lyap=float('-inf')
        x_isset=False
        for sim in range(sims):
            #within each Monte Carlo simulation
            #we generate a random Watts-Stroggatz Graph with parameter p,n=500
            #we generate a random seed node sample always with seed_p=0.05
            #we generate a random marketing mix
            #we generate a random distribution of the thresholds uniformly
            #we generate a random distribution of the budgets uniformly
            mkt_mix=random_mkt_mix(budget,iterations)
            prod,price,place,promo=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
            prod_inc='inc' if np.all(prod[1:] >= prod[:-1]) else 'dec'
            price_inc='inc' if np.all(price[1:] >= price[:-1]) else 'dec'
            place_inc='inc' if np.all(place[1:] >= place[:-1]) else 'dec'
            promo_inc='inc' if np.all(promo[1:] >= promo[:-1]) else 'dec'
            try:
                simulation=simulate_control(n,p,seed_set_p,mkt_mix,promo_budget,r_promo,r_price,last)
                diffusion=simulation.sum(axis=0)
                steadystate=diffusion[-last:].round(2)
            except ValueError:
                steadystate=np.zeros((last))
            lyap=nolds.lyap_e(steadystate)[0]
            if lyap !=float('inf') and lyap>max_lyap:
                max_lyap=lyap
                x=steadystate
                x_isset=True
            if lyap != float('inf') and max_lyap != float('-inf'): #check if lyapunov exponent is finite
                print(
                      "r_promo="+str(round(r_promo,2))+
                      ", r_price="+str(round(r_price,2))+
                      ", p="+str(round(p,4))+
                      ", sim="+str(sim+1)+
                      ", prod_qual="+prod_inc+
                      ", price_level="+price_inc+
                      ", dist_intensity="+place_inc+
                      ", promo_exp="+promo_inc+
                      ", run. max lyapunov="+str(round(np.max([lyap,max_lyap]),2))
                      )
            else:
                print(
                      "r_promo="+str(round(r_promo,2))+
                      ", r_price="+str(round(r_price,2))+
                      ", p="+str(round(p,4))+
                      ", sim="+str(sim+1)+
                      ", prod_qual="+prod_inc+
                      ", price_level="+price_inc+
                      ", dist_intensity="+place_inc+
                      ", promo_exp="+promo_inc+
                      ", inf. lyapunov"
                      )
        stop = time.time()
        duration = stop-start
        if x_isset:
            steadystates.append((unit[0],x))
            lyapunovs.append((unit[0],max_lyap))
            print(
                      "r_promo="+str(round(unit[0],2))+
                      "r_price="+str(round(unit[1],2))+
                      ", p="+str(round(unit[2],10))+
                      ", largest lyapunov="+str(round(max_lyap,2))+
                      ", time="+str(round(duration,0))+"s"
                      )
    return np.array(steadystates),np.array(lyapunovs)

prod_place_budget=4000
promo_budget=300
n=500          #graph network size
seed_set_p=0.05 #seed set size
iterations=500  #diffusion iterations
simulations=5  #Monte Carlo simulations within each DoE unit
last=50         #steadystate size
r_levels=3    #r factor levels
p_levels=1000     #p factor levels
r_promo = np.linspace(2.5, 4.0, r_levels)
r_price = np.linspace(2.5, 4.0, r_levels)
p = np.linspace(0.01, 0.05, p_levels)

#total of 400*20=8000 simulations

doe=build.full_fact({
     'r_promo':r_promo,
     'r_price':r_price,
     'p':p
     }).values.tolist()

steadystates,lyapunovs = simulate_system(doe,prod_place_budget,promo_budget,n,seed_set_p,iterations,simulations,last)

rows=steadystates.shape[0]
data=pd.DataFrame(np.concatenate([steadystates,lyapunovs[:,2].reshape(rows,1)],axis=1),columns=['p','steadystate','lyapunov'])
data=data.groupby(['p']).agg({'lyapunov':'max','steadystate':'first'})

#Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                               sharex=True)
lyapunov=data['lyapunov'].values
x=np.array([np.array(xi) for xi in data['steadystate'].values])
ax1.plot(p, x, ',k', alpha=.25)
ax1.set_xlim(2.5, 4)
ax1.set_title("Bifurcation diagram")

# We display the Lyapunov exponent.
# Horizontal line.
ax2.axhline(0, color='k', lw=.5, alpha=.3)
# Negative Lyapunov exponent.
ax2.plot(p[lyapunov < 0],
         lyapunov[lyapunov < 0] / last,
         '.k', alpha=.3, ms=.5)
# Positive Lyapunov exponent.
ax2.plot(p[lyapunov >= 0],
         lyapunov[lyapunov >= 0] / last,
         '.r', alpha=.3, ms=.5)
ax2.set_xlim(2.5, 4)
ax2.set_ylim(-0.005, 0.015)
ax2.set_title("Lyapunov exponent")
plt.tight_layout()

