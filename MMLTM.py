import networkx as nx
import numpy as np
import nolds
import matplotlib.pyplot as plt
from numpy.random import choice
from numpy.random import uniform
from doepy import build
import pandas as pd
import time

NUM_FLOOR=1e-4

def logistic(x,r):
    return abs(1-r*x)

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

def influence(node,G,prod,place,promo,wom):
    if wom:
        influence=ngm_sum(node,G,prod)*place*promo
    else:
        influence=place*promo
    return influence

def activation(node,G,prod,price,place,promo,wom=True):
    na=G.nodes()[node]
    threshold,budget=na['threshold'],na['budget']
    
    inf=influence(node,G,prod,place,promo,wom)
    if inf>=threshold and budget>=price:
        active = 1
    else:
        active = 0
    return active

def random_mkt_mix(budget,periods,granularity=1000):
    exp=float('inf')
    while(exp>budget):
        exp=0
        mkt_mix=[]
        for i in range(4):
            mkt_mix.append(uniform(0,1))
            if i!=1: #price doesn't count
                exp+=mkt_mix[-1]
    mkt_mix=np.array(mkt_mix)+NUM_FLOOR
    return mkt_mix

def simulate_control(iterations,n,p,seed_set_p,mkt_mix,r_prod=None,r_price=None,r_place=None,r_promo=None,last=25):
    periods=iterations#mkt_mix.shape[1]
    G=nx.gnp_random_graph(n,p)
    nodes=G.nodes()
    attributes=random_attributes(G,seed_set_p)
    nx.set_node_attributes(G,attributes)
    #prods,prices,places,promos=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
    prods,prices,places,promos=[mkt_mix[0]],[mkt_mix[1]],[mkt_mix[2]],[mkt_mix[3]]
    
    nodes=[n for n in G.nodes]
    promo_cost=0
    
    adoption=np.sum([st[-1] for st in nx.get_node_attributes(G,'states').values()])
    demand=(np.sum([attr['budget']>prices[0] for attr in attributes.values()])-adoption)/n
    availability=places[0]*demand
    utility=prods[0]*promos[0]*adoption/n
    cost=(prods[0]+places[0]+promos[0])/3
    revenue=(adoption/n)*prices[0]
    
    cost_to_revenue=revenue/cost
    prod_control=cost_to_revenue
    price_control=demand
    place_control=availability
    promo_control=utility
    #print("t="+str(0)+" adoption="+str(round(adoption/n,2))+", curr_prod_qual="+str(round(prods[0],2))+", curr_price="+str(round(prices[0],2))+", curr_dist_int="+str(round(places[0],2))+", curr_ad_exp="+str(round(promos[0],2)))#+" curr_price="+str(round(price_control,2))+" curr_ad_exp="+str(round(promo_control,2))+" cum_ad_exp="+str(round(promo_cost,2)))
    for t in range(periods):
        prod_control=1/logistic(prod_control,r_prod)
        price_control=1/abs(logistic(price_control,r_price)-price_control)
        place_control=1/logistic(place_control,r_place)
        promo_control=1/logistic(promo_control,r_promo)

        promo_cost+=promo_control
        if t%10==0 or t==periods-1:
            print("t="+str(t+1)+" adoption="+str(round(adoption/n,2))+", prod_qual="+str(round(prod_control,2))+", price="+str(round(price_control,2))+", dist_int="+str(round(place_control,2))+", ad_exp="+str(round(promo_control,2)))#+" curr_price="+str(round(price_control,2))+" curr_ad_exp="+str(round(promo_control,2))+" cum_ad_exp="+str(round(promo_cost,2)))
        prod,price,place,promo=prod_control,price_control,place_control,promo_control

        states=dict()
        for node in nodes:
            curr_state=[s for s in G.nodes()[node]['states']]
            curr_state.append(activation(node,G,prod,price,place,promo))
            states[node]=curr_state
        nx.set_node_attributes(G, states, "states")   

        adoption=np.sum([st[-1] for st in nx.get_node_attributes(G,'states').values()])
        demand=(np.sum([attr['budget']>price_control for attr in attributes.values()])-adoption)/n
        availability=place_control*demand
        utility=prod_control*promo_control*adoption/n
        cost=(prod_control+place_control+promo_control)
        revenue=(adoption/n)*price_control
        cost_to_revenue=revenue/cost
    state_matrix=np.array(list(nx.get_node_attributes(G,"states").values()))
    state_matrix=np.array(state_matrix)/n
    return state_matrix

def simulate_system(doe,budget,n,seed_set_p,iterations,sims,last):
    print("Starting simulation with "+str(len(doe))+" exp. units each with "+str(sims)+" Monte Carlo simulations: a total of "+str(len(doe)*sims)+" runs...")
    steadystates=[]
    lyapunovs=[]
    for unit in doe:
        start = time.time()
        #r_promo=unit[0]
        #r_price=unit[1]
        r_prod=r_price=unit[0]
        r_place=r_promo=unit[1]
        p=unit[2]
        #r_prod=unit[0]
        #r_price=unit[1]
        #r_place=unit[2]
        #r_promo=unit[3]
        max_lyap=float('-inf')
        x_isset=False
        for sim in range(sims):
            #within each Monte Carlo simulation
            #we generate a random Watts-Stroggatz Graph with parameter p,n=500
            #we generate a random seed node sample always with seed_p=0.05
            #we generate a random initial marketing mix
            #we generate a random distribution of the thresholds uniformly
            #we generate a random distribution of the budgets uniformly
            mkt_mix=random_mkt_mix(budget,iterations) #generate initial mkt mix
            #prod,price,place,promo=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
            prod,price,place,promo=mkt_mix[0],mkt_mix[1],mkt_mix[2],mkt_mix[3]
            #prod_inc='inc' if np.all(prod[1:] >= prod[:-1]) else 'dec'
            #price_inc='inc' if np.all(price[1:] >= price[:-1]) else 'dec'
            #place_inc='inc' if np.all(place[1:] >= place[:-1]) else 'dec'
            #promo_inc='inc' if np.all(promo[1:] >= promo[:-1]) else 'dec'
            try:
                simulation=simulate_control(iterations,n,p,seed_set_p,mkt_mix,r_prod,r_price,r_place,r_promo,last)
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
                      "r_prod="+str(round(unit[0],2))+
                      ", r_price="+str(round(unit[0],2))+
                      ", r_place="+str(round(unit[1],2))+
                      ", r_promo="+str(round(unit[1],2))+
                      ", p="+str(round(p,4))+
                      ", sim="+str(sim+1)+
                      ", run. max lyapunov="+str(round(np.max([lyap,max_lyap]),2))
                      )
            else:
                print(
                      "r_prod="+str(round(unit[0],2))+
                      ", r_price="+str(round(unit[0],2))+
                      ", r_place="+str(round(unit[1],2))+
                      ", r_promo="+str(round(unit[1],2))+
                      ", p="+str(round(p,4))+
                      ", sim="+str(sim+1)+
                      ", inf. lyapunov"
                      )
        stop = time.time()
        duration = stop-start
        if x_isset:
            steadystates.append((unit[0],x))
            lyapunovs.append((unit[0],max_lyap))
            print(
                      "r_prod="+str(round(unit[0],2))+
                      ", r_price="+str(round(unit[0],2))+
                      ", r_place="+str(round(unit[0],2))+
                      ", r_promo="+str(round(unit[0],2))+
                      ", p="+str(round(unit[1],10))+
                      ", largest lyapunov="+str(round(max_lyap,2))+
                      ", time="+str(round(duration,0))+"s"
                      )
    return np.array(steadystates),np.array(lyapunovs)

launch_mkt_budget=0.5
n=500          #graph network size
seed_set_p=0.05#seed set size
iterations=150  #diffusion iterations
simulations=5  #Monte Carlo simulations within each DoE unit
last=25         #steadystate size
r_levels=100    #r factor levels
p_levels=1     #p factor levels
r = np.linspace(2.5, 4, r_levels)
r_prod = np.linspace(4, 2.5, r_levels)
r_price = np.linspace(2.5, 4.0, r_levels)
r_place = np.linspace(2.5, 4.0, r_levels)
r_promo = np.linspace(2.5, 4.0, r_levels)
p = np.linspace(0.03, 0.03, p_levels)

#     'r_prod':r_prod,
#     'r_price':r_price,
#     'r_place':r_place,
#     'r_promo':r_promo,

doe=build.full_fact({
     'r':r,
     'r2':r_prod,
     'p':p
     }).values.tolist()
np.random.shuffle(doe)

steadystates,lyapunovs = simulate_system(doe,launch_mkt_budget,n,seed_set_p,iterations,simulations,last)

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

