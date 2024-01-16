import networkx as nx
import numpy as np
import nolds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.random import choice
from numpy.random import uniform
from doepy import build
import pandas as pd

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

def random_mkt_mix(periods,granularity=1000):
    x= np.linspace(0,1,periods)
    mkt_mix=[]
    for i in range(4):
        m=uniform(-10,10)
        b=uniform(-10,10)
        y=m*x+b
        y = MinMaxScaler(feature_range=(0,1)).fit_transform(y.reshape(-1, 1)).reshape(-1)
        mkt_mix.append(y)
    mkt_mix=np.array(mkt_mix)+NUM_FLOOR
    return mkt_mix

def simulate_control(n,p,seed_set_p,mkt_mix,r_promo=None,r_price=None,last=25):
    periods=mkt_mix.shape[1]
    G=nx.gnp_random_graph(n,p)
    attributes=random_attributes(G,seed_set_p)
    nx.set_node_attributes(G,attributes)
    prods,prices,places,promos=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
    nodes=[n for n in G.nodes]
    promo_cost=0
    previous_state=np.sum([attr['states'][0] for attr in attributes.values()])
    for t in range(periods):
        prod,price,place,promo=prods[t],prices[t],places[t],promos[t]
        states=dict()
        for node in nodes:
            curr_state=[s for s in G.nodes()[node]['states']]
            curr_state.append(activation(node,G,prod,price,place,promo))
            states[node]=curr_state
        nx.set_node_attributes(G, states, "states")   
        previous_state=np.sum([st[-1] for st in nx.get_node_attributes(G,'states').values()])
    state_matrix=np.array(list(nx.get_node_attributes(G,"states").values()))
    state_matrix=np.array(state_matrix)/n
    return state_matrix

def simulate_system(doe,n,seed_set_p,iterations,sims,last):
    print("Starting simulation with "+str(len(doe))+" exp. units each with "+str(sims)+" Monte Carlo simulations: a total of "+str(len(doe)*sims)+" runs...")
    steadystates=[]
    lyapunovs=[]
    for unit in doe:
        #r_promo=unit[0]
        #r_price=unit[1]
        p=unit[0]
        max_lyap=float('-inf')
        x_isset=False
        for sim in range(sims):
            #within each Monte Carlo simulation
            #we generate a random Watts-Stroggatz Graph with parameter p,n=1000
            #we generate a random seed node sample always with seed_p=0.05
            #we generate a random marketing mix
            #we generate a random distribution of the thresholds uniformly
            #we generate a random distribution of the budgets uniformly
            mkt_mix=random_mkt_mix(iterations)
            prod,price,place,promo=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
            prod_inc='increasing' if np.all(prod[1:] >= prod[:-1]) else 'decreasing'
            price_inc='increasing' if np.all(price[1:] >= price[:-1]) else 'decreasing'
            place_inc='increasing' if np.all(place[1:] >= place[:-1]) else 'decreasing'
            promo_inc='increasing' if np.all(promo[1:] >= promo[:-1]) else 'decreasing'
            try:
                simulation=simulate_control(n,p,seed_set_p,mkt_mix,None,None,last)
                diffusion=simulation.sum(axis=0)
                steadystate=diffusion[-last:]
            except ValueError:
                steadystate=np.zeros((last))
            lyap=nolds.lyap_e(steadystate)[0]
            if lyap !=float('inf') and lyap>max_lyap:
                max_lyap=lyap
                x=steadystate
                x_isset=True
            if lyap != float('inf') and max_lyap != float('-inf'): #check if lyapunov exponent is finite
                print(
                      "p="+str(round(p,2))+
                      ", sim="+str(sim+1)+
                      ", prod_qual="+prod_inc+
                      ", price_level="+price_inc+
                      ", dist_intensity="+place_inc+
                      ", promo_exp="+promo_inc+
                      ", run. max lyapunov="+str(round(np.max([lyap,max_lyap]),2))
                      )
            else:
                print(
                      "p="+str(round(p,2))+
                      ", sim="+str(sim+1)+
                      ", prod_qual="+prod_inc+
                      ", price_level="+price_inc+
                      ", dist_intensity="+place_inc+
                      ", promo_exp="+promo_inc+
                      ", infinite lyapunov"
                      )
        if x_isset:
            steadystates.append((unit[0],unit[1],x))
            lyapunovs.append((unit[0],unit[1],max_lyap))
    return np.array(steadystates),np.array(lyapunovs)

n=1000          #graph network size
seed_set_p=0.05 #seed set size
iterations=1000  #diffusion iterations
simulations=20  #Monte Carlo simulations within each DoE unit
last=100         #steadystate size
p_levels=100      #p factor levels
p = np.linspace(0.01, 0.05, p_levels)

doe=build.full_fact({
     'p':p
     }).values.tolist()

steadystates,lyapunovs = simulate_system(doe,n,seed_set_p,iterations,simulations,last)

rows=steadystates.shape[0]
data=pd.DataFrame(np.concatenate([steadystates,lyapunovs[:,2].reshape(rows,1)],axis=1),columns=['r','p','steadystate','lyapunov'])
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

