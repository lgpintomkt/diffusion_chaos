import networkx as nx
import numpy as np
import nolds
from numpy.random import randint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings("ignore")

def logistic(x,r):
    return r*x*(1-x)

def random_attributes(G,seed_set_p=0.05):
    from numpy.random import choice
    from numpy.random import uniform
    
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

def activation(node,G,prod,price,place,promo):
    na=G.nodes()[node]
    threshold,budget=na['threshold'],na['budget']
    
    influence=ngm_sum(node,G,prod)*place*promo
    if influence>threshold and budget>=price:
        active = 1
    else:
        active = 0
    return active

def random_mkt_mix(periods,granularity=1000):
    x= np.linspace(0,1,periods)
    mkt_mix=[]
    for i in range(4):    
        y = randint(0,100)*np.sin(randint(0,100)*x)+randint(0,100)*np.cos(randint(0,100)*x)
        y = MinMaxScaler(feature_range=(0,1)).fit_transform(y.reshape(-1, 1)).reshape(-1)
        mkt_mix.append(y)
    mkt_mix=np.array(mkt_mix)
    return mkt_mix

def simulate_adv_control(n,p,seed_set_p,mkt_mix,r=None):
    periods=mkt_mix.shape[1]
    G=nx.gnp_random_graph(n,p)
    attributes=random_attributes(G)
    nx.set_node_attributes(G,attributes)
    prods,prices,places,_=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
    
    nodes=[n for n in G.nodes]
    previous_state=np.sum([G.nodes()[node]['states'] for node in G.nodes()])
    for t in range(periods):
        promo_control=logistic(previous_state/n,r)
        prod,price,place,promo=prods[t],prices[t],places[t],promo_control
        states=dict()
        for node in nodes:
            curr_state=[s for s in G.nodes()[node]['states']]
            curr_state.append(activation(node,G,prod,price,place,promo))
            states[node]=curr_state
        nx.set_node_attributes(G, states, "states")   
        state_matrix=np.array(list(nx.get_node_attributes(G,"states").values()))
        previous_state=np.sum([G.nodes()[node]['states'] for node in nodes])
    state_matrix=np.array(state_matrix)
    return state_matrix

def simulate_system(r,n,p,seed_set_p,iterations,sims,last):
    transients,lyapunovs=[],[]
    for ri in r:
        mkt_mix=random_mkt_mix(iterations)
        max_lyap=float("-inf")
        for sim in range(sims):
            print("sim="+str(sim+1)+", r="+str(ri))
            simulation=simulate_adv_control(n,p,seed_set_p,mkt_mix,ri)/n
            diffusion=simulation.sum(axis=1)
            lyap=nolds.lyap_e(diffusion)
            transient=simulation[-last]
            if lyap > max_lyap:
                max_lyap=lyap
                x=transient
        transients.append(x)
        lyapunovs.append(max_lyap)
    return np.array(transients),np.array(lyapunovs)

n=100
p=0.3
seed_set_p=0.05
iterations=300
simulations=1
last=100
r = np.linspace(2.5, 4.0, 100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

x,lyapunov = simulate_system(r,n,p,seed_set_p,iterations,simulations,last)
ax1.plot(r, x, ',k', alpha=.25)
ax1.set_xlim(2.5, 4)
ax1.set_title("Bifurcation diagram")

# We display the Lyapunov exponent.
# Horizontal line.
ax2.axhline(0, color='k', lw=.5, alpha=.5)
# Negative Lyapunov exponent.
ax2.plot(r[lyapunov < 0],
         lyapunov[lyapunov < 0] / iterations,
         '.k', alpha=.5, ms=.5)
# Positive Lyapunov exponent.
ax2.plot(r[lyapunov >= 0],
         lyapunov[lyapunov >= 0] / iterations,
         '.r', alpha=.5, ms=.5)
ax2.set_xlim(2.5, 4)
ax2.set_ylim(-2, 1)
ax2.set_title("Lyapunov exponent")
plt.tight_layout()
