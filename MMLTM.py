import networkx as nx
import numpy as np
import nolds
from numpy.random import randint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.random import choice
from numpy.random import uniform
import warnings
#warnings.filterwarnings("ignore")

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
        m=uniform(-1,1)
        b=uniform(-1,1)
        y=m*x+b
        #y = np.ones(periods)*uniform(0,1)
        #y = np.uniform(0,1)randint(0,100)*np.sin(randint(0,100)*x)+randint(0,100)*np.cos(randint(0,100)*x)
        y = MinMaxScaler(feature_range=(0,1)).fit_transform(y.reshape(-1, 1)).reshape(-1)
        mkt_mix.append(y)
    mkt_mix=np.array(mkt_mix)
    return mkt_mix

def simulate_control(n,p,seed_set_p,mkt_mix,r=None,last=25):
    periods=mkt_mix.shape[1]
    G=nx.gnp_random_graph(n,p)
    attributes=random_attributes(G)
    nx.set_node_attributes(G,attributes)
    prods,prices,places,_=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
    
    nodes=[n for n in G.nodes]
    promo_cost=0
    previous_state=np.sum([attr['states'][0] for attr in attributes.values()])
    for t in range(periods):
        #prod_control=logistic(previous_state/n,r)
        #price_control=-logistic(previous_state/n,r)
        #place_control=logistic(previous_state/n,r)
        promo_control=logistic(previous_state/n,r)
        promo_cost+=promo_control
        if t%10==0:
            print("t="+str(t+1)+" adoption="+str(previous_state)+" curr_ad_exp="+str(round(promo_control,2))+" cum_ad_exp="+str(round(promo_cost,2)))
        prod,price,place,promo=prods[t],prices[t],places[t],promo_control
        states=dict()
        for node in nodes:
            curr_state=[s for s in G.nodes()[node]['states']]
            curr_state.append(activation(node,G,prod,price,place,promo))
            states[node]=curr_state
        nx.set_node_attributes(G, states, "states")   
        previous_state=np.sum([st[-1] for st in nx.get_node_attributes(G,'states').values()])
        if previous_state==0 and t<periods-last:
            print("t="+str(t+1)+" adoption="+str(previous_state)+" cum_ad_exp="+str(round(promo_cost,2)))
            raise ValueError
    state_matrix=np.array(list(nx.get_node_attributes(G,"states").values()))
    state_matrix=np.array(state_matrix)/n
    print("t="+str(t+1)+" adoption="+str(previous_state)+" cum_ad_exp="+str(round(promo_cost,2)))
    return state_matrix

def simulate_system(r,n,seed_set_p,iterations,sims,last):
    transients,lyapunovs=[],[]
    for ri in r:
        max_lyap=float("-inf")
        for sim in range(sims):
            mkt_mix=random_mkt_mix(iterations)
            prod,price,place=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:]
            prod_inc='increasing' if np.all(prod[1:] >= prod[:-1]) else 'decreasing'
            price_inc='increasing' if np.all(price[1:] >= price[:-1]) else 'decreasing'
            place_inc='increasing' if np.all(place[1:] >= place[:-1]) else 'decreasing'
            p=uniform(1e-1,0.5)
            print("r="+str(ri)+
                  ", sim="+str(sim+1)+
                  ", p="+str(round(p,2))+
                  ", prod_qual="+prod_inc+
                  ", price_level="+price_inc+
                  ", dist_intensity="+place_inc
                  )
            try:
                simulation=simulate_control(n,p,seed_set_p,mkt_mix,ri,last)
                diffusion=simulation.sum(axis=0)
                transient=diffusion[-last:]
            except ValueError:
                transient=np.zeros((last))
            lyap=nolds.lyap_e(transient)[0]
            if lyap > max_lyap:
                max_lyap=lyap
                x=transient
        transients.append(x)
        lyapunovs.append(max_lyap)
    return np.array(transients),np.array(lyapunovs)

n=500
seed_set_p=0.05
iterations=150
simulations=20 #mkt mix simulations
last=25
r_sims=100 #r parameter simulations
r = np.linspace(2.5, 4.0, r_sims)

transients,lyapunovs = simulate_system(r,n,seed_set_p,iterations,simulations,last)

#Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                               sharex=True)

lyapunov = np.zeros(r_sims)
for i in range(last):
    x=transients[:,i]
    lyapunov += np.log(abs(r - 2 * r * x))
    # We display the bifurcation diagram.
ax1.plot(r, x, ',k', alpha=.25)
ax1.set_xlim(2.5, 4)
ax1.set_title("Bifurcation diagram")

# We display the Lyapunov exponent.
# Horizontal line.
ax2.axhline(0, color='k', lw=.5, alpha=.5)
# Negative Lyapunov exponent.
ax2.plot(r[lyapunov < 0],
         lyapunov[lyapunov < 0] / last,
         '.k', alpha=.5, ms=.5)
# Positive Lyapunov exponent.
ax2.plot(r[lyapunov >= 0],
         lyapunov[lyapunov >= 0] / last,
         '.r', alpha=.5, ms=.5)
ax2.set_xlim(2.5, 4)
ax2.set_ylim(-2, 1)
ax2.set_title("Lyapunov exponent")
plt.tight_layout()
