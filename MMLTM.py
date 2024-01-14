import networkx as nx
import numpy as np
import nolds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.random import choice
from numpy.random import uniform
from doepy import build
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
        m=uniform(-10,10)
        b=uniform(-10,10)
        y=m*x+b
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
        #if t%10==0:
        #    print("t="+str(t+1)+" adoption="+str(previous_state)+" curr_ad_exp="+str(round(promo_control,2))+" cum_ad_exp="+str(round(promo_cost,2)))
        prod,price,place,promo=prods[t],prices[t],places[t],promo_control
        states=dict()
        for node in nodes:
            curr_state=[s for s in G.nodes()[node]['states']]
            curr_state.append(activation(node,G,prod,price,place,promo))
            states[node]=curr_state
        nx.set_node_attributes(G, states, "states")   
        previous_state=np.sum([st[-1] for st in nx.get_node_attributes(G,'states').values()])
        if previous_state==0 and t<periods-last:
            #print("t="+str(t+1)+" adoption="+str(previous_state)+" cum_ad_exp="+str(round(promo_cost,2)))
            raise ValueError
    state_matrix=np.array(list(nx.get_node_attributes(G,"states").values()))
    state_matrix=np.array(state_matrix)/n
    #print("t="+str(t+1)+" adoption="+str(previous_state)+" cum_ad_exp="+str(round(promo_cost,2)))
    return state_matrix

def simulate_system(doe,n,seed_set_p,iterations,sims,last):
    print("Starting simulation with "+str(len(doe))+" exp. units each with "+str(sims)+" Monte Carlo simulations: a total of "+str(len(doe)*sims)+" runs...")
    r_units=dict()
    p_units=dict()
    for unit in doe:
        steadystates,lyapunovs=[],[]
        ri=unit[0]
        p=unit[1]
        #max_lyap=float('-inf')
        for sim in range(sims):
            #within each Monte Carlo simulation
            #we generate a random Watts-Stroggatz Graph with parameter p,n=500
            #we generate a random seed node sample always with seed_p=0.05
            #we generate a random marketing mix
            #we generate a random distribution of the thresholds uniformly
            #we generate a random distribution of the budgets uniformly
            mkt_mix=random_mkt_mix(iterations)
            prod,price,place=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:]
            prod_inc='increasing' if np.all(prod[1:] >= prod[:-1]) else 'decreasing'
            price_inc='increasing' if np.all(price[1:] >= price[:-1]) else 'decreasing'
            place_inc='increasing' if np.all(place[1:] >= place[:-1]) else 'decreasing'
            try:
                simulation=simulate_control(n,p,seed_set_p,mkt_mix,ri,last)
                diffusion=simulation.sum(axis=0)
                steadystate=diffusion[-last:]
            except ValueError:
                steadystate=np.zeros((last))
            lyap=nolds.lyap_e(steadystate)[0]
            if lyap != float('inf'): #check if lyapunov exponent is finite
                steadystates.append(steadystate)
                lyapunovs.append(lyap)
                r_units[ri]=(steadystates,lyapunovs)
                p_units[p]=(steadystates,lyapunovs)
                print("r="+str(round(ri,2))+
                      ", p="+str(round(p,2))+
                      ", sim="+str(sim+1)+
                      ", prod_qual="+prod_inc+
                      ", price_level="+price_inc+
                      ", dist_intensity="+place_inc+
                      ", run. avg lyapunov="+str(round(np.mean(lyapunovs),2))
                      )
            else:
                print("r="+str(round(ri,2))+
                      ", p="+str(round(p,2))+
                      ", sim="+str(sim+1)+
                      ", prod_qual="+prod_inc+
                      ", price_level="+price_inc+
                      ", dist_intensity="+place_inc+
                      ", infinite lyapunov"
                      )
            
    r_steadystate_agg=[unit[0] for r,unit in zip(r_units.keys(),r_units.values())]
    r_lyapunov_agg=[np.mean(unit[1]) for r,unit in zip(r_units.keys(),r_units.values())]
    p_steadystate_agg=[unit[0] for p,unit in zip(p_units.keys(),p_units.values())]
    p_lyapunov_agg=[np.mean(unit[1]) for p,unit in zip(p_units.keys(),p_units.values())]
    return np.array(r_steadystate_agg),np.array(r_lyapunov_agg),np.array(p_steadystate_agg),np.array(p_lyapunov_agg)

n=500           #graph network size
seed_set_p=0.05 #seed set size
iterations=150  #diffusion iterations
simulations=20  #Monte Carlo simulations within each DoE unit
last=25         #steadystate size
r_levels=100    #r factor levels
p_levels=3      #p factor levels
r = np.linspace(2.5, 4.0, r_levels)
p = np.linspace(0.01, 0.3, p_levels)

#total of 400*20=8000 simulations

doe=build.full_fact({
     'r':r,
     'p':p
     }).values.tolist()

steadystates,lyapunovs,p_st,p_lyap = simulate_system(doe,n,seed_set_p,iterations,simulations,last)

#Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                               sharex=True)
lyapunov=lyapunovs
x=steadystates
ax1.plot(r, x, ',k', alpha=.25)
ax1.set_xlim(2.5, 4)
ax1.set_title("Bifurcation diagram")

# We display the Lyapunov exponent.
# Horizontal line.
ax2.axhline(0, color='k', lw=.5, alpha=.8)
# Negative Lyapunov exponent.
ax2.plot(r[lyapunov < 0],
         lyapunov[lyapunov < 0] / last,
         '.k', alpha=.8, ms=.5)
# Positive Lyapunov exponent.
ax2.plot(r[lyapunov >= 0],
         lyapunov[lyapunov >= 0] / last,
         '.r', alpha=.8, ms=.5)
ax2.set_xlim(2.5, 4)
ax2.set_ylim(-0.005, 0.015)
ax2.set_title("Lyapunov exponent")
plt.tight_layout()

