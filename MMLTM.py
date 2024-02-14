import networkx as nx
import numpy as np
import nolds
import matplotlib.pyplot as plt
from numpy.random import choice
from numpy.random import uniform
from doepy import build
import pandas as pd
import time
from pyformlang.regular_expression import Regex
import cmath
import math
import copy

#See Chapter 3 - Sofic Shifts in
#Lind & Marcus "An Introduction to Symbolic Dynamics and Coding"
def symbolic_adjacency_matrix(alphabet,states,transition_function):
    sam=np.zeros([len(states),len(states)])
    for k,symbol in enumerate(alphabet):
        for i,state_from in enumerate(states):
            for j,state_to in enumerate(states):
                if state_to in transition_function(s_from=state_from,symb_by=symbol):
                    sam[i,j]+=1
                    sam[j,i]+=1
    return sam

NUM_FLOOR=1e-4

#def chunks(lst, n):
#    """Yield successive n-sized chunks from lst."""
#    for i in range(0, len(lst), n):
#        yield lst[i:i + n]

def chunks(s, w):
    return [s[i:i + w] for i in range(0, len(s), w)]

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

def simulate_control(iterations,G,attributes,seed_set_p,mkt_mix,r_prod=None,r_price=None,r_place=None,r_promo=None,last=25):
    periods=iterations#mkt_mix.shape[1]
    
    #prods,prices,places,promos=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
    prods,prices,places,promos=[mkt_mix[0]],[mkt_mix[1]],[mkt_mix[2]],[mkt_mix[3]]
    
    nodes=[n for n in G.nodes]
    promo_cost=0
    
    adoption=np.sum([st[-1] for st in nx.get_node_attributes(G,'states').values()])
    demand=(np.sum([attr['budget']>prices[0] for attr in attributes.values()])-adoption)/G.number_of_nodes()
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
    symbols=[]
    for t in range(periods):
        prod_control=1/logistic(prod_control,r_prod)
        price_control=1/abs(logistic(price_control,r_price)-price_control)
        place_control=1/logistic(place_control,r_place)
        promo_control=1/logistic(promo_control,r_promo)

        promo_cost+=promo_control
        
        if t>=periods-last:
            state=adoption/n
            if state==0:
                symbol='C'
            elif state>0.5:
                symbol='R'
            else:
                symbol='L'
            symbols.append(symbol)
        #if (t%1==0 or t==periods-1) and t>=periods-last:
        #     print("t="+str(t+1)+" symbol="+symbol+" adoption="+str(state)+", prod_qual="+str(round(prod_control,2))+", price="+str(round(price_control,2))+", dist_int="+str(round(place_control,2))+", ad_exp="+str(round(promo_control,2)))#+" curr_price="+str(round(price_control,2))+" curr_ad_exp="+str(round(promo_control,2))+" cum_ad_exp="+str(round(promo_cost,2)))
        prod,price,place,promo=prod_control,price_control,place_control,promo_control
        #prod,price,place,promo=prod_control,price_control,place_control,promo_control
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
        #if previous_state==0 and t<periods-last:
            #print("t="+str(t+1)+" adoption="+str(previous_state)+" cum_ad_exp="+str(round(promo_cost,2)))
            #raise ValueError
    #words=list(set([''.join(word) for word in [i for i in chunks(symbols,2)] if len(word)==2]))
    #words.sort()
    #print("list of 2-words obtained: "+str(words))
    #words=list(set([''.join(word) for word in [i for i in chunks(symbols,3)] if len(word)==3]))
    #words.sort()
    #print("list of 3-words obtained: "+str(words))
    words=[]
    max_period=4
    for period in range(1,max_period+1):
        words+=list(set([''.join(word) for word in [i for i in chunks(symbols,period)] if len(word)==period]))
    #size=3
    #words=list(set([''.join(word) for word in [i for i in chunks(symbols,period)] if len(word)==period]))
    words=list(set(words))+list("")
    words.sort()
    #print("words: "+str(words))
    for i,word in enumerate(words):
        words[i]=" ".join(word)
    regex = Regex('|'.join(words))
    
    dfa=regex.to_epsilon_nfa().to_deterministic()
    alphabet=list(dfa.symbols)
    states=list(dfa.states)
    transition_function=dfa._transition_function
    sam=symbolic_adjacency_matrix(alphabet,states,transition_function)
    eigenvals=np.linalg.eigvals(sam)
    max_eigenval=eigenvals.max()
    entropy=cmath.log(max_eigenval)
    #print("number of DFA states="+str(len(states)))
    #print(alphabet)
    #print('Entropy: '+str(round(entropy,2)))
    
    #print(str(len(words))+" words of len 4: "+str(words))
    state_matrix=np.array(list(nx.get_node_attributes(G,"states").values()))
    state_matrix=np.array(state_matrix)/n
    #print("t="+str(t+1)+" adoption="+str(previous_state)+" cum_ad_exp="+str(round(promo_cost,2)))
    return state_matrix,len(words),entropy,len(states)

def simulate_system(doe,budget,n,p,seed_set_p,iterations,sims,last):
    print("Starting simulation with "+str(len(doe))+" exp. units each with "+str(sims)+" Monte Carlo simulations: a total of "+str(len(doe)*sims)+" runs...")
    steadystates=[]
    lyapunovs=[]
    entropies=[]
    print("Generating random Watts-Strogatz graph...")
    G=nx.gnp_random_graph(n,p)
    nodes=G.nodes()
    print("Generating initial seed set, budget, and thresholds...")
    attributes=random_attributes(G,seed_set_p)
    nx.set_node_attributes(G,attributes)
    for unit in doe:
        start = time.time()
        #r_promo=unit[0]
        #r_price=unit[1]
        r_prod=unit[0]
        r_price=unit[0]
        r_place=unit[0]
        r_promo=unit[0]
        #p=unit[1]
        #r_prod=unit[0]
        #r_price=unit[1]
        #r_place=unit[2]
        #r_promo=unit[3]
        max_lyap=float('-inf')
        x_isset=False
        word_lens=[]
        entrs=[]
        state_comps=[]
        for sim in range(sims):
            #within each Monte Carlo simulation
            #we generate a random Watts-Stroggatz Graph with parameter p,n=500
            #we generate a random initial marketing mix within a given launch marketing budget
            #we generate a random distribution of the thresholds
            #we generate a random distribution of the budgets
            mkt_mix=random_mkt_mix(budget,iterations) #generate initial mkt mix
            #prod,price,place,promo=mkt_mix[0,:],mkt_mix[1,:],mkt_mix[2,:],mkt_mix[3,:]
            prod,price,place,promo=mkt_mix[0],mkt_mix[1],mkt_mix[2],mkt_mix[3]
            #prod_inc='inc' if np.all(prod[1:] >= prod[:-1]) else 'dec'
            #price_inc='inc' if np.all(price[1:] >= price[:-1]) else 'dec'
            #place_inc='inc' if np.all(place[1:] >= place[:-1]) else 'dec'
            #promo_inc='inc' if np.all(promo[1:] >= promo[:-1]) else 'dec'
            try:
                simulation,word_length,entropy,states=simulate_control(iterations,copy.deepcopy(G),copy.deepcopy(attributes),seed_set_p,mkt_mix,r_prod,r_price,r_place,r_promo,last)
                diffusion=simulation.sum(axis=0)
                steadystate=diffusion[-last:]
                word_lens.append(word_length)
                entrs.append(entropy.real)
                state_comps.append(states)
            except ValueError:
                steadystate=np.zeros((last))
            lyap=nolds.lyap_e(steadystate).max()
            #if not 0 in steadystate:
            #lyap=np.log(np.absolute(np.diff(steadystate[:-1)+NUM_FLOOR).sum()/len(steadystate)
            #else:
            #    lyap=float('inf')
            if lyap !=float('inf') and lyap>max_lyap:
                max_lyap=lyap
                x=steadystate
                x_isset=True
            # if lyap != float('inf') and max_lyap != float('-inf'): #check if lyapunov exponent is finite
            #     print(
            #           "r="+str(round(unit[0],2))+
            #           #", r_price="+str(round(unit[0],2))+
            #           #", r_place="+str(round(unit[1],2))+
            #           #", r_promo="+str(round(unit[1],2))+
            #           #", p="+str(round(p,4))+
            #           ", sim="+str(sim+1)+
            #           #", prod_qual="+prod_inc+
            #           #", price_level="+price_inc+
            #           #", dist_intensity="+place_inc+
            #           #", promo_exp="+promo_inc+
            #           ", run. max lyapunov="+str(round(np.max([lyap,max_lyap]),2))
            #           )
            # else:
            #     print(
            #           "r="+str(round(unit[0],2))+
            #           #", r_price="+str(round(unit[0],2))+
            #           #", r_place="+str(round(unit[1],2))+
            #           #", r_promo="+str(round(unit[1],2))+
            #           #", p="+str(round(p,4))+
            #           ", sim="+str(sim+1)+
            #           #", prod_qual="+prod_inc+
            #           #", price_level="+price_inc+
            #           #", dist_intensity="+place_inc+
            #           #", promo_exp="+promo_inc+
            #           ", inf. lyapunov"
            #           )
        stop = time.time()
        duration = stop-start
        if x_isset:
            steadystates.append((unit[0],x))
            lyapunovs.append((unit[0],max_lyap))
            entropies.append((unit[0],entropy.real))
            print(
                      "r={:.4f}".format(round(unit[0], 4))+
                      #"r="+str(round(unit[0],4))+
                      #"r_prod="+str(round(unit[0],2))+
                      #", r_price="+str(round(unit[0],2))+
                      #", r_place="+str(round(unit[0],2))+
                      #", r_promo="+str(round(unit[0],2))+
                      ", state complexity={:02}".format(max(state_comps))+
                      ", topological entropy={:.1f}".format(round(np.mean(entrs),1))+
                      #", avg_num_words="+str(round(np.mean(word_lens),1))+
                      ", largest lyapunov={:+.1f}".format(round(max_lyap,1))+
                      ", time="+str(round(duration,0))+"s"
                      )
            #print("")
        else:
            entropies.append((unit[0],entropy.real))
            print(
                      "r={:.4f}".format(round(unit[0], 4))+
                      
                      #"r="+str(round(unit[0],4))+
                      #"r_prod="+str(round(unit[0],2))+
                      #", r_price="+str(round(unit[0],2))+
                      #", r_place="+str(round(unit[0],2))+
                      #", r_promo="+str(round(unit[0],2))+
                      ", state complexity={:02}".format(max(state_comps))+
                      ", topological entropy={:.1f}".format(round(np.mean(entrs),1))+
                      #", avg_num_words="+str(round(np.mean(word_lens),1))+
                      ", largest lyapunov=inf"+
                      ", time="+str(round(duration,0))+"s"
                      )
            #print("")
    return np.array(steadystates),np.array(lyapunovs),np.array(entropies)

launch_mkt_budget=0.5
n=300          #graph network size
seed_set_p=0.05#seed set size
iterations=250  #diffusion iterations
simulations=1  #Monte Carlo simulations within each DoE unit
last=50         #steadystate size
r_levels=10000    #r factor levels
p_levels=1     #p factor levels
r = np.linspace(2, 4, r_levels)
r_prod  = np.linspace(2.5, 4.0, r_levels)
r_price = np.linspace(2.5, 4.0, r_levels)
r_place = np.linspace(2.5, 4.0, r_levels)
r_promo = np.linspace(2.5, 4.0, r_levels)
p = 0.01#np.linspace(0.01, 0.01, p_levels)

#     'r_prod':r_prod,
#     'r_price':r_price,
#     'r_place':r_place,
#     'r_promo':r_promo,

doe=build.full_fact({
    'r':r,
    #'p':p
     }).values.tolist()
#np.random.shuffle(doe)

steadystates,lyapunovs,entropies = simulate_system(doe,launch_mkt_budget,n,p,seed_set_p,iterations,simulations,last)
steadystates=steadystates[steadystates[:, 0].argsort()]
lyapunovs=lyapunovs[lyapunovs[:,0].argsort()]
rows=steadystates.shape[0]
data=pd.DataFrame(np.concatenate([steadystates,lyapunovs[:,1].reshape(rows,1)],axis=1),columns=['r','steadystate','lyapunov'])
data=data.groupby(['r']).agg({'lyapunov':'max','steadystate':'first'})

#Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                               sharex=True)
r=data.index
lyapunov=data['lyapunov'].values
x=np.array([np.array(xi) for xi in data['steadystate'].values])
ax1.plot(r, x, ',k', alpha=.55)
ax1.set_xlim(2, 4)
ax1.set_ylim(0, 1)
ax1.set_title("Bifurcation diagram")

# We display the Lyapunov exponent.
# Horizontal line.
ax2.axhline(0, color='k', lw=.5, alpha=.3)
# Negative Lyapunov exponent.
ax2.plot(r[lyapunov < 0],
         lyapunov[lyapunov < 0],
         '.k', alpha=1, ms=.5)
# Positive Lyapunov exponent.
ax2.plot(r[lyapunov >= 0],
         lyapunov[lyapunov >= 0],
         '.r', alpha=1, ms=.5)
ax2.set_xlim(2, 4)
ax2.set_ylim(-0.2, 0.2)
ax2.set_title("Lyapunov exponent")
plt.tight_layout()

