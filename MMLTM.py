import networkx as nx
import numpy as np

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

def NGM_sum(node, prod_qual):
    neighbors=G.neighbors(node)
    state_vectors=[]
    for neighbor in neighbors:
        state_vectors.append(neighbor['states'])
    state_matrix=np.array(state_vectors)
    return state_matrix.dot(prod_qual).sum()

def activation(node,prod,price):
    na=G.nodes()[node]
    threshold,budget,place,promo=na['threshold'],na['budget'],na['distribution'],na['advertising']
    
    influence=NGM_sum(node,prod)*place*promo
    if influence>threshold and budget>=price:
        active = 1
    else:
        active = 0
    return active
    
n=100
p=0.3

G=nx.gnp_random_graph(n,p)
attributes=random_attributes(G)
nx.set_node_attributes(G,attributes)

