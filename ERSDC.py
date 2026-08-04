import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice, uniform
import pandas as pd
import time
import cmath
import copy

# --- PYTHON 3.11+ COMPATIBILITY PATCH FOR NOLDS ---
import sys
import types
try:
    import nolds
except TypeError:
    mock_datasets = types.ModuleType('nolds.datasets')
    mock_datasets.brown72 = None
    mock_datasets.tent_map = None
    mock_datasets.logistic_map = None
    mock_datasets.fbm = None
    mock_datasets.fgn = None
    mock_datasets.qrandom = None
    mock_datasets.load_qrandom = lambda *a, **k: None
    mock_datasets.load_financial = lambda *a, **k: (None, None, None)
    mock_datasets.barabasi1991_fractal = None
    sys.modules['nolds.datasets'] = mock_datasets
    import nolds
# ------------------------------------------------

NUM_FLOOR = 1e-4
MAX_PERIOD = 4

def chunks(s, w):
    return [s[i:i + w] for i in range(0, len(s), w)]

def logistic(x, r):
    return abs(1 - r * x)

def random_attributes(G, seed_set_p=0.05):
    nodes = G.nodes()
    n_nodes = G.number_of_nodes()
    thresholds = uniform(0, 1, n_nodes)
    budgets = uniform(0, 1, n_nodes)
    states = np.array([0, 1])
    state_vectors = choice(states, size=n_nodes, p=[1 - seed_set_p, seed_set_p])
    
    attributes = dict()
    for node, th, bg, state in zip(nodes, thresholds, budgets, state_vectors):
        attributes[node] = {
            'threshold': th,
            'budget': bg,
            'states': [state]
        }
    return attributes

def ngm_sum(node, G, prod_qual):
    neighbors = G.neighbors(node)
    state_vectors = []
    for neighbor in neighbors:
        neighbor_attributes = G.nodes()[neighbor]
        states = neighbor_attributes['states']
        state_vectors.append(states)
    if not state_vectors:
        return 0.0
    state_matrix = np.array(state_vectors)
    return state_matrix.dot(prod_qual).sum()

def influence(node, G, prod, place, promo, wom):
    if wom:
        influence = ngm_sum(node, G, prod) * place * promo
    else:
        influence = place * promo
    return influence

def activation(node, G, prod, price, place, promo, wom=True):
    na = G.nodes()[node]
    threshold, budget = na['threshold'], na['budget']
    inf = influence(node, G, prod, place, promo, wom)
    if inf >= threshold and budget >= price:
        active = 1
    else:
        active = 0
    return active

def random_mkt_mix(budget, periods, granularity=1000):
    exp = float('inf')
    while(exp > budget):
        exp = 0
        mkt_mix = []
        for i in range(4):
            mkt_mix.append(uniform(0, 1))
            if i != 1: # price doesn't count
                exp += mkt_mix[-1]
    mkt_mix = np.array(mkt_mix) + NUM_FLOOR
    return mkt_mix

# --- Adapted Epoch Simulation using your exact original equations ---
def simulate_epoch_control(G, attributes, mkt_vars, r_val, epoch_length):
    n = G.number_of_nodes()
    prod_control, price_control, place_control, promo_control = mkt_vars
    nodes = list(G.nodes)
    
    epoch_adoptions = []
    symbols = []
    
    for t in range(epoch_length):
        prod_control = 1 / logistic(prod_control, r_val)
        price_control = 1 / abs(logistic(price_control, r_val) - price_control)
        place_control = 1 / logistic(place_control, r_val)
        promo_control = 1 / logistic(promo_control, r_val)
        
        adoption = np.sum([st[-1] for st in nx.get_node_attributes(G, 'states').values()])
        adoption_rate = adoption / n
        epoch_adoptions.append(adoption_rate)
        
        if adoption_rate > 0.5:
            symbol = 'R'
        elif adoption_rate > 0.0:
            symbol = 'L'
        else:
            symbol = 'C'
        symbols.append(symbol)
        
        prod, price, place, promo = prod_control, price_control, place_control, promo_control
        
        states = dict()
        for node in nodes:
            curr_state = list(G.nodes()[node]['states'])
            curr_state.append(activation(node, G, prod, price, place, promo))
            states[node] = curr_state
        nx.set_node_attributes(G, states, "states")   

        adoption = np.sum([st[-1] for st in nx.get_node_attributes(G, 'states').values()])
        demand = (np.sum([attr['budget'] > price_control for attr in attributes.values()]) - adoption) / n
        availability = place_control * demand
        utility = prod_control * promo_control * adoption / n
        cost = (prod_control + place_control + promo_control)
        revenue = (adoption / n) * price_control
        cost_to_revenue = revenue / cost if cost != 0 else 1.0
        
        prod_control = cost_to_revenue
        price_control = demand
        place_control = availability
        promo_control = utility
        
    updated_mkt_vars = (prod_control, price_control, place_control, promo_control)
    return G, updated_mkt_vars, epoch_adoptions

# --- Online ERSDC Control Framework ---

def run_ersdc_control_loop(n, p, seed_set_p, launch_budget, K_epochs=30, epoch_length=15, r_init=2.2):
    G = nx.gnp_random_graph(n, p)
    attributes = random_attributes(G, seed_set_p)
    nx.set_node_attributes(G, attributes)
    
    mkt_mix = random_mkt_mix(launch_budget, epoch_length)
    prices0 = mkt_mix[1]
    adoption0 = np.sum([st[-1] for st in nx.get_node_attributes(G, 'states').values()])
    demand0 = (np.sum([attr['budget'] > prices0 for attr in attributes.values()]) - adoption0) / n
    availability0 = mkt_mix[2] * demand0
    utility0 = mkt_mix[0] * mkt_mix[3] * adoption0 / n
    cost0 = (mkt_mix[0] + mkt_mix[2] + mkt_mix[3])
    revenue0 = (adoption0 / n) * prices0
    
    prod_control = revenue0 / cost0 if cost0 != 0 else 1.0
    price_control = demand0
    place_control = availability0
    promo_control = utility0
    
    mkt_vars = (prod_control, price_control, place_control, promo_control)
    r_current = r_init
    r_trajectory = [r_current]
    full_adoptions = []
    
    delta_macro = 0.4
    delta_local = 0.05
    
    for k in range(K_epochs):
        G, mkt_vars, epoch_adoptions = simulate_epoch_control(G, attributes, mkt_vars, r_current, epoch_length)
        full_adoptions.extend(epoch_adoptions)
        
        try:
            lyap = nolds.lyap_e(np.array(epoch_adoptions)).max()
        except:
            lyap = 0.0
            
        if r_current < 2.3:
            estimated_region = 'A'
        elif r_current < 2.5:
            estimated_region = 'B'
        elif r_current < 2.8:
            estimated_region = 'CD'
        elif r_current < 3.5:
            estimated_region = 'FE'
        else:
            estimated_region = 'G'
            
        mean_adoption = np.mean(epoch_adoptions)
        
        if mean_adoption < 0.02:
            action = delta_macro  # Extinction safeguard shift
        elif lyap > 0.05 or estimated_region in ['A', 'FE']:
            action = choice([-delta_local, delta_local])  # Chaos mitigation shift
        elif estimated_region in ['B', 'CD']:
            action = delta_macro  # Macro-jump toward Golden Region G
        else:
            action = 0.0  # Stable in Region G
            
        r_current = np.clip(r_current + action, 2.0, 4.0)
        r_trajectory.append(r_current)
        
    return r_trajectory, full_adoptions

def run_monte_carlo(num_runs=20):
    print(f"Running Monte Carlo evaluation with {num_runs} independent trajectories...")
    success_count = 0
    results = []
    
    for run in range(num_runs):
        r_init = uniform(2.0, 2.7)
        r_traj, adpt_traj = run_ersdc_control_loop(
            n=200, p=0.03, seed_set_p=0.05, launch_budget=0.5, 
            K_epochs=25, epoch_length=15, r_init=r_init
        )
        
        final_r = r_traj[-1]
        reached_golden = (final_r >= 3.5)
        survived = (np.mean(adpt_traj) > 0.0)
        
        if survived and reached_golden:
            success_count += 1
            
        results.append({
            'run': run,
            'survived': survived,
            'final_r': final_r,
            'reached_golden': reached_golden,
            'r_trajectory': r_traj
        })
        print(f"Run {run+1}/{num_runs} | Initial r: {r_init:.2f} | Final r: {final_r:.2f} | Survived: {survived} | Reached Golden (G): {reached_golden}")

    print(f"\nMonte Carlo Summary:")
    print(f"Total Runs: {num_runs}")
    print(f"Success Rate (Survived & Reached Region G): {(success_count / num_runs) * 100:.1f}%")
    return results

if __name__ == "__main__":
    mc_results = run_monte_carlo(num_runs=20)
    
    plt.figure(figsize=(10, 6))
    for res in mc_results:
        plt.plot(res['r_trajectory'], color='blue', alpha=0.3)
        
    plt.axhline(y=3.5, color='orange', linestyle='--', label='Golden Region Boundary (G)')
    plt.axhspan(3.5, 4.0, color='orange', alpha=0.1, label='Target Golden Zone')
    
    plt.xlim(0, 25)
    plt.ylim(2.0, 4.0)
    plt.xlabel("Control Epochs ($k$)")
    plt.ylabel("Bifurcation Parameter ($r_k$)")
    plt.title("Monte Carlo Control Trajectories using ERSDC")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()