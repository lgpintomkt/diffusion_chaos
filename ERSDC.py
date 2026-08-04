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
        return ngm_sum(node, G, prod) * place * promo
    else:
        return place * promo

def activation(node, G, prod, price, place, promo, wom=True):
    na = G.nodes()[node]
    threshold, budget = na['threshold'], na['budget']
    inf = influence(node, G, prod, place, promo, wom)
    if inf >= threshold and budget >= price:
        return 1
    return 0

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

def simulate_epoch_control(G, attributes, mkt_vars, r_val, epoch_length):
    n = G.number_of_nodes()
    prod_control, price_control, place_control, promo_control = mkt_vars
    nodes = list(G.nodes)
    
    epoch_adoptions = []
    
    for t in range(epoch_length):
        prod_control = 1 / logistic(prod_control, r_val)
        price_control = 1 / abs(logistic(price_control, r_val) - price_control)
        place_control = 1 / logistic(place_control, r_val)
        promo_control = 1 / logistic(promo_control, r_val)
        
        adoption = np.sum([st[-1] for st in nx.get_node_attributes(G, 'states').values()])
        adoption_rate = adoption / n
        epoch_adoptions.append(adoption_rate)
        
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

def build_regional_dictionaries(word_length=3, m_bins=3):
    print("Building offline regional symbolic dictionaries (Phase 1)...")
    regions = {'A': 2.1, 'B': 2.4, 'CD': 2.7, 'FE': 3.2, 'G': 3.6, 'FE_HIGH': 3.85}
    G_dummy = nx.gnp_random_graph(100, 0.05)
    attr_dummy = random_attributes(G_dummy)
    nx.set_node_attributes(G_dummy, attr_dummy)
    
    dictionaries = {}
    for reg, r_rep in regions.items():
        mkt_mix = random_mkt_mix(0.5, 10)
        mkt_vars = (mkt_mix[0], mkt_mix[1], mkt_mix[2], mkt_mix[3])
        _, _, adoptions = simulate_epoch_control(G_dummy, attr_dummy, mkt_vars, r_rep, epoch_length=150)
        
        bins = np.linspace(0, 1, m_bins + 1)
        symbols = np.digitize(adoptions, bins) - 1
        symbols = np.clip(symbols, 0, m_bins - 1)
        
        word_counts = {}
        total_words = 0
        for i in range(len(symbols) - word_length + 1):
            word = tuple(symbols[i:i+word_length])
            word_counts[word] = word_counts.get(word, 0) + 1
            total_words += 1
            
        prob_dist = {}
        vocab_size = m_bins ** word_length
        for w_idx in range(vocab_size):
            w_tuple = tuple(int(x) for x in np.base_repr(w_idx, base=m_bins, padding=word_length).zfill(word_length))
            prob_dist[w_tuple] = (word_counts.get(w_tuple, 0) + 1.0) / (total_words + vocab_size)
            
        dictionaries[reg] = prob_dist
    print("Offline dictionaries constructed successfully.\n")
    return dictionaries, m_bins

def infer_region_bayes(epoch_adoptions, dictionaries, word_length=3, m_bins=3):
    recent_adpt = epoch_adoptions[-word_length:] if len(epoch_adoptions) >= word_length else epoch_adoptions
    if len(recent_adpt) < word_length:
        recent_adpt = [0.0] * (word_length - len(recent_adpt)) + recent_adpt
        
    bins = np.linspace(0, 1, m_bins + 1)
    symbols = np.digitize(recent_adpt, bins) - 1
    symbols = tuple(np.clip(symbols, 0, m_bins - 1))
    
    regions = list(dictionaries.keys())
    p_prior = 1.0 / len(regions)
    posteriors = {}
    evidence = 0.0
    for reg in regions:
        p_word_given_r = dictionaries[reg].get(symbols, 1e-6)
        posterior_unnormalized = p_word_given_r * p_prior
        posteriors[reg] = posterior_unnormalized
        evidence += posterior_unnormalized
    for reg in regions:
        posteriors[reg] /= evidence
    return max(posteriors, key=posteriors.get), posteriors

def find_stabilization_step(r_trajectory, target_min=3.5, target_max=3.7):
    """Finds the first epoch index where r enters the target zone and stays there."""
    for i in range(len(r_trajectory)):
        if target_min <= r_trajectory[i] <= target_max:
            if all(target_min <= val <= target_max for val in r_trajectory[i:]):
                return i
    return None

def run_ersdc_control_loop(n, p, seed_set_p, launch_budget, dictionaries, word_length, m_bins, K_epochs=30, epoch_length=15, r_init=2.2, h_target=0.45, eta=1.0):
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
    
    delta_macro = 0.3
    delta_local = 0.04
    A_min = 0.02
    
    for k in range(K_epochs):
        G, mkt_vars, epoch_adoptions = simulate_epoch_control(G, attributes, mkt_vars, r_current, epoch_length)
        full_adoptions.extend(epoch_adoptions)
        
        mean_adoption = np.mean(epoch_adoptions)
        try:
            htop = float(nolds.sampen(np.array(epoch_adoptions)))
            if np.isnan(htop) or np.isinf(htop):
                htop = h_target
        except:
            htop = h_target
            
        estimated_region, _ = infer_region_bayes(epoch_adoptions, dictionaries, word_length, m_bins)
            
        if mean_adoption < A_min:
            action = delta_macro
        else:
            entropy_error = htop - h_target
            action = -eta * np.sign(entropy_error) * delta_local
            
        r_current = np.clip(r_current + action, 2.1, 3.7)
        r_trajectory.append(r_current)
        
    return r_trajectory, full_adoptions

def run_monte_carlo(num_runs=10):
    word_length = 3
    m_bins = 3
    dictionaries, m_bins = build_regional_dictionaries(word_length, m_bins)
    
    print(f"Running Monte Carlo evaluation with {num_runs} independent trajectories (Tracking Stabilization Steps)...")
    success_count = 0
    stabilization_steps = []
    results = []
    
    for run in range(num_runs):
        r_init = uniform(2.1, 2.6)
        r_traj, adpt_traj = run_ersdc_control_loop(
            n=200, p=0.03, seed_set_p=0.05, launch_budget=0.5, 
            dictionaries=dictionaries, word_length=word_length, m_bins=m_bins,
            K_epochs=30, epoch_length=15, r_init=r_init, h_target=0.45
        )
        
        final_r = r_traj[-1]
        survived = (np.mean(adpt_traj) > 0.0)
        stab_step = find_stabilization_step(r_traj)
        
        reached_golden = (stab_step is not None)
        if survived and reached_golden:
            success_count += 1
            stabilization_steps.append(stab_step)
            
        results.append({
            'run': run,
            'survived': survived,
            'final_r': final_r,
            'stabilization_step': stab_step,
            'r_trajectory': r_traj
        })
        
        step_str = f"Epoch {stab_step}" if stab_step is not None else "Did not stabilize"
        print(f"Run {run+1}/{num_runs} | Initial r: {r_init:.2f} | Final r: {final_r:.2f} | Stabilized at: {step_str}")

    avg_steps = np.mean(stabilization_steps) if stabilization_steps else float('nan')
    print(f"\nMonte Carlo Summary:")
    print(f"Total Runs: {num_runs}")
    print(f"Success Rate (Target Golden Zone): {(success_count / num_runs) * 100:.1f}%")
    print(f"Average Steps to Stabilization: {avg_steps:.1f} epochs")
    return results

if __name__ == "__main__":
    mc_results = run_monte_carlo(num_runs=10)
    
    plt.figure(figsize=(10, 6))
    for res in mc_results:
        plt.plot(res['r_trajectory'], color='blue', alpha=0.5)
        
    plt.axhline(y=3.5, color='orange', linestyle='--', label='Golden Region Lower Bound ($r = 3.5$)')
    plt.axhline(y=3.7, color='orange', linestyle=':', label='Golden Region Upper Bound ($r = 3.7$)')
    plt.axhline(y=3.6, color='red', linestyle='-', label='Interior Target ($r^* = 3.6$)')
    plt.axhspan(3.5, 3.7, color='orange', alpha=0.1, label='Target Golden Zone')
    
    plt.xlim(0, 30)
    plt.ylim(2.0, 3.9)
    plt.xlabel("Control Epochs ($k$)")
    plt.ylabel("Bifurcation Parameter ($r_k$)")
    plt.title("Blind ERSDC Trajectories with Stabilization Tracking")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
