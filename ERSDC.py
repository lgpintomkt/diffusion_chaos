import networkx as nx
import numpy as np
import nolds
import matplotlib.pyplot as plt
from numpy.random import choice, uniform
import pandas as pd
import time
import cmath
import copy

# --- Helper Functions ---

NUM_FLOOR = 1e-4

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
        return 1
    else:
        return 0

# --- ERSDC Online Control Environment ---

def run_ersdc_episode(n, p, seed_set_p, launch_budget, K_epochs, epoch_length, r_init, lambda_penalty=0.1):
    """
    Runs a single online control episode using the ERSDC algorithm over K_epochs.
    Each epoch lasts 'epoch_length' steps.
    """
    # 1. Initialize Graph and Attributes
    G = nx.gnp_random_graph(n, p)
    attributes = random_attributes(G, seed_set_p)
    nx.set_node_attributes(G, attributes)
    
    # Initial marketing mix and control state variables
    mkt_mix = [uniform(0, 1), uniform(0, 1), uniform(0, 1), uniform(0, 1)]
    prod_control, price_control, place_control, promo_control = mkt_mix[0]+NUM_FLOOR, mkt_mix[1]+NUM_FLOOR, mkt_mix[2]+NUM_FLOOR, mkt_mix[3]+NUM_FLOOR
    
    r_current = r_init
    r_trajectory = [r_current]
    adoption_trajectory = []
    entropy_trajectory = []
    
    delta_macro = 0.5   # Large jump (+Delta_L) to navigate across regions
    delta_local = 0.05  # Small local shift (+/- Delta_l) for chaos mitigation
    
    extinct = False

    for k in range(K_epochs):
        epoch_symbols = []
        epoch_adoptions = []
        
        # System evolution over epoch window 's' (epoch_length)
        for t in range(epoch_length):
            prod_control = 1 / logistic(prod_control, r_current)
            price_control = 1 / abs(logistic(price_control, r_current) - price_control)
            place_control = 1 / logistic(place_control, r_current)
            promo_control = 1 / logistic(promo_control, r_current)
            
            nodes = list(G.nodes)
            adoption = np.sum([st[-1] for st in nx.get_node_attributes(G, 'states').values()])
            adoption_rate = adoption / n
            epoch_adoptions.append(adoption_rate)
            
            if adoption_rate == 0.0:
                extinct = True
                break
                
            # Symbolic encoding for window observation
            if adoption_rate > 0.5:
                symbol = 'R'
            elif adoption_rate > 0.0:
                symbol = 'L'
            else:
                symbol = 'C'
            epoch_symbols.append(symbol)
            
            # Update network states
            states = dict()
            for node in nodes:
                curr_state = list(G.nodes()[node]['states'])
                curr_state.append(activation(node, G, prod_control, price_control, place_control, promo_control))
                states[node] = curr_state
            nx.set_node_attributes(G, states, "states")
            
        if extinct:
            break
            
        # Estimate Lyapunov / Entropy proxy over the epoch window
        try:
            lyap = nolds.lyap_e(np.array(epoch_adoptions)).max()
        except:
            lyap = 0.0
            
        # Simplified region classification based on current r and dynamical feedback
        # Regions: A (2.0-2.3), B (2.3-2.5), CD (2.5-2.8), FE (2.8-3.5), G (3.5-4.0)
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
        entropy_trajectory.append(lyap if lyap > 0 else 0)
        
        # --- ERSDC Policy Logic ---
        # 1. Extinction Safeguard
        if mean_adoption < 0.05:
            action = delta_macro  # Aggressive push upward to revive market
        # 2. Chaos Mitigation (Region A or FE, or positive Lyapunov exponent)
        elif lyap > 0.02 or estimated_region in ['A', 'FE']:
            # Apply local shift heuristic to find stability windows
            action = choice([-delta_local, delta_local])
        # 3. Target Golden Region (G) via Macro Jumps
        elif estimated_region in ['B', 'CD']:
            action = delta_macro
        else:
            # Already in Region G or stable zone, minimal adjustment
            action = 0.0
            
        # Update control parameter with domain clipping [2.0, 4.0]
        r_current = np.clip(r_current + action, 2.0, 4.0)
        r_trajectory.append(r_current)
        adoption_trajectory.extend(epoch_adoptions)
        
    return r_trajectory, adoption_trajectory, not extinct

# --- Monte Carlo Experiment Execution ---

def run_monte_carlo(num_runs=30, n=200, p=0.02, seed_set_p=0.05, launch_budget=0.5, K_epochs=40, epoch_length=15):
    print(f"Running Monte Carlo evaluation with {num_runs} independent trajectories...")
    success_count = 0
    results = []
    
    for run in range(num_runs):
        # Start each run from a random chaotic or low region (e.g., r_init between 2.0 and 3.0)
        r_init = uniform(2.0, 3.0)
        r_traj, adpt_traj, survived = run_ersdc_episode(n, p, seed_set_p, launch_budget, K_epochs, epoch_length, r_init)
        
        final_r = r_traj[-1]
        reached_golden = (final_r >= 3.5)
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

# --- Execution and Plotting ---

if __name__ == "__main__":
    # Parameters for the experiment
    NUM_RUNS = 20
    N_NODES = 200
    P_EDGE = 0.02
    SEED_P = 0.05
    BUDGET = 0.5
    EPOCHS = 35
    EPOCH_LEN = 15
    
    # Run Monte Carlo simulation
    mc_results = run_monte_carlo(num_runs=NUM_RUNS, n=N_NODES, p=P_EDGE, seed_set_p=SEED_P, launch_budget=BUDGET, K_epochs=EPOCHS, epoch_length=EPOCH_LEN)
    
    # Plotting Control Trajectories across Monte Carlo runs
    plt.figure(figsize=(10, 6))
    for res in mc_results:
        plt.plot(res['r_trajectory'], color='blue', alpha=0.3)
        
    # Highlight Region G threshold (r >= 3.5)
    plt.axhline(y=3.5, color='orange', linestyle='--', label='Golden Region Boundary (G)')
    plt.axhspan(3.5, 4.0, color='orange', alpha=0.1, label='Target Golden Zone')
    
    plt.xlim(0, EPOCHS)
    plt.ylim(2.0, 4.0)
    plt.xlabel("Control Epochs ($k$)")
    plt.ylabel("Bifurcation Parameter ($r_k$)")
    plt.title("Monte Carlo Control Trajectories using ERSDC")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()