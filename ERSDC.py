import cmath
import copy
import sys
import time
import types
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import choice, uniform
from pyformlang.regular_expression import Regex
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- PYTHON COMPATIBILITY PATCH FOR NOLDS ---
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

sns.set_theme(style="whitegrid", palette="muted")

# =====================================================================
# 0. Topological Algebra & DFA Engine (Binary Alphabet: {'L', 'R'})
# =====================================================================

def symbolic_adjacency_matrix(alphabet, states, transition_function):
    n = len(states)
    state_to_idx = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((n, n), dtype=float)
    
    for src_state in states:
        src_idx = state_to_idx[src_state]
        for symbol in alphabet:
            try:
                next_states = transition_function.get_next_states(src_state, symbol)
                for dst_state in next_states:
                    if dst_state in state_to_idx:
                        dst_idx = state_to_idx[dst_state]
                        matrix[src_idx, dst_idx] += 1.0
            except (AttributeError, KeyError):
                pass
                
    return matrix

def compute_topological_entropy(words):
    if not words:
        return 0.0
        
    formatted_words = []
    for word in words:
        if isinstance(word, (list, tuple)):
            formatted_words.append(" ".join(word))
        else:
            formatted_words.append(" ".join(list(str(word))))
            
    try:
        regex_str = '|'.join(formatted_words)
        if not regex_str:
            return 0.0
            
        regex = Regex(regex_str)
        dfa = regex.to_epsilon_nfa().to_deterministic()
        alphabet = list(dfa.symbols)
        states = list(dfa.states)
        
        if not states or not alphabet:
            return 0.0
            
        transition_function = dfa._transition_function
        sam = symbolic_adjacency_matrix(alphabet, states, transition_function)
        
        if sam.size == 0:
            return 0.0
            
        eigenvals = np.linalg.eigvals(sam)
        max_eigenval = np.max(eigenvals)
        
        if max_eigenval > 0:
            return float(cmath.log(max_eigenval).real)
        else:
            return 0.0
    except Exception:
        return 0.0

# =====================================================================
# 1. Marketing Mix Threshold (MMT) Network Simulator
# =====================================================================

NUM_FLOOR = 1e-4

def logistic(x, r):
    return abs(1 - r * x)

def random_attributes(G, seed_set_p=0.05):
    nodes = G.nodes()
    n_nodes = G.number_of_nodes()
    thresholds = uniform(0, 1, n_nodes)
    budgets = uniform(0.4, 2.0, n_nodes)

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
        state_vectors.append(states[-1])  # latest activation state
    
    if not state_vectors:
        return 0.0
    return np.sum(state_vectors) * prod_qual

def influence(node, G, prod, place, promo, wom=True):
    base_advertising = 0.2
    if wom:
        return (ngm_sum(node, G, prod) + base_advertising) * place * promo
    else:
        return place * promo

def activation(node, G, prod, price, place, promo, wom=True):
    na = G.nodes()[node]
    threshold, budget = na['threshold'], na['budget']
    
    inf = influence(node, G, prod, place, promo, wom)
    if inf >= threshold and budget >= price:
        return 1
    else:
        return 0

def random_mkt_mix(budget):
    exp = float('inf')
    while exp > budget:
        exp = 0
        mkt_mix = []
        for i in range(4):
            mkt_mix.append(uniform(0, 1))
            if i != 1:  # Price does not consume marketing launch expenditure
                exp += mkt_mix[-1]
    return np.array(mkt_mix) + NUM_FLOOR

# =====================================================================
# 2. ERSDC Online Controller & Region Classifier (Updated Regimes A-EF)
# =====================================================================

def assign_region(r: float) -> str:
    if 0 <= r < 0.8:
        return 'A'
    elif 0.8 <= r < 1.1:
        return 'B'
    elif 1.1 <= r < 3.0:
        return 'CD'
    elif 3.0 <= r <= 4.0:
        return 'EF'
    else:
        return 'EF'

def build_region_naive_bayes(data_pairs):
    region_documents = {'A': [], 'B': [], 'CD': [], 'EF': []}
    
    for r, strings in data_pairs:
        region = assign_region(r)
        document = " ".join(strings)
        region_documents[region].append(document)
        
    X_texts, y_labels = [], []
    for region, docs in region_documents.items():
        if docs:
            X_texts.extend(docs)
            y_labels.extend([region] * len(docs))
            
    vectorizer = CountVectorizer(token_pattern=r"\b\w+\b")
    X_vectorized = vectorizer.fit_transform(X_texts)
    
    model = MultinomialNB()
    model.fit(X_vectorized, y_labels)
    model.vectorizer_ = vectorizer
    return model

def ersdc_online_step(
    trajectory_window,
    clf,
    w=4,
    delta_step=0.15,
    extinction_reset_r=0.4,
    entropy_critical=0.4,
):
    """
    Blind symbolic controller: decides an action based only on the adoption window
    and the offline-trained region classifier.
    
    Returns:
        reset_flag (bool): if True, the simulation should set r = value_or_delta.
        value_or_delta (float): absolute r if reset_flag else signed increment.
        predicted_region (str): classifier output (for diagnostics).
        h_top (float): estimated topological entropy.
        action (float): the signed increment (0, +delta, -delta) for logging.
    """
    current_A = float(np.mean(trajectory_window))
    latest_A = float(trajectory_window[-1])
    
    # 1. Extinction / collapse safeguard → reset to high-reward chaos
    if latest_A <= 0.0 or current_A == 0.0:
        return True, extinction_reset_r, "EXTINCT_RECOVERY", 0.0, 0.0

    # 2. Binary symbolisation: L / R based on adoption threshold 0.5
    symbols = ['R' if a > 0.5 else 'L' for a in trajectory_window]
    words = ["".join(symbols[i:i+w]) for i in range(len(symbols) - w + 1)]
    active_input = " ".join(words) if words else "".join(symbols)
    
    # 3. Topological entropy from the sofic shift
    h_top = compute_topological_entropy(words if words else symbols)

    # 4. Naive Bayes region prediction (A, B, CD, EF)
    predicted_region = "UNKNOWN"
    vectorizer = getattr(clf, 'vectorizer_', None)
    if vectorizer is not None:
        try:
            X_features = vectorizer.transform([active_input])
            predicted_region = clf.predict(X_features)[0]
        except Exception:
            pass

    # 5. Purely symbolic control logic (no r)
    if predicted_region == "EF":
        # Stagnation band: push left towards chaos
        action = -delta_step
    elif predicted_region in ("B", "CD"):
        # Transitional zones: also push left to reach A
        action = -delta_step
    elif predicted_region == "A":
        # Already in the high-reward window; avoid drifting too low (loss of chaos)
        if h_top < entropy_critical:
            action = delta_step   # increase to maintain chaotic activity
        else:
            action = 0.0
    else:
        # Unknown region – conservative hold
        action = 0.0

    return False, action, predicted_region, h_top, action

# =====================================================================
# 3. Closed-Loop MMT + ERSDC Network Simulation Engine
# =====================================================================

def simulate_mmt_closed_loop(
    G_orig,
    attributes_orig,
    r_init,
    horizon=80,
    window_len=8,
    control_mode='ERSDC',  # 'ERSDC', 'CONSTANT', or 'RANDOM'
    nb_model=None
):
    G = copy.deepcopy(G_orig)
    attributes = copy.deepcopy(attributes_orig)
    nx.set_node_attributes(G, attributes)
    
    n_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    mkt_mix = random_mkt_mix(budget=2.5)
    
    # Initialize Marketing Mix Controls
    adoption = np.sum([st['states'][-1] for st in attributes.values()])
    demand = (np.sum([attr['budget'] > mkt_mix[1] for attr in attributes.values()]) - adoption) / n_nodes
    availability = mkt_mix[2] * demand
    utility = mkt_mix[0] * mkt_mix[3] * adoption / n_nodes
    cost = (mkt_mix[0] + mkt_mix[2] + mkt_mix[3])
    revenue = (adoption / n_nodes) * mkt_mix[1]
    
    prod_control = revenue / (cost + 1e-6)
    price_control = demand
    place_control = availability
    promo_control = utility

    r_t = r_init
    adoption_history = [adoption / n_nodes]
    r_history = [r_t]
    profit_history = []
    
    window = [adoption / n_nodes]

    for t in range(horizon):
        # Update marketing mix controls via parameter r_t
        prod_control = 1 / (logistic(prod_control, r_t) + 1e-4)
        price_control = 1 / (abs(logistic(price_control, r_t) - price_control) + 1e-4)
        place_control = 1 / (logistic(place_control, r_t) + 1e-4)
        promo_control = 1 / (logistic(promo_control, r_t) + 1e-4)

        # Scale controls to valid numerical bounds
        prod = np.clip(prod_control, 0.01, 2.0)
        price = np.clip(price_control, 0.01, 1.0)
        place = np.clip(place_control, 0.01, 1.0)
        promo = np.clip(promo_control, 0.01, 2.0)

        # Execute Node Activations across Graph Topology
        states = dict()
        for node in nodes:
            curr_state = [s for s in G.nodes()[node]['states']]
            act = activation(node, G, prod, price, place, promo, wom=True)
            curr_state.append(act)
            states[node] = {'states': curr_state}
            
        nx.set_node_attributes(G, states)

        # Measure Macroscopic System States
        adoption = np.sum([st['states'][-1] for st in G.nodes.values()])
        adoption_frac = adoption / n_nodes
        adoption_history.append(adoption_frac)

        # Calculate Financial Performance
        current_cost = (prod + place + promo) * 1.0
        current_revenue = adoption_frac * price * 400.0
        profit = current_revenue - current_cost
        profit_history.append(profit)

        # Update Rolling Trajectory Window
        if len(window) >= window_len:
            window.pop(0)
        window.append(adoption_frac)

        # Apply Policy Control Strategy
        if control_mode == 'ERSDC':
            r_t, _, _, _ = ersdc_online_step(window, r_t, clf=nb_model)
        elif control_mode == 'RANDOM':
            r_t = uniform(0.0, 4.0)
        elif control_mode == 'CONSTANT':
            pass  # Keep r_t unchanged

        r_history.append(r_t)

    return np.array(adoption_history), np.array(profit_history), np.array(r_history)

# =====================================================================
# 4. Monte Carlo Benchmarking Framework (Updated Regimes A-EF)
# =====================================================================

def generate_synthetic_training_data(num_samples=200, window_len=16, w=4):
    data_pairs = []
    r_values = np.linspace(0.1, 3.98, num_samples)
    
    for r in r_values:
        x = uniform(0.1, 0.9)
        for _ in range(30):
            x = r * x * (1 - x)
            
        trajectory = []
        for _ in range(window_len):
            x = r * x * (1 - x)
            trajectory.append(x)
            
        symbols = ['R' if val > 0.5 else 'L' for val in trajectory]
        words = ["".join(symbols[i:i+w]) for i in range(len(symbols) - w + 1)]
        data_pairs.append((r, words))
        
    return data_pairs

def run_mmt_profit_benchmarking(nb_model, num_trials=100, horizon=80, n_nodes=200, p_edge=0.04):
    region_intervals = {
        'Region A (r in [0, 0.8])': (0.0, 0.8),
        'Region B (r in [0.8, 1.1])': (0.8, 1.1),
        'Region CD (r in [1.1, 3.0])': (1.1, 3.0),
        'Region EF (r in [3.0, 4.0])': (3.0, 4.0),
        'Random Policy (r in [0, 4])': (0.0, 4.0)
    }
    
    all_keys = list(region_intervals.keys()) + ['ERSDC Dynamic Control']
    results = {key: np.zeros((num_trials, horizon)) for key in all_keys}
    r_trajectories = []

    print(f"Executing Monte Carlo MMT Network Simulations ({num_trials} trials across updated regimes A-EF)...")
    
    for trial in range(num_trials):
        G = nx.gnp_random_graph(n_nodes, p_edge)
        attributes = random_attributes(G, seed_set_p=0.05)
        r_init = uniform(0.2, 2.5)

        # 1. ERSDC Closed-Loop Control
        _, profits_ersdc, r_hist = simulate_mmt_closed_loop(
            G, attributes, r_init, horizon=horizon, control_mode='ERSDC', nb_model=nb_model
        )
        results['ERSDC Dynamic Control'][trial, :] = profits_ersdc
        r_trajectories.append(r_hist)

        # 2. Interval-based Region Baselines
        for label, (r_min, r_max) in region_intervals.items():
            const_r = uniform(r_min, r_max)
            _, profits_const, _ = simulate_mmt_closed_loop(
                G, attributes, const_r, horizon=horizon, control_mode='CONSTANT', nb_model=nb_model
            )
            results[label][trial, :] = profits_const

    return results, np.array(r_trajectories)

# =====================================================================
# 5. Visualizations & Output Formatting
# =====================================================================

def plot_mmt_results(profit_data, r_trajectories):
    df_final = pd.DataFrame({label: np.sum(profits, axis=1) for label, profits in profit_data.items()})
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Parameter Trajectories under ERSDC ---
    num_plot = min(50, len(r_trajectories))
    for i in range(num_plot):
        axes[0].plot(r_trajectories[i], alpha=0.2, color='#55A868')
    
    axes[0].axhspan(0.0, 0.8, color='#C44E52', alpha=0.15, label='Region A (High-Value Chaos)')
    axes[0].set_title("ERSDC Parameter 'r' Closed-Loop Trajectories (Chaos Targeting)", fontsize=13)
    axes[0].set_xlabel("Simulation Horizon Step (t)", fontsize=11)
    axes[0].set_ylabel("Marketing Mix Control Parameter (r)", fontsize=11)
    axes[0].set_ylim(0.0, 4.2)
    axes[0].legend(loc="upper right")

    # --- Plot 2: Total Accumulated Profit Distribution ---
    order = df_final.mean().sort_values(ascending=False).index
    sns.boxplot(data=df_final, orient="h", ax=axes[1], palette="muted", order=order)
    axes[1].set_title("Total MMT Network Profit Comparison (Regimes A-EF)", fontsize=13)
    axes[1].set_xlabel("Accumulated Profit ($)", fontsize=11)
    
    plt.tight_layout()
    plt.show()
    return df_final

# =====================================================================
# Execution Entry Point
# =====================================================================

if __name__ == "__main__":
    print("Training Naive Bayes Region Classifier on Binary Symbolic Grammar...")
    training_data = generate_synthetic_training_data()
    nb_model = build_region_naive_bayes(training_data)

    start_time = time.time()
    profit_data, r_trajectories = run_mmt_profit_benchmarking(
        nb_model, num_trials=100, horizon=80, n_nodes=200, p_edge=0.04
    )
    duration = time.time() - start_time

    df_summary = plot_mmt_results(profit_data, r_trajectories)

    print("\n" + "="*70)
    print(f"CLOSED-LOOP MMT NETWORK CONTROL RESULTS ({round(duration, 1)}s execution time)")
    print("="*70)
    for col in df_summary.mean().sort_values(ascending=False).index:
        mean_p = df_summary[col].mean()
        std_p = df_summary[col].std()
        print(f"{col:36s} | Mean Profit: ${mean_p:9.2f} | Std: ${std_p:7.2f}")
    print("="*70)
