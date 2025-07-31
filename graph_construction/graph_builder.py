import pandas as pd
import numpy as np
import networkx as nx
import os

def load_and_validate_data():
    """Load agents and transactions data with validation"""
    print("Loading data...")
    
    # ✅ UPDATED: Flexible path handling for different execution contexts
    paths_to_try = [
        ('../output/gnn_features.csv', '../output/transactions.csv', '../output/agents.csv'),
        ('output/gnn_features.csv', 'output/transactions.csv', 'output/agents.csv'),
        ('./output/gnn_features.csv', './output/transactions.csv', './output/agents.csv')
    ]
    
    agents_df = None
    transactions_df = None
    original_agents_df = None
    
    for agents_path, transactions_path, original_path in paths_to_try:
        try:
            if os.path.exists(agents_path) and os.path.exists(transactions_path):
                agents_df = pd.read_csv(agents_path)
                transactions_df = pd.read_csv(transactions_path)
                # Try to load original agents for additional labels
                if os.path.exists(original_path):
                    original_agents_df = pd.read_csv(original_path)
                break
        except Exception as e:
            continue
    
    if agents_df is None:
        raise FileNotFoundError("Could not find gnn_features.csv and transactions.csv in any expected location")
    
    print(f"Loaded {len(agents_df)} agents with {agents_df.shape[1]} features")
    print(f"Loaded {len(transactions_df)} transactions")
    
    # Create agent ID mapping for graph indexing
    agent_ids = agents_df['agent_id'].tolist()
    agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    idx_to_agent_id = {idx: agent_id for agent_id, idx in agent_id_to_idx.items()}
    
    print(f"Created agent ID mappings for {len(agent_id_to_idx)} agents")
    
    return agents_df, transactions_df, original_agents_df, agent_id_to_idx, idx_to_agent_id

def create_transaction_graph(agents_df, transactions_df, original_agents_df=None):
    """Create directed graph from transaction data focused on credit risk"""
    print("Building transaction graph for credit risk assessment...")
    
    G = nx.DiGraph()
    
    # ✅ UPDATED: Add nodes with credit risk features (not fraud-focused)
    for _, agent in agents_df.iterrows():
        node_attrs = agent.drop('agent_id').to_dict()
        
        # ✅ UPDATED: Add credit risk labels from original data if available
        if original_agents_df is not None:
            original_info = original_agents_df[original_agents_df['agent_id'] == agent['agent_id']]
            if not original_info.empty:
                # Primary credit risk targets
                node_attrs['risk_score'] = original_info['risk_score'].iloc[0]
                node_attrs['risk_profile'] = original_info['risk_profile'].iloc[0]
                
                # Keep fraud labels for validation (but not primary target)
                if 'fraud_type' in original_info.columns:
                    node_attrs['fraud_type'] = original_info['fraud_type'].iloc[0]
            else:
                # Default credit risk values for missing agents
                node_attrs['risk_score'] = 0.5  # Neutral risk
                node_attrs['risk_profile'] = 'Medium'
                node_attrs['fraud_type'] = None
        else:
            print("Warning: No original agents data - using default credit risk values")
            node_attrs['risk_score'] = 0.5
            node_attrs['risk_profile'] = 'Medium' 
            node_attrs['fraud_type'] = None
            
        G.add_node(agent['agent_id'], **node_attrs)
    
    # ✅ UPDATED: Filter P2P transactions for credit network analysis
    p2p_conditions = (
        (transactions_df['recipient_id'].notna()) & 
        (transactions_df['description'].str.contains('P2P|Transfer', na=False, case=False))
    )
    p2p_txns = transactions_df[p2p_conditions].copy()
    
    print(f"Found {len(p2p_txns)} P2P transactions out of {len(transactions_df)} total")
    
    # ✅ UPDATED: Build edges with enhanced transaction features for credit analysis
    for _, txn in p2p_txns.iterrows():
        source = txn['agent_id']
        target = txn['recipient_id']
        
        # Only add edges between existing nodes
        if source in G.nodes() and target in G.nodes():
            
            # Aggregate multiple transactions between same agents
            if G.has_edge(source, target):
                G[source][target]['total_amount'] += txn['amount']
                G[source][target]['frequency'] += 1
                G[source][target]['channels'].add(txn['channel'])
                # ✅ NEW: Track amount patterns for credit risk
                G[source][target]['amounts'].append(txn['amount'])
            else:
                G.add_edge(source, target, 
                          total_amount=txn['amount'],
                          frequency=1,
                          channels={txn['channel']},
                          first_date=txn['date'],
                          amounts=[txn['amount']])  # ✅ NEW: Amount history
    
    # ✅ NEW: Calculate derived edge features for credit analysis
    for source, target, data in G.edges(data=True):
        amounts = data['amounts']
        data['avg_amount'] = np.mean(amounts)
        data['amount_std'] = np.std(amounts) if len(amounts) > 1 else 0
        data['amount_consistency'] = 1 - (data['amount_std'] / data['avg_amount']) if data['avg_amount'] > 0 else 0
        
        # ✅ NEW: Credit risk indicators from transaction patterns
        data['high_value_ratio'] = sum(1 for amt in amounts if amt > 50000) / len(amounts)
        data['regularity_score'] = min(1.0, data['frequency'] / 30)  # Normalize to monthly frequency
    
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # ✅ NEW: Print credit risk distribution
    if original_agents_df is not None:
        risk_scores = [G.nodes[node].get('risk_score', 0.5) for node in G.nodes()]
        print(f"Credit risk score distribution - Mean: {np.mean(risk_scores):.3f}, Std: {np.std(risk_scores):.3f}")
        
        risk_profiles = [G.nodes[node].get('risk_profile', 'Unknown') for node in G.nodes()]
        profile_counts = pd.Series(risk_profiles).value_counts()
        print(f"Risk profile distribution: {dict(profile_counts)}")
    
    return G

if __name__ == "__main__":
    # ✅ UPDATED: Test the credit risk graph construction
    agents_df, transactions_df, original_agents_df, agent_id_to_idx, idx_to_agent_id = load_and_validate_data()
    transaction_graph = create_transaction_graph(agents_df, transactions_df, original_agents_df)
    
    print(f"\n=== Credit Risk Graph Summary ===")
    print(f"Nodes: {transaction_graph.number_of_nodes()}")
    print(f"Edges: {transaction_graph.number_of_edges()}")
    print(f"Graph density: {nx.density(transaction_graph):.6f}")
    
    # ✅ NEW: Sample node inspection for credit risk features
    sample_node = list(transaction_graph.nodes())[0]
    sample_features = transaction_graph.nodes[sample_node]
    print(f"Sample node features: {len(sample_features)} total")
    print(f"Credit risk score: {sample_features.get('risk_score', 'N/A')}")
    print(f"Risk profile: {sample_features.get('risk_profile', 'N/A')}")
