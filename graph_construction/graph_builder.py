import pandas as pd
import numpy as np
import networkx as nx
import os

def load_and_validate_data():
    """Load agents and transactions data with validation"""
    print("Loading data...")
    
    # Load your simulation output data
    agents_df = pd.read_csv('../output/gnn_features.csv')
    transactions_df = pd.read_csv('../output/transactions.csv')
    
    print(f"Loaded {len(agents_df)} agents with {agents_df.shape[1]} features")
    print(f"Loaded {len(transactions_df)} transactions")
    
    # Create agent ID mapping for graph indexing
    agent_ids = agents_df['agent_id'].tolist()
    agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    idx_to_agent_id = {idx: agent_id for agent_id, idx in agent_id_to_idx.items()}
    
    print(f"Created agent ID mappings for {len(agent_id_to_idx)} agents")
    
    return agents_df, transactions_df, agent_id_to_idx, idx_to_agent_id

def create_transaction_graph(agents_df, transactions_df):
    """Create directed graph from transaction data"""
    print("Building transaction graph...")
    
    # ✅ FIXED: Load fraud labels from original agents.csv
    try:
        original_agents = pd.read_csv('../output/agents.csv')[['agent_id', 'fraud_type']]
        print(f"Loaded fraud labels from original agents.csv")
    except FileNotFoundError:
        print("Warning: Original agents.csv not found - no fraud labels available")
        original_agents = pd.DataFrame(columns=['agent_id', 'fraud_type'])
    
    G = nx.DiGraph()
    
    # Add nodes with features from gnn_features.csv AND fraud labels from agents.csv
    for _, agent in agents_df.iterrows():
        node_attrs = agent.drop('agent_id').to_dict()
        
        # ✅ FIXED: Add fraud label from original data
        fraud_info = original_agents[original_agents['agent_id'] == agent['agent_id']]
        if not fraud_info.empty:
            fraud_type = fraud_info['fraud_type'].iloc[0]
            node_attrs['fraud_type'] = fraud_type
        else:
            node_attrs['fraud_type'] = None  # Legitimate agent
            
        G.add_node(agent['agent_id'], **node_attrs)
    
    # Rest of the function remains the same...
    p2p_conditions = (
        (transactions_df['recipient_id'].notna()) & 
        (transactions_df['description'].str.contains('P2P|Transfer', na=False, case=False))
    )
    p2p_txns = transactions_df[p2p_conditions].copy()
    
    print(f"Found {len(p2p_txns)} P2P transactions out of {len(transactions_df)} total")
    
    for _, txn in p2p_txns.iterrows():
        source = txn['agent_id']
        target = txn['recipient_id']
        
        if source in G.nodes() and target in G.nodes():
            if G.has_edge(source, target):
                G[source][target]['total_amount'] += txn['amount']
                G[source][target]['frequency'] += 1
                G[source][target]['channels'].add(txn['channel'])
            else:
                G.add_edge(source, target, 
                          total_amount=txn['amount'],
                          frequency=1,
                          channels={txn['channel']},
                          first_date=txn['date'])
    
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


if __name__ == "__main__":
    # Test the graph construction
    agents_df, transactions_df, agent_id_to_idx, idx_to_agent_id = load_and_validate_data()
    transaction_graph = create_transaction_graph(agents_df, transactions_df)
    
    print(f"\n=== Graph Summary ===")
    print(f"Nodes: {transaction_graph.number_of_nodes()}")
    print(f"Edges: {transaction_graph.number_of_edges()}")
    print(f"Graph density: {nx.density(transaction_graph):.6f}")
