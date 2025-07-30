import networkx as nx
import numpy as np
import pandas as pd

def analyze_graph_structure(G):
    """Analyze basic graph properties for fraud detection"""
    print("=== Detailed Graph Analysis ===")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.6f}")
    
    # Check connectivity
    if nx.is_strongly_connected(G):
        print("Graph is strongly connected")
    else:
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        print(f"Largest strongly connected component: {len(largest_scc)} nodes")
    
    # Degree distribution analysis
    degrees = [d for n, d in G.degree()]
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Max degree: {max(degrees)}")
    print(f"Min degree: {min(degrees)}")
    
    # ✅ FIXED: Robust fraud detection
    fraud_nodes = []
    fraud_types_found = {}
    
    for node, data in G.nodes(data=True):
        fraud_val = data.get('fraud_type')
        
        # Check for valid fraud types (not None, NaN, or empty)
        if fraud_val and pd.notna(fraud_val) and fraud_val != '':
            if fraud_val in ['ring', 'bust_out', 'mule']:
                fraud_nodes.append(node)
                fraud_types_found[fraud_val] = fraud_types_found.get(fraud_val, 0) + 1
    
    print(f"Fraud nodes in graph: {len(fraud_nodes)}")
    if fraud_types_found:
        print("Fraud type breakdown:", fraud_types_found)
    
    # Debug: Show sample of fraud_type values
    sample_fraud_values = [G.nodes[node].get('fraud_type') for node in list(G.nodes())[:10]]
    print(f"Sample fraud_type values: {sample_fraud_values}")
    
    # Analyze edge weights (transaction amounts)
    edge_weights = [d['total_amount'] for u, v, d in G.edges(data=True)]
    print(f"Average transaction amount: ₹{np.mean(edge_weights):,.2f}")
    print(f"Max transaction amount: ₹{max(edge_weights):,.2f}")
    
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(), 
        'fraud_nodes': len(fraud_nodes),
        'avg_degree': np.mean(degrees),
        'avg_transaction': np.mean(edge_weights)
    }



if __name__ == "__main__":
    # Import and run analysis
    from graph_builder import load_and_validate_data, create_transaction_graph
    
    agents_df, transactions_df, agent_id_to_idx, idx_to_agent_id = load_and_validate_data()
    transaction_graph = create_transaction_graph(agents_df, transactions_df)
    
    # Run detailed analysis
    graph_stats = analyze_graph_structure(transaction_graph)
    print(f"\nGraph ready for GNN training with {graph_stats['fraud_nodes']} fraud nodes!")
