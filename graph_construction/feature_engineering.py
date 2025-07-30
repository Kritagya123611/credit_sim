import networkx as nx
import numpy as np
import pandas as pd

def calculate_centrality_features(G):
    """Calculate graph centrality measures for fraud detection"""
    print("Calculating centrality measures...")
    
    # Basic centralities
    degree_cent = nx.degree_centrality(G)
    in_degree_cent = nx.in_degree_centrality(G)
    out_degree_cent = nx.out_degree_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    
    # For large graphs, sample for expensive centralities
    sample_size = min(500, G.number_of_nodes())
    betweenness_cent = nx.betweenness_centrality(G, k=sample_size, normalized=True)
    
    # Eigenvector centrality (may fail for disconnected graphs)
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=200)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        print("Eigenvector centrality failed - using degree centrality")
        eigenvector_cent = degree_cent
    
    # Add centrality features to nodes
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = degree_cent.get(node, 0)
        G.nodes[node]['in_degree_centrality'] = in_degree_cent.get(node, 0)
        G.nodes[node]['out_degree_centrality'] = out_degree_cent.get(node, 0)
        G.nodes[node]['pagerank'] = pagerank.get(node, 0)
        G.nodes[node]['betweenness_centrality'] = betweenness_cent.get(node, 0)
        G.nodes[node]['eigenvector_centrality'] = eigenvector_cent.get(node, 0)
    
    print("Centrality measures calculated and added to graph")
    return G

def detect_communities_and_risk_clusters(G):
    """Detect communities and analyze fraud concentration"""
    print("Detecting communities...")
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Greedy modularity community detection
    import networkx.algorithms.community as nx_comm
    communities = list(nx_comm.greedy_modularity_communities(G_undirected))
    
    print(f"Found {len(communities)} communities")
    
    # Analyze fraud in communities
    community_risk_analysis = {}
    for i, community in enumerate(communities):
        fraud_count = 0
        for node in community:
            fraud_type = G.nodes[node].get('fraud_type')
            if fraud_type and pd.notna(fraud_type) and fraud_type in ['ring', 'mule', 'bust_out']:
                fraud_count += 1
        
        total_nodes = len(community)
        risk_ratio = fraud_count / total_nodes if total_nodes > 0 else 0
        
        community_risk_analysis[i] = {
            'size': total_nodes,
            'fraud_count': fraud_count,
            'risk_ratio': risk_ratio
        }
        
        # Add community features to nodes
        for node in community:
            G.nodes[node]['community'] = i
            G.nodes[node]['community_risk_ratio'] = risk_ratio
            G.nodes[node]['community_fraud_count'] = fraud_count
    
    # Identify high-risk communities
    high_risk_communities = [
        comm_id for comm_id, stats in community_risk_analysis.items()
        if stats['risk_ratio'] > 0.1  # More than 10% fraud
    ]
    
    print(f"High-risk communities: {len(high_risk_communities)}")
    
    return G, communities, community_risk_analysis

def calculate_transaction_features(G, transactions_df):
    """Calculate transaction-based network features"""
    print("Calculating transaction-based features...")
    
    # Initialize node features
    for node in G.nodes():
        G.nodes[node].update({
            'total_sent': 0.0,
            'total_received': 0.0,
            'unique_recipients': 0,
            'unique_senders': 0,
            'transaction_frequency': 0,
            'velocity_score': 0.0
        })
    
    # Filter P2P transactions
    p2p_txns = transactions_df[
        (transactions_df['recipient_id'].notna()) & 
        (transactions_df['description'].str.contains('P2P|Transfer', na=False, case=False))
    ]
    
    # Aggregate by sender
    sent_stats = p2p_txns.groupby('agent_id').agg({
        'amount': ['sum', 'mean', 'count'],
        'recipient_id': 'nunique'
    }).round(2)
    sent_stats.columns = ['total_sent', 'avg_sent_amount', 'send_frequency', 'unique_recipients']
    
    # Aggregate by receiver
    received_stats = p2p_txns.groupby('recipient_id').agg({
        'amount': ['sum', 'mean', 'count'],
        'agent_id': 'nunique'
    }).round(2)
    received_stats.columns = ['total_received', 'avg_received_amount', 'receive_frequency', 'unique_senders']
    
    # Update node features
    for agent_id in G.nodes():
        if agent_id in sent_stats.index:
            stats = sent_stats.loc[agent_id]
            G.nodes[agent_id]['total_sent'] = stats['total_sent']
            G.nodes[agent_id]['avg_sent_amount'] = stats['avg_sent_amount']
            G.nodes[agent_id]['send_frequency'] = stats['send_frequency']
            G.nodes[agent_id]['unique_recipients'] = stats['unique_recipients']
        
        if agent_id in received_stats.index:
            stats = received_stats.loc[agent_id]
            G.nodes[agent_id]['total_received'] = stats['total_received']
            G.nodes[agent_id]['avg_received_amount'] = stats['avg_received_amount']
            G.nodes[agent_id]['receive_frequency'] = stats['receive_frequency']
            G.nodes[agent_id]['unique_senders'] = stats['unique_senders']
        
        # Calculate derived features
        total_sent = G.nodes[agent_id]['total_sent']
        total_received = G.nodes[agent_id]['total_received']
        
        # Transaction velocity (total volume)
        G.nodes[agent_id]['velocity_score'] = total_sent + total_received
        
        # Balance flow ratio
        total_volume = total_sent + total_received
        if total_volume > 0:
            G.nodes[agent_id]['balance_flow_ratio'] = total_sent / total_volume
        else:
            G.nodes[agent_id]['balance_flow_ratio'] = 0.5  # Neutral
    
    print("Transaction features calculated")
    return G

if __name__ == "__main__":
    from graph_builder import load_and_validate_data, create_transaction_graph
    
    print("Running feature engineering...")
    
    # Load data and create graph
    agents_df, transactions_df, agent_id_to_idx, idx_to_agent_id = load_and_validate_data()
    G = create_transaction_graph(agents_df, transactions_df)
    
    # Add centrality features
    G = calculate_centrality_features(G)
    
    # Add community features
    G, communities, community_analysis = detect_communities_and_risk_clusters(G)
    
    # Add transaction features
    G = calculate_transaction_features(G, transactions_df)
    
    print("Feature engineering complete!")
    
    # Save enhanced graph
    import pickle
    with open('../output/enhanced_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    print("Enhanced graph saved to '../output/enhanced_graph.pkl'")
    
    # Print summary
    sample_node = list(G.nodes())[0]
    sample_features = G.nodes[sample_node]
    print(f"\nSample node features: {len(sample_features)} total features")
    print("Enhanced graph ready for PyTorch Geometric conversion!")
