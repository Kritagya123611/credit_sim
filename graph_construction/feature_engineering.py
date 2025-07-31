import networkx as nx
import numpy as np
import pandas as pd

def calculate_centrality_features(G):
    """Calculate graph centrality measures for credit risk assessment"""
    print("Calculating centrality measures...")
    
    # Basic centralities for credit risk influence analysis
    degree_cent = nx.degree_centrality(G)
    in_degree_cent = nx.in_degree_centrality(G)  # Incoming money flow influence
    out_degree_cent = nx.out_degree_centrality(G)  # Outgoing money flow influence
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)  # Overall network influence
    
    # For large graphs, sample for expensive centralities
    sample_size = min(500, G.number_of_nodes())
    betweenness_cent = nx.betweenness_centrality(G, k=sample_size, normalized=True)
    
    # Eigenvector centrality (may fail for disconnected graphs)
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=200)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        print("Eigenvector centrality failed - using degree centrality")
        eigenvector_cent = degree_cent
    
    # Add centrality features to nodes for credit risk assessment
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = degree_cent.get(node, 0)
        G.nodes[node]['in_degree_centrality'] = in_degree_cent.get(node, 0)
        G.nodes[node]['out_degree_centrality'] = out_degree_cent.get(node, 0)
        G.nodes[node]['pagerank'] = pagerank.get(node, 0)
        G.nodes[node]['betweenness_centrality'] = betweenness_cent.get(node, 0)
        G.nodes[node]['eigenvector_centrality'] = eigenvector_cent.get(node, 0)
    
    print("Centrality measures calculated and added to graph")
    return G

def calculate_personalized_pagerank(G):
    """Calculate personalized PageRank from high-risk borrowers for credit risk propagation"""
    print("Calculating personalized PageRank for credit risk propagation...")
    
    # ✅ NEW: Find high-risk seed nodes (risk_score > 0.7)
    high_risk_nodes = [n for n, d in G.nodes(data=True) 
                      if d.get('risk_score', 0) > 0.7]
    
    print(f"Found {len(high_risk_nodes)} high-risk seed nodes")
    
    # Calculate personalized PageRank from high-risk seeds
    if high_risk_nodes:
        # Limit to top 10 high-risk nodes for computational efficiency
        top_high_risk = high_risk_nodes[:min(10, len(high_risk_nodes))]
        
        for i, seed in enumerate(top_high_risk):
            try:
                ppr = nx.pagerank(G, personalization={seed: 1.0}, alpha=0.85, max_iter=100)
                for node in G.nodes():
                    G.nodes[node][f'ppr_high_risk_{i}'] = ppr.get(node, 0)
            except:
                print(f"Personalized PageRank failed for seed {seed}")
                # Add zero values if calculation fails
                for node in G.nodes():
                    G.nodes[node][f'ppr_high_risk_{i}'] = 0.0
    
    print("Personalized PageRank for credit risk propagation calculated")
    return G

def detect_communities_and_risk_clusters(G):
    """Detect communities and analyze credit risk concentration"""
    print("Detecting communities...")
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # ✅ UPDATED: Use Louvain algorithm if available, fallback to greedy modularity
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_undirected)
        communities = []
        community_dict = {}
        for node, comm_id in partition.items():
            if comm_id not in community_dict:
                community_dict[comm_id] = []
            community_dict[comm_id].append(node)
        communities = [set(nodes) for nodes in community_dict.values()]
        print(f"Used Louvain algorithm - found {len(communities)} communities")
    except ImportError:
        import networkx.algorithms.community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(G_undirected))
        print(f"Used greedy modularity - found {len(communities)} communities")
    
    # ✅ UPDATED: Analyze credit risk concentration in communities
    community_risk_analysis = {}
    for i, community in enumerate(communities):
        high_risk_count = 0
        total_risk_score = 0
        fraud_count = 0  # Keep for validation
        
        for node in community:
            # Primary: Credit risk analysis
            risk_score = G.nodes[node].get('risk_score', 0.5)
            total_risk_score += risk_score
            if risk_score > 0.7:  # High credit risk threshold
                high_risk_count += 1
            
            # Secondary: Fraud analysis (for validation)
            fraud_type = G.nodes[node].get('fraud_type')
            if fraud_type and pd.notna(fraud_type) and fraud_type in ['ring', 'mule', 'bust_out']:
                fraud_count += 1
        
        total_nodes = len(community)
        avg_risk_score = total_risk_score / total_nodes if total_nodes > 0 else 0.5
        high_risk_ratio = high_risk_count / total_nodes if total_nodes > 0 else 0
        fraud_ratio = fraud_count / total_nodes if total_nodes > 0 else 0
        
        community_risk_analysis[i] = {
            'size': total_nodes,
            'avg_risk_score': avg_risk_score,
            'high_risk_count': high_risk_count,
            'high_risk_ratio': high_risk_ratio,
            'fraud_count': fraud_count,  # Keep for validation
            'fraud_ratio': fraud_ratio   # Keep for validation
        }
        
        # Add community features to nodes
        for node in community:
            G.nodes[node]['community'] = i
            G.nodes[node]['community_avg_risk'] = avg_risk_score
            G.nodes[node]['community_high_risk_ratio'] = high_risk_ratio
            G.nodes[node]['community_fraud_ratio'] = fraud_ratio  # Keep for validation
    
    # ✅ UPDATED: Identify high credit risk communities
    high_risk_communities = [
        comm_id for comm_id, stats in community_risk_analysis.items()
        if stats['avg_risk_score'] > 0.6 or stats['high_risk_ratio'] > 0.2
    ]
    
    print(f"High credit risk communities: {len(high_risk_communities)}")
    
    return G, communities, community_risk_analysis

def calculate_transaction_features(G, transactions_df):
    """Calculate transaction-based network features for credit assessment"""
    print("Calculating transaction-based features...")
    
    # Initialize node features
    for node in G.nodes():
        G.nodes[node].update({
            'total_sent': 0.0,
            'total_received': 0.0,
            'unique_recipients': 0,
            'unique_senders': 0,
            'transaction_frequency': 0,
            'velocity_score': 0.0,
            'avg_sent_amount': 0.0,
            'avg_received_amount': 0.0,
            'send_frequency': 0,
            'receive_frequency': 0
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
        
        # ✅ UPDATED: Calculate derived features for credit risk
        total_sent = G.nodes[agent_id]['total_sent']
        total_received = G.nodes[agent_id]['total_received']
        
        # Transaction velocity (total volume) - indicator of financial activity
        G.nodes[agent_id]['velocity_score'] = total_sent + total_received
        
        # Balance flow ratio - spending vs receiving pattern
        total_volume = total_sent + total_received
        if total_volume > 0:
            G.nodes[agent_id]['balance_flow_ratio'] = total_sent / total_volume
        else:
            G.nodes[agent_id]['balance_flow_ratio'] = 0.5  # Neutral
        
        # ✅ NEW: Credit-specific transaction patterns
        # Transaction consistency (regular vs sporadic patterns)
        send_freq = G.nodes[agent_id]['send_frequency']
        receive_freq = G.nodes[agent_id]['receive_frequency']
        G.nodes[agent_id]['transaction_regularity'] = min(1.0, (send_freq + receive_freq) / 30)  # Monthly regularity
        
        # Network diversity (how many different people they transact with)
        unique_contacts = G.nodes[agent_id]['unique_recipients'] + G.nodes[agent_id]['unique_senders']
        G.nodes[agent_id]['network_diversity'] = min(1.0, unique_contacts / 20)  # Normalize to reasonable max
    
    print("Transaction features calculated")
    return G

if __name__ == "__main__":
    from graph_builder import load_and_validate_data, create_transaction_graph
    
    print("Running credit risk feature engineering...")
    
    # ✅ UPDATED: Load data with updated function signature
    agents_df, transactions_df, original_agents_df, agent_id_to_idx, idx_to_agent_id = load_and_validate_data()
    G = create_transaction_graph(agents_df, transactions_df, original_agents_df)
    
    # Add centrality features for credit risk
    G = calculate_centrality_features(G)
    
    # ✅ NEW: Add personalized PageRank for risk propagation
    G = calculate_personalized_pagerank(G)
    
    # Add community features with credit risk focus
    G, communities, community_analysis = detect_communities_and_risk_clusters(G)
    
    # Add transaction features for credit assessment
    G = calculate_transaction_features(G, transactions_df)
    
    print("Credit risk feature engineering complete!")
    
    # Save enhanced graph
    import pickle
    with open('../output/enhanced_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    print("Enhanced graph saved to '../output/enhanced_graph.pkl'")
    
    # ✅ UPDATED: Print credit risk summary
    sample_node = list(G.nodes())[0]
    sample_features = G.nodes[sample_node]
    print(f"\nSample node features: {len(sample_features)} total features")
    print(f"Sample risk score: {sample_features.get('risk_score', 'N/A')}")
    print(f"Sample community risk: {sample_features.get('community_avg_risk', 'N/A')}")
    print("Enhanced graph ready for credit risk GNN training!")
