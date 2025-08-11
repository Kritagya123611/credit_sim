import networkx as nx
import numpy as np
import pandas as pd


def calculate_centrality_features(G):
    """Calculate graph centrality measures for credit risk assessment"""
    print("📊 Calculating centrality features...")
    
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
        print("   ⚠️ Eigenvector centrality failed - using degree centrality fallback")
        eigenvector_cent = degree_cent
    
    # Add centrality features to nodes for credit risk assessment
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = degree_cent.get(node, 0.0)
        G.nodes[node]['in_degree_centrality'] = in_degree_cent.get(node, 0.0)
        G.nodes[node]['out_degree_centrality'] = out_degree_cent.get(node, 0.0)
        G.nodes[node]['pagerank_score'] = pagerank.get(node, 0.0)
        G.nodes[node]['betweenness_centrality'] = betweenness_cent.get(node, 0.0)
        G.nodes[node]['eigenvector_centrality'] = eigenvector_cent.get(node, 0.0)
    
    print("   ✅ Centrality features calculated and added to graph")
    return G


def calculate_personalized_pagerank(G):
    """✅ ENHANCED: Calculate personalized PageRank from high-risk behavioral patterns"""
    print("🔍 Calculating personalized PageRank for behavioral risk propagation...")
    
    # ✅ PHASE 2: Use composite behavioral risk instead of stored risk_score
    high_risk_nodes = []
    
    for node, data in G.nodes(data=True):
        # Calculate composite risk from Phase 2 behavioral features
        financial_stability = data.get('financial_stability_score', 0.5)
        behavioral_consistency = data.get('behavioral_consistency', 0.5)
        device_stability = data.get('device_stability_score', 0.5)
        
        # Composite risk (higher score = higher risk)
        composite_risk = (
            (1 - financial_stability) * 0.4 +
            (1 - behavioral_consistency) * 0.3 +
            (1 - device_stability) * 0.3
        )
        
        if composite_risk > 0.7:  # High behavioral risk threshold
            high_risk_nodes.append(node)
    
    print(f"   📈 Found {len(high_risk_nodes)} high behavioral risk nodes")
    
    if not high_risk_nodes:
        print("   ⚠️ No high behavioral risk nodes found - skipping personalized PageRank")
        return G
    
    # Limit to top 10 for computational efficiency
    top_high_risk = high_risk_nodes[:10]
    
    for i, seed in enumerate(top_high_risk):
        try:
            ppr = nx.pagerank(G, personalization={seed: 1.0}, alpha=0.85, max_iter=100)
            for node in G.nodes():
                G.nodes[node][f'ppr_behavioral_risk_{i}'] = ppr.get(node, 0.0)
        except Exception as e:
            print(f"   ❌ Personalized PageRank failed for seed {seed}: {e}")
            for node in G.nodes():
                G.nodes[node][f'ppr_behavioral_risk_{i}'] = 0.0
    
    print("   ✅ Personalized PageRank for behavioral risk propagation calculated")
    return G


def detect_communities_and_risk_clusters(G):
    """✅ ENHANCED: Detect communities and analyze behavioral risk clusters"""
    print("🏘️ Detecting communities and analyzing behavioral risk concentration...")
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Try Louvain first, fallback to greedy modularity
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
        print(f"   📊 Louvain algorithm found {len(communities)} communities")
    except ImportError:
        import networkx.algorithms.community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(G_undirected))
        print(f"   📊 Greedy modularity found {len(communities)} communities")
    
    # ✅ ENHANCED: Analyze behavioral risk concentration in communities
    community_risk_analysis = {}
    for i, community in enumerate(communities):
        behavioral_risk_scores = []
        financial_instability_count = 0
        evaluation_fraud_count = 0  # Only for evaluation
        
        for node in community:
            node_data = G.nodes[node]
            
            # ✅ BEHAVIORAL RISK: Calculate from Phase 2 features
            financial_stability = node_data.get('financial_stability_score', 0.5)
            behavioral_consistency = node_data.get('behavioral_consistency', 0.5)
            device_stability = node_data.get('device_stability_score', 0.5)
            
            behavioral_risk = (
                (1 - financial_stability) * 0.4 +
                (1 - behavioral_consistency) * 0.3 +
                (1 - device_stability) * 0.3
            )
            behavioral_risk_scores.append(behavioral_risk)
            
            if financial_stability < 0.3:  # Financial instability indicator
                financial_instability_count += 1
            
            # ✅ EVALUATION ONLY: Count fraud for validation (not used in training)
            evaluation_fraud = node_data.get('evaluation_fraud_label', 0)
            if evaluation_fraud == 1:
                evaluation_fraud_count += 1
        
        total_nodes = len(community)
        avg_behavioral_risk = np.mean(behavioral_risk_scores) if total_nodes > 0 else 0.5
        high_behavioral_risk_count = sum(1 for score in behavioral_risk_scores if score > 0.7)
        high_behavioral_risk_ratio = high_behavioral_risk_count / total_nodes if total_nodes > 0 else 0
        financial_instability_ratio = financial_instability_count / total_nodes if total_nodes > 0 else 0
        evaluation_fraud_ratio = evaluation_fraud_count / total_nodes if total_nodes > 0 else 0
        
        community_risk_analysis[i] = {
            'size': total_nodes,
            'avg_behavioral_risk_score': avg_behavioral_risk,
            'high_behavioral_risk_count': high_behavioral_risk_count,
            'high_behavioral_risk_ratio': high_behavioral_risk_ratio,
            'financial_instability_ratio': financial_instability_ratio,
            'evaluation_fraud_count': evaluation_fraud_count,  # For evaluation only
            'evaluation_fraud_ratio': evaluation_fraud_ratio   # For evaluation only
        }
        
        # Add community features to nodes
        for node in community:
            G.nodes[node]['community_id'] = i
            G.nodes[node]['community_avg_behavioral_risk'] = avg_behavioral_risk
            G.nodes[node]['community_high_risk_ratio'] = high_behavioral_risk_ratio
            G.nodes[node]['community_financial_instability_ratio'] = financial_instability_ratio
            # Evaluation only features
            G.nodes[node]['community_evaluation_fraud_ratio'] = evaluation_fraud_ratio
    
    # ✅ IDENTIFY: High behavioral risk communities
    high_risk_communities = [
        comm_id for comm_id, stats in community_risk_analysis.items()
        if stats['avg_behavioral_risk_score'] > 0.6 or stats['high_behavioral_risk_ratio'] > 0.2
    ]
    
    print(f"   🚨 High behavioral risk communities identified: {len(high_risk_communities)}")
    
    return G, communities, community_risk_analysis


def calculate_transaction_features(G, transactions_df):
    """✅ ENHANCED: Calculate transaction-based network features for behavioral analysis"""
    print("💳 Calculating enhanced transaction features...")

    # Initialize enhanced node features
    for node in G.nodes():
        G.nodes[node].update({
            'total_sent_volume': 0.0,
            'total_received_volume': 0.0,
            'unique_recipients_count': 0,
            'unique_senders_count': 0,
            'p2p_transaction_frequency': 0,
            'transaction_velocity_score': 0.0,
            'avg_sent_amount': 0.0,
            'avg_received_amount': 0.0,
            'send_frequency': 0,
            'receive_frequency': 0,
            'transaction_amount_variance': 0.0,
            'high_value_transaction_ratio': 0.0
        })

    # ✅ FLEXIBLE: Handle missing 'description' column gracefully
    p2p_conditions = (
        (transactions_df['recipient_id'].notna()) &
        (transactions_df['recipient_id'] != '') &
        (transactions_df['recipient_id'] != transactions_df['agent_id'])  # Avoid self-loops
    )

    # Check for P2P indicator columns
    if 'description' in transactions_df.columns:
        print("   📝 Using 'description' column for P2P filtering")
        p2p_conditions &= transactions_df['description'].str.contains('P2P|Transfer|UPI', na=False, case=False)
    elif 'txn_type' in transactions_df.columns:
        print("   📝 Using 'txn_type' column for P2P filtering")
        p2p_conditions &= transactions_df['txn_type'].str.contains('P2P|Transfer', na=False, case=False)
    elif 'channel' in transactions_df.columns:
        print("   📝 Using 'channel' column for P2P filtering")
        p2p_conditions &= transactions_df['channel'].isin(['UPI', 'IMPS', 'NEFT', 'RTGS', 'P2P'])
    else:
        print("   ⚠️ No description/txn_type/channel columns found - using recipient_id only for P2P detection")

    p2p_txns = transactions_df[p2p_conditions].copy()
    print(f"   📊 Processing {len(p2p_txns)} P2P transactions")

    if len(p2p_txns) == 0:
        print("   ⚠️ No P2P transactions found - skipping transaction feature calculation")
        return G

    # Aggregate sender statistics with variance calculations
    try:
        sent_agg = p2p_txns.groupby('agent_id').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'recipient_id': 'nunique'
        })
        sent_agg.columns = ['total_sent', 'avg_sent', 'send_frequency', 'sent_amount_std', 'unique_recipients']
        sent_agg['sent_amount_std'] = sent_agg['sent_amount_std'].fillna(0)
        high_value_threshold = 50000   
        high_value_sent = p2p_txns[p2p_txns['amount'] > high_value_threshold].groupby('agent_id').size()
        sent_agg['high_value_sent_count'] = high_value_sent.reindex(sent_agg.index, fill_value=0)
        sent_agg['high_value_sent_ratio'] = sent_agg['high_value_sent_count'] / sent_agg['send_frequency']
    except Exception as e:
        print(f"   ❌ Error calculating sender statistics: {e}")
        sent_agg = pd.DataFrame()

    # Aggregate receiver statistics
    try:
        received_agg = p2p_txns.groupby('recipient_id').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'agent_id': 'nunique'
        })
        received_agg.columns = ['total_received', 'avg_received', 'receive_frequency', 'received_amount_std', 'unique_senders']
        received_agg['received_amount_std'] = received_agg['received_amount_std'].fillna(0)
        high_value_received = p2p_txns[p2p_txns['amount'] > high_value_threshold].groupby('recipient_id').size()
        received_agg['high_value_received_count'] = high_value_received.reindex(received_agg.index, fill_value=0)
        received_agg['high_value_received_ratio'] = received_agg['high_value_received_count'] / received_agg['receive_frequency']
    except Exception as e:
        print(f"   ❌ Error calculating receiver statistics: {e}")
        received_agg = pd.DataFrame()

    for node in G.nodes():
        try:
            if not sent_agg.empty and node in sent_agg.index:
                stats = sent_agg.loc[node]
                G.nodes[node]['total_sent_volume'] = stats['total_sent']
                G.nodes[node]['avg_sent_amount'] = stats['avg_sent']
                G.nodes[node]['send_frequency'] = stats['send_frequency']
                G.nodes[node]['unique_recipients_count'] = stats['unique_recipients']
                G.nodes[node]['sent_amount_variance'] = stats['sent_amount_std'] ** 2
                G.nodes[node]['high_value_sent_ratio'] = stats['high_value_sent_ratio']

            if not received_agg.empty and node in received_agg.index:
                stats = received_agg.loc[node]
                G.nodes[node]['total_received_volume'] = stats['total_received']
                G.nodes[node]['avg_received_amount'] = stats['avg_received']
                G.nodes[node]['receive_frequency'] = stats['receive_frequency']
                G.nodes[node]['unique_senders_count'] = stats['unique_senders']
                G.nodes[node]['received_amount_variance'] = stats['received_amount_std'] ** 2
                G.nodes[node]['high_value_received_ratio'] = stats['high_value_received_ratio']

            total_sent = G.nodes[node]['total_sent_volume']
            total_received = G.nodes[node]['total_received_volume']
            G.nodes[node]['transaction_velocity_score'] = total_sent + total_received

            total_volume = total_sent + total_received
            if total_volume > 0:
                G.nodes[node]['balance_flow_ratio'] = total_sent / total_volume
            else:
                G.nodes[node]['balance_flow_ratio'] = 0.5

            send_freq = G.nodes[node]['send_frequency']
            receive_freq = G.nodes[node]['receive_frequency']
            total_freq = send_freq + receive_freq
            G.nodes[node]['transaction_regularity'] = min(1.0, total_freq / 30)

            unique_contacts = G.nodes[node]['unique_recipients_count'] + G.nodes[node]['unique_senders_count']
            G.nodes[node]['network_diversity_score'] = min(1.0, unique_contacts / 20)

            sent_var = G.nodes[node]['sent_amount_variance']
            received_var = G.nodes[node]['received_amount_variance']
            avg_sent = G.nodes[node]['avg_sent_amount']
            avg_received = G.nodes[node]['avg_received_amount']

            sent_cv = (np.sqrt(sent_var) / avg_sent) if avg_sent > 0 else 0
            received_cv = (np.sqrt(received_var) / avg_received) if avg_received > 0 else 0
            G.nodes[node]['transaction_amount_consistency'] = 1 - min(1.0, (sent_cv + received_cv) / 2)

            high_val_sent = G.nodes[node].get('high_value_sent_ratio', 0)
            high_val_received = G.nodes[node].get('high_value_received_ratio', 0)
            G.nodes[node]['high_value_transaction_propensity'] = (high_val_sent + high_val_received) / 2
        except Exception as e:
            print(f"   ⚠️ Error processing node {node}: {e}")
            continue

    print("   ✅ Enhanced transaction features calculated and added")
    return G



if __name__ == "__main__":
    try:
        print("🚀 Starting Phase 2 enhanced credit risk feature engineering...")
        
        # ✅ UPDATED: Import Phase 2 functions
        from graph_builder import load_and_validate_data, create_heterogeneous_credit_graph
        
        # Load Phase 2 ultra-sanitized data
        agents_df, transactions_df, labels_df, agent_id_to_idx, idx_to_agent_id = load_and_validate_data()
        
        # Create heterogeneous credit risk graph
        G = create_heterogeneous_credit_graph(
            agents_df, transactions_df, labels_df, 
            edge_construction_method='enhanced'
        )
        
        print(f"📊 Base graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        
        # Apply enhanced feature engineering
        G = calculate_centrality_features(G)
        G = calculate_personalized_pagerank(G)
        G, communities, community_analysis = detect_communities_and_risk_clusters(G)
        G = calculate_transaction_features(G, transactions_df)
        
        print("💾 Saving enhanced credit risk graph...")
        
        # Save enhanced graph
        import pickle
        import os
        
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        enhanced_graph_path = os.path.join(output_dir, "phase2_enhanced_credit_risk_graph.pkl")
        with open(enhanced_graph_path, 'wb') as f:
            pickle.dump(G, f)
        
        print(f"✅ Enhanced graph saved to '{enhanced_graph_path}'")
        
        # ✅ FEATURE SUMMARY
        print(f"\n📋 === ENHANCED FEATURE SUMMARY ===")
        
        if G.nodes():
            sample_node = list(G.nodes())[0]
            sample_features = G.nodes[sample_node]
            
            # Count feature categories
            centrality_features = [k for k in sample_features.keys() if 'centrality' in k or 'pagerank' in k]
            behavioral_features = [k for k in sample_features.keys() if any(term in k for term in 
                                 ['behavioral', 'consistency', 'stability', 'frequency', 'diversity'])]
            transaction_features = [k for k in sample_features.keys() if any(term in k for term in 
                                  ['transaction', 'volume', 'amount', 'velocity', 'flow'])]
            community_features = [k for k in sample_features.keys() if 'community' in k]
            
            print(f"🔗 Centrality features: {len(centrality_features)}")
            print(f"🎯 Behavioral features: {len(behavioral_features)}")
            print(f"💳 Transaction features: {len(transaction_features)}")
            print(f"🏘️ Community features: {len(community_features)}")
            print(f"📊 Total enhanced features per node: {len(sample_features)}")
            
            print(f"\n🔍 Sample node enhanced features:")
            key_features = ['financial_stability_score', 'behavioral_consistency', 'pagerank_score', 
                          'transaction_velocity_score', 'community_avg_behavioral_risk']
            for feature in key_features:
                if feature in sample_features:
                    print(f"   {feature}: {sample_features[feature]:.4f}")
        
        print(f"\n🏘️ Community Analysis:")
        print(f"   📊 Total communities detected: {len(communities)}")
        high_risk_communities = [cid for cid, stats in community_analysis.items() 
                               if stats['avg_behavioral_risk_score'] > 0.6]
        print(f"   🚨 High behavioral risk communities: {len(high_risk_communities)}")
        
        print(f"\n✅ Phase 2 enhanced credit risk feature engineering completed!")
        print(f"🎯 Graph ready for advanced GNN training with comprehensive behavioral features")
        
    except Exception as e:
        print(f"❌ Error in enhanced feature engineering: {e}")
        import traceback
        traceback.print_exc()
