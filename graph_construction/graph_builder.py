import pandas as pd
import numpy as np
import networkx as nx
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', message='.*A value is trying to be set on a copy of a slice from a DataFrame.*')


def load_and_validate_data():
    """Load Phase 2 ultra-sanitized agents and transactions data with validation"""
    print("🔍 Loading Phase 2 ultra-sanitized data...")
    
    # ✅ UPDATED: Priority paths for Phase 2 ultra-sanitized files
    paths_to_try = [
        ('output/phase2_agents_ultra_sanitized.csv', 'output/phase2_transactions_ultra_sanitized.csv', 'output/phase2_gnn_labels_evaluation_only.csv'),
        ('output/phase2_gnn_features_ultra_sanitized.csv', 'output/phase2_transactions_ultra_sanitized.csv', 'output/phase2_gnn_labels_evaluation_only.csv'),
        ('output/gnn_features_sanitized.csv', 'output/gnn_ready_transactions_sanitized.csv', 'output/ground_truth_labels.csv'),
        ('../output/phase2_agents_ultra_sanitized.csv', '../output/phase2_transactions_ultra_sanitized.csv', '../output/phase2_gnn_labels_evaluation_only.csv'),
        ('./output/phase2_agents_ultra_sanitized.csv', './output/phase2_transactions_ultra_sanitized.csv', './output/phase2_gnn_labels_evaluation_only.csv')
    ]
    
    agents_df = None
    transactions_df = None
    labels_df = None
    
    for agents_path, transactions_path, labels_path in paths_to_try:
        try:
            if os.path.exists(agents_path) and os.path.exists(transactions_path):
                agents_df = pd.read_csv(agents_path)
                transactions_df = pd.read_csv(transactions_path)
                # Labels are separate for evaluation only
                if os.path.exists(labels_path):
                    labels_df = pd.read_csv(labels_path)
                    print("✅ Loaded ground truth labels for evaluation")
                break
        except Exception as e:
            continue
    
    if agents_df is None:
        raise FileNotFoundError("❌ Could not find Phase 2 ultra-sanitized data files. Run simulation_engine.py first.")
    
    print(f"✅ Loaded {len(agents_df)} agents with {agents_df.shape[1]} ultra-sanitized behavioral features")
    print(f"✅ Loaded {len(transactions_df)} transactions")
    
    # ✅ ENHANCED: Validate data quality for Phase 2
    _validate_data_quality(agents_df, transactions_df, labels_df)
    
    # Create agent ID mapping for graph indexing
    agent_ids = agents_df['agent_id'].tolist()
    agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    idx_to_agent_id = {idx: agent_id for agent_id, idx in agent_id_to_idx.items()}
    
    print(f"🔗 Created agent ID mappings for {len(agent_id_to_idx)} agents")
    
    return agents_df, transactions_df, labels_df, agent_id_to_idx, idx_to_agent_id

def _validate_data_quality(agents_df: pd.DataFrame, transactions_df: pd.DataFrame, labels_df: pd.DataFrame = None):
    """Validate Phase 2 ultra-sanitized data quality"""
    print("🛡️ Validating ultra-sanitized data quality...")
    
    # ✅ PRIVACY COMPLIANCE: Ensure no overfitting fields present
    forbidden_patterns = ['risk_score', 'risk_profile', 'fraud_type', 'economic_class', 'archetype', 'personality']
    risky_columns = []
    
    for col in agents_df.columns:
        if any(pattern in col.lower() for pattern in forbidden_patterns):
            risky_columns.append(col)
    
    if risky_columns:
        warnings.warn(f"⚠️ Potentially risky columns found: {risky_columns}. These should be removed for anti-overfitting compliance.")
    
    # ✅ FEATURE VALIDATION: Check for expected Phase 2 features
    expected_features = [
        'account_balance', 'avg_daily_balance', 'balance_volatility',
        'total_transactions', 'transaction_frequency', 'p2p_ratio',
        'device_stability_score', 'financial_stability_score', 'behavioral_consistency'
    ]
    
    missing_features = [f for f in expected_features if f not in agents_df.columns]
    if missing_features:
        print(f"⚠️ Missing expected features: {missing_features}")
    else:
        print("✅ All expected Phase 2 behavioral features present")
    
    # ✅ TRANSACTION VALIDATION
    required_txn_fields = ['agent_id', 'txn_type', 'amount', 'date', 'channel', 'balance']
    missing_txn_fields = [f for f in required_txn_fields if f not in transactions_df.columns]
    if missing_txn_fields:
        print(f"⚠️ Missing transaction fields: {missing_txn_fields}")

def create_heterogeneous_credit_graph(agents_df: pd.DataFrame, 
                                    transactions_df: pd.DataFrame, 
                                    labels_df: pd.DataFrame = None,
                                    edge_construction_method: str = 'enhanced') -> nx.DiGraph:
    """
    ✅ ENHANCED: Create heterogeneous graph for Phase 2 credit risk assessment
    
    Args:
        agents_df: Ultra-sanitized agent behavioral features
        transactions_df: Ultra-sanitized transaction data
        labels_df: Ground truth labels (evaluation only)
        edge_construction_method: Method for edge construction ('enhanced', 'similarity', 'temporal')
    
    Returns:
        NetworkX directed graph with enhanced credit risk features
    """
    print(f"🏗️ Building heterogeneous credit risk graph using '{edge_construction_method}' method...")
    
    G = nx.DiGraph()
    
    # ✅ PHASE 2: Add nodes with ultra-sanitized behavioral features
    for _, agent in agents_df.iterrows():
        agent_id = agent['agent_id']
        
        # ✅ ANTI-OVERFITTING: Only use behavioral features (no demographics)
        node_features = agent.drop('agent_id').to_dict()
        
        # ✅ ENHANCED: Calculate derived credit risk indicators from behavioral patterns
        derived_features = _calculate_derived_credit_features(agent, transactions_df)
        node_features.update(derived_features)
        
        # ✅ EVALUATION ONLY: Add ground truth labels if available (stored separately)
        if labels_df is not None:
            label_info = labels_df[labels_df['agent_id'] == agent_id]
            if not label_info.empty:
                # Labels for evaluation - never used in training
                node_features['evaluation_fraud_label'] = label_info['is_fraud'].iloc[0]
                node_features['evaluation_fraud_type'] = label_info.get('fraud_type', [None]).iloc[0]
            else:
                node_features['evaluation_fraud_label'] = 0  # Default: not fraud
                node_features['evaluation_fraud_type'] = None
        
        G.add_node(agent_id, **node_features)
    
    # ✅ ENHANCED: Multi-method edge construction
    if edge_construction_method == 'enhanced':
        G = _create_enhanced_edges(G, agents_df, transactions_df)
    elif edge_construction_method == 'similarity':
        G = _create_similarity_based_edges(G, agents_df)
    elif edge_construction_method == 'temporal':
        G = _create_temporal_edges(G, transactions_df)
    else:
        raise ValueError(f"Unknown edge construction method: {edge_construction_method}")
    
    # ✅ ENHANCED: Calculate graph-level credit risk features
    G = _calculate_graph_level_features(G)
    
    print(f"🎯 Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"📊 Graph density: {nx.density(G):.6f}")
    
    # ✅ EVALUATION METRICS: Print evaluation label distribution if available
    if labels_df is not None:
        fraud_count = labels_df['is_fraud'].sum()
        total_count = len(labels_df)
        print(f"📈 Evaluation labels: {fraud_count} fraud, {total_count - fraud_count} legitimate ({fraud_count/total_count*100:.1f}% fraud rate)")
    
    return G

def _calculate_derived_credit_features(agent: pd.Series, transactions_df: pd.DataFrame) -> Dict:
    """Calculate derived credit risk features from behavioral patterns"""
    agent_id = agent['agent_id']
    agent_txns = transactions_df[transactions_df['agent_id'] == agent_id].copy()  # ✅ ADD .copy()
    
    derived = {}
    
    if len(agent_txns) > 0:
        # ✅ TEMPORAL PATTERNS: Transaction timing analysis
        if 'date' in agent_txns.columns:
            try:
                agent_txns.loc[:, 'date'] = pd.to_datetime(agent_txns['date'])  # ✅ USE .loc
                # Transaction spread over time
                date_range = (agent_txns['date'].max() - agent_txns['date'].min()).days
                derived['transaction_time_span_days'] = max(1, date_range)
                
                # Weekend vs weekday activity
                agent_txns.loc[:, 'weekday'] = agent_txns['date'].dt.weekday  # ✅ USE .loc
                weekend_ratio = len(agent_txns[agent_txns['weekday'] >= 5]) / len(agent_txns)
                derived['weekend_activity_ratio'] = round(weekend_ratio, 4)
            except:
                derived['transaction_time_span_days'] = 1
                derived['weekend_activity_ratio'] = 0.5
        
        # ✅ CHANNEL DIVERSITY: Payment method patterns
        if 'channel' in agent_txns.columns:
            unique_channels = agent_txns['channel'].nunique()
            derived['channel_diversity_score'] = min(unique_channels / 5.0, 1.0)  # Normalize to 0-1
        
        # ✅ AMOUNT PATTERNS: Transaction amount analysis
        if 'amount' in agent_txns.columns:
            amounts = agent_txns['amount']
            derived['amount_coefficient_variation'] = amounts.std() / amounts.mean() if amounts.mean() > 0 else 0
            derived['large_transaction_ratio'] = (amounts > amounts.quantile(0.9)).mean()
            
        # ✅ BEHAVIORAL CONSISTENCY: Transaction pattern regularity
        derived['transaction_consistency_score'] = _calculate_transaction_consistency(agent_txns)
    else:
        # Default values for agents with no transactions
        derived.update({
            'transaction_time_span_days': 0,
            'weekend_activity_ratio': 0.5,
            'channel_diversity_score': 0.0,
            'amount_coefficient_variation': 0.0,
            'large_transaction_ratio': 0.0,
            'transaction_consistency_score': 0.0
        })
    
    return derived

def _calculate_transaction_consistency(agent_txns: pd.DataFrame) -> float:
    """Calculate transaction pattern consistency score"""
    if len(agent_txns) < 3:
        return 0.5  # Neutral for low activity
    
    try:
        # Amount consistency
        amounts = agent_txns['amount'].values
        amount_consistency = 1 - (np.std(amounts) / np.mean(amounts)) if np.mean(amounts) > 0 else 0
        
        # Frequency consistency (if date available)
        if 'date' in agent_txns.columns:
            dates = pd.to_datetime(agent_txns['date'])
            time_diffs = dates.diff().dt.days.dropna()
            if len(time_diffs) > 0:
                frequency_consistency = 1 - (time_diffs.std() / time_diffs.mean()) if time_diffs.mean() > 0 else 0
                consistency = (amount_consistency * 0.6) + (frequency_consistency * 0.4)
            else:
                consistency = amount_consistency
        else:
            consistency = amount_consistency
        
        return max(0, min(1, consistency))  # Clamp to [0, 1]
    except:
        return 0.5

def _create_enhanced_edges(G: nx.DiGraph, agents_df: pd.DataFrame, transactions_df: pd.DataFrame) -> nx.DiGraph:
    """Create enhanced edges using multiple signals"""
    print("🔗 Creating enhanced edges using multiple signals...")
    
    # ✅ METHOD 1: P2P Transaction edges (direct financial relationships)
    p2p_conditions = (
        (transactions_df['recipient_id'].notna()) &
        (transactions_df['recipient_id'] != '') &
        (transactions_df['recipient_id'] != transactions_df['agent_id'])  # Avoid self-loops
    )
    p2p_txns = transactions_df[p2p_conditions].copy()
    
    print(f"   📊 Found {len(p2p_txns)} P2P transactions for edge creation")
    
    # Group P2P transactions by agent pairs
    p2p_edges = {}
    for _, txn in p2p_txns.iterrows():
        source = txn['agent_id']
        target = txn['recipient_id']
        
        # Only create edges between existing nodes
        if source in G.nodes() and target in G.nodes():
            edge_key = (source, target)
            
            if edge_key not in p2p_edges:
                p2p_edges[edge_key] = {
                    'total_amount': 0,
                    'frequency': 0,
                    'channels': set(),
                    'amounts': [],
                    'dates': []
                }
            
            p2p_edges[edge_key]['total_amount'] += txn['amount']
            p2p_edges[edge_key]['frequency'] += 1
            p2p_edges[edge_key]['channels'].add(txn.get('channel', 'Unknown'))
            p2p_edges[edge_key]['amounts'].append(txn['amount'])
            if 'date' in txn:
                p2p_edges[edge_key]['dates'].append(txn['date'])
    
    # Add P2P edges with enhanced features
    for (source, target), edge_data in p2p_edges.items():
        amounts = edge_data['amounts']
        
        # ✅ ENHANCED: Credit risk edge features
        edge_features = {
            'total_amount': edge_data['total_amount'],
            'frequency': edge_data['frequency'],
            'avg_amount': np.mean(amounts),
            'amount_std': np.std(amounts) if len(amounts) > 1 else 0,
            'amount_consistency': 1 - (np.std(amounts) / np.mean(amounts)) if np.mean(amounts) > 0 else 0,
            'channel_diversity': len(edge_data['channels']),
            'channels': list(edge_data['channels']),
            'relationship_strength': min(edge_data['frequency'] / 10.0, 1.0),  # Normalize frequency
            'high_value_ratio': sum(1 for amt in amounts if amt > 50000) / len(amounts),
            'edge_type': 'p2p_transaction'
        }
        
        # ✅ TEMPORAL FEATURES: Time-based relationship analysis
        if edge_data['dates']:
            try:
                dates = pd.to_datetime(edge_data['dates'])
                time_span = (dates.max() - dates.min()).days
                edge_features['relationship_duration_days'] = max(1, time_span)
                edge_features['transaction_velocity'] = edge_data['frequency'] / max(1, time_span / 30)  # Per month
            except:
                edge_features['relationship_duration_days'] = 1
                edge_features['transaction_velocity'] = edge_data['frequency']
        
        G.add_edge(source, target, **edge_features)
    
    print(f"   ✅ Added {len(p2p_edges)} P2P transaction edges")
    
    # ✅ METHOD 2: Behavioral similarity edges (indirect risk relationships)
    similarity_edges = _create_behavioral_similarity_edges(G, agents_df, threshold=0.85, max_connections=5)
    print(f"   ✅ Added {similarity_edges} behavioral similarity edges")
    
    return G

def _create_behavioral_similarity_edges(G: nx.DiGraph, agents_df: pd.DataFrame, threshold: float = 0.85, max_connections: int = 5) -> int:
    """Create edges based on behavioral similarity for credit risk assessment"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # ✅ SELECT: Key behavioral features for similarity
    similarity_features = [
        'financial_stability_score', 'behavioral_consistency', 'transaction_frequency',
        'device_stability_score', 'p2p_ratio', 'balance_volatility'
    ]
    
    # Filter to available features
    available_features = [f for f in similarity_features if f in agents_df.columns]
    
    if len(available_features) < 3:
        print(f"   ⚠️ Insufficient features for similarity calculation: {available_features}")
        return 0
    
    # Create feature matrix
    feature_matrix = agents_df[available_features].fillna(0).values
    similarity_matrix = cosine_similarity(feature_matrix)
    
    edges_added = 0
    agent_ids = agents_df['agent_id'].tolist()
    
    for i, agent_id in enumerate(agent_ids):
        # Find most similar agents
        similarities = similarity_matrix[i]
        similar_indices = np.argsort(similarities)[::-1][1:max_connections+1]  # Exclude self
        
        for j in similar_indices:
            similarity_score = similarities[j]
            if similarity_score >= threshold:
                target_id = agent_ids[j]
                
                # Add bidirectional similarity edge
                G.add_edge(agent_id, target_id,
                          similarity_score=float(similarity_score),
                          edge_type='behavioral_similarity',
                          relationship_strength=float(similarity_score),
                          features_used=available_features)
                edges_added += 1
    
    return edges_added

def _create_similarity_based_edges(G: nx.DiGraph, agents_df: pd.DataFrame) -> nx.DiGraph:
    """Create edges based purely on behavioral similarity"""
    print("🔗 Creating similarity-based edges...")
    similarity_edges = _create_behavioral_similarity_edges(G, agents_df, threshold=0.80, max_connections=8)
    print(f"   ✅ Added {similarity_edges} similarity-based edges")
    return G

def _create_temporal_edges(G: nx.DiGraph, transactions_df: pd.DataFrame) -> nx.DiGraph:
    """Create edges based on temporal transaction patterns"""
    print("🔗 Creating temporal-based edges...")
    
    if 'date' not in transactions_df.columns:
        print("   ⚠️ No date column found for temporal edge creation")
        return G
    
    try:
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df['date_only'] = transactions_df['date'].dt.date
        
        # Group by date and find agents active on the same days
        daily_activity = transactions_df.groupby('date_only')['agent_id'].apply(list).to_dict()
        
        edges_added = 0
        for date, active_agents in daily_activity.items():
            if len(active_agents) > 1:
                # Create temporal co-activity edges
                active_agents = list(set(active_agents))  # Remove duplicates
                for i, agent1 in enumerate(active_agents):
                    for agent2 in active_agents[i+1:]:
                        if agent1 in G.nodes() and agent2 in G.nodes():
                            # Check if edge already exists
                            if not G.has_edge(agent1, agent2):
                                G.add_edge(agent1, agent2,
                                          edge_type='temporal_coactivity',
                                          coactivity_date=str(date),
                                          relationship_strength=0.3)  # Lower strength for temporal
                                edges_added += 1
        
        print(f"   ✅ Added {edges_added} temporal co-activity edges")
    except Exception as e:
        print(f"   ❌ Error creating temporal edges: {e}")
    
    return G

def _calculate_graph_level_features(G: nx.DiGraph) -> nx.DiGraph:
    """Calculate graph-level features for enhanced credit risk assessment"""
    print("📊 Calculating graph-level credit risk features...")
    
    try:
        # ✅ CENTRALITY MEASURES: Node importance in credit network
        print("   📈 Computing centrality measures...")
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))  # Sample for large graphs
        
        # ✅ CLUSTERING: Community detection for risk groups
        try:
            clustering_coef = nx.clustering(G.to_undirected())
        except:
            clustering_coef = {node: 0.0 for node in G.nodes()}
        
        # ✅ PAGE RANK: Authority/influence in credit network
        try:
            pagerank = nx.pagerank(G, max_iter=50)
        except:
            pagerank = {node: 1.0/G.number_of_nodes() for node in G.nodes()}
        
        # Add features to nodes
        for node in G.nodes():
            G.nodes[node].update({
                'degree_centrality': degree_centrality.get(node, 0.0),
                'betweenness_centrality': betweenness_centrality.get(node, 0.0),
                'clustering_coefficient': clustering_coef.get(node, 0.0),
                'pagerank_score': pagerank.get(node, 0.0),
                'total_connections': G.degree(node),
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node)
            })
        
        print("   ✅ Graph-level features calculated successfully")
        
    except Exception as e:
        print(f"   ⚠️ Error calculating some graph features: {e}")
    
    return G

def export_graph_for_gnn(G: nx.DiGraph, output_dir: str = 'output') -> Dict:
    """Export graph in formats suitable for GNN training"""
    print(f"💾 Exporting graph for GNN training to '{output_dir}'...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ✅ NODE FEATURES: Extract node feature matrix
    node_features = []
    node_ids = []
    feature_names = None
    
    for node_id, node_data in G.nodes(data=True):
        node_ids.append(node_id)
        
        # Remove evaluation labels from features (keep separate)
        features = {k: v for k, v in node_data.items() 
                   if not k.startswith('evaluation_') and k != 'agent_id'}
        
        if feature_names is None:
            feature_names = list(features.keys())
        
        # Create feature vector in consistent order
        feature_vector = [features.get(fname, 0.0) for fname in feature_names]
        node_features.append(feature_vector)
    
    # ✅ EDGE FEATURES: Extract edge information
    edge_list = []
    edge_features = []
    edge_feature_names = None
    
    for source, target, edge_data in G.edges(data=True):
        source_idx = node_ids.index(source)
        target_idx = node_ids.index(target)
        edge_list.append([source_idx, target_idx])
        
        if edge_feature_names is None:
            edge_feature_names = [k for k in edge_data.keys() if isinstance(edge_data[k], (int, float))]
        
        edge_vector = [edge_data.get(fname, 0.0) for fname in edge_feature_names]
        edge_features.append(edge_vector)
    
    # ✅ EXPORT: Save in multiple formats
    export_data = {
        'node_features': np.array(node_features, dtype=np.float32),
        'node_ids': node_ids,
        'feature_names': feature_names,
        'edge_list': np.array(edge_list, dtype=np.int64).T,  # Transpose for PyTorch Geometric
        'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else None,
        'edge_feature_names': edge_feature_names,
        'num_nodes': len(node_ids),
        'num_edges': len(edge_list)
    }
    
    # ✅ SAVE: NumPy format for GNN frameworks
    np.savez_compressed(
        os.path.join(output_dir, 'credit_risk_graph.npz'),
        node_features=export_data['node_features'],
        edge_index=export_data['edge_list'],
        edge_features=export_data['edge_features'] if export_data['edge_features'] is not None else np.array([]),
        node_ids=np.array(node_ids),
        feature_names=np.array(feature_names),
        edge_feature_names=np.array(edge_feature_names) if edge_feature_names else np.array([])
    )
    
    # ✅ SAVE: Metadata
    import json
    metadata = {
        'num_nodes': export_data['num_nodes'],
        'num_edges': export_data['num_edges'],
        'num_node_features': len(feature_names),
        'num_edge_features': len(edge_feature_names) if edge_feature_names else 0,
        'feature_names': feature_names,
        'edge_feature_names': edge_feature_names,
        'graph_type': 'heterogeneous_credit_risk',
        'data_version': 'phase2_ultra_sanitized'
    }
    
    with open(os.path.join(output_dir, 'graph_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Graph exported successfully:")
    print(f"   📊 Nodes: {export_data['num_nodes']}, Edges: {export_data['num_edges']}")
    print(f"   📈 Node features: {len(feature_names)}, Edge features: {len(edge_feature_names) if edge_feature_names else 0}")
    
    return export_data

if __name__ == "__main__":
    # ✅ ENHANCED: Test the Phase 2 graph construction
    try:
        print("🚀 Starting Phase 2 Credit Risk Graph Construction...")
        
        # Load Phase 2 ultra-sanitized data
        agents_df, transactions_df, labels_df, agent_id_to_idx, idx_to_agent_id = load_and_validate_data()
        
        # Create heterogeneous credit risk graph
        credit_graph = create_heterogeneous_credit_graph(
            agents_df, transactions_df, labels_df, 
            edge_construction_method='enhanced'
        )
        
        # Export for GNN training
        export_data = export_graph_for_gnn(credit_graph)
        
        print(f"\n🎯 === PHASE 2 CREDIT RISK GRAPH SUMMARY ===")
        print(f"🏗️ Graph Type: Heterogeneous Credit Risk Assessment")
        print(f"📊 Nodes: {credit_graph.number_of_nodes():,}")
        print(f"🔗 Edges: {credit_graph.number_of_edges():,}")
        print(f"📈 Graph density: {nx.density(credit_graph):.6f}")
        print(f"🎯 Node features: {len(export_data['feature_names'])}")
        print(f"🔗 Edge features: {len(export_data['edge_feature_names']) if export_data['edge_feature_names'] else 0}")
        
        # ✅ SAMPLE ANALYSIS: Show sample node features
        if credit_graph.nodes():
            sample_node = list(credit_graph.nodes())[0]
            sample_features = credit_graph.nodes[sample_node]
            
            print(f"\n🔍 === SAMPLE NODE ANALYSIS ===")
            print(f"Node ID: {sample_node}")
            print("Key behavioral features:")
            behavioral_keys = ['financial_stability_score', 'behavioral_consistency', 'device_stability_score', 
                             'transaction_frequency', 'degree_centrality', 'pagerank_score']
            for key in behavioral_keys:
                if key in sample_features:
                    print(f"  {key}: {sample_features[key]:.4f}")
        
        # ✅ EVALUATION: Show fraud distribution if available
        if labels_df is not None:
            fraud_nodes = [node for node, data in credit_graph.nodes(data=True) 
                          if data.get('evaluation_fraud_label', 0) == 1]
            print(f"\n📈 === EVALUATION METRICS ===")
            print(f"Fraud nodes in graph: {len(fraud_nodes)}")
            print(f"Legitimate nodes in graph: {credit_graph.number_of_nodes() - len(fraud_nodes)}")
            print(f"Graph fraud rate: {len(fraud_nodes)/credit_graph.number_of_nodes()*100:.1f}%")
        
        print(f"\n✅ Phase 2 Credit Risk Graph ready for GNN training!")
        print(f"📁 Files saved: credit_risk_graph.npz, graph_metadata.json")
        
    except Exception as e:
        print(f"❌ Error in graph construction: {e}")
        import traceback
        traceback.print_exc()
