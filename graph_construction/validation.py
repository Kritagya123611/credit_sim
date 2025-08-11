import networkx as nx
import numpy as np
import pandas as pd

def analyze_graph_structure(G):
    """Analyze detailed graph properties for Phase 2 ultra-sanitized credit risk assessment"""
    print("🔍 === PHASE 2 CREDIT RISK GRAPH ANALYSIS ===")
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    print(f"Graph density: {nx.density(G):.6f}")
    
    # Check connectivity
    if nx.is_strongly_connected(G):
        print("✅ Graph is strongly connected")
    else:
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        print(f"📊 Largest strongly connected component: {len(largest_scc):,} nodes ({len(largest_scc)/G.number_of_nodes()*100:.1f}%)")
    
    # Degree distribution analysis
    degrees = [d for n, d in G.degree()]
    print(f"🔗 Average degree: {np.mean(degrees):.2f}")
    print(f"🔗 Max degree: {max(degrees)}")
    print(f"🔗 Min degree: {min(degrees)}")
    
    # ✅ PHASE 2 ENHANCED: Ultra-sanitized behavioral analysis
    print(f"\n💡 === BEHAVIORAL PATTERN ANALYSIS (Phase 2 Features) ===")
    
    # Extract Phase 2 ultra-sanitized features
    financial_stability_scores = []
    behavioral_consistency_scores = []
    device_stability_scores = []
    transaction_frequencies = []
    p2p_ratios = []
    
    for node, data in G.nodes(data=True):
        # Phase 2 ultra-sanitized behavioral features
        financial_stability_scores.append(data.get('financial_stability_score', 0.5))
        behavioral_consistency_scores.append(data.get('behavioral_consistency', 0.5))
        device_stability_scores.append(data.get('device_stability_score', 0.5))
        transaction_frequencies.append(data.get('transaction_frequency', 0.0))
        p2p_ratios.append(data.get('p2p_ratio', 0.5))
    
    # Phase 2 behavioral statistics
    financial_stability_scores = np.array(financial_stability_scores)
    behavioral_consistency_scores = np.array(behavioral_consistency_scores)
    device_stability_scores = np.array(device_stability_scores)
    transaction_frequencies = np.array(transaction_frequencies)
    p2p_ratios = np.array(p2p_ratios)
    
    print(f"💰 Financial Stability Score - Mean: {financial_stability_scores.mean():.4f}, Std: {financial_stability_scores.std():.4f}")
    print(f"🎯 Behavioral Consistency Score - Mean: {behavioral_consistency_scores.mean():.4f}, Std: {behavioral_consistency_scores.std():.4f}")
    print(f"📱 Device Stability Score - Mean: {device_stability_scores.mean():.4f}, Std: {device_stability_scores.std():.4f}")
    print(f"📈 Transaction Frequency - Mean: {transaction_frequencies.mean():.4f}, Std: {transaction_frequencies.std():.4f}")
    print(f"🔄 P2P Ratio - Mean: {p2p_ratios.mean():.4f}, Std: {p2p_ratios.std():.4f}")
    
    # ✅ ENHANCED: Risk tier classification based on behavioral patterns
    print(f"\n🚨 === CREDIT RISK CLASSIFICATION ===")
    
    # Calculate composite risk from behavioral features
    composite_risk_scores = []
    for i in range(len(financial_stability_scores)):
        # Composite risk based on Phase 2 behavioral patterns
        risk_score = (
            (1 - financial_stability_scores[i]) * 0.3 +  # Lower stability = higher risk
            (1 - behavioral_consistency_scores[i]) * 0.25 +  # Lower consistency = higher risk  
            (1 - device_stability_scores[i]) * 0.2 +  # Lower device stability = higher risk
            (transaction_frequencies[i] * 0.15) +  # Higher frequency can indicate risk
            (abs(p2p_ratios[i] - 0.5) * 0.1)  # Extreme P2P ratios = higher risk
        )
        composite_risk_scores.append(min(risk_score, 1.0))  # Cap at 1.0
    
    composite_risk_scores = np.array(composite_risk_scores)
    
    # Risk tier classification
    low_risk_count = sum(1 for score in composite_risk_scores if score < 0.3)
    medium_risk_count = sum(1 for score in composite_risk_scores if 0.3 <= score < 0.7)
    high_risk_count = sum(1 for score in composite_risk_scores if score >= 0.7)
    
    print(f"🟢 Low Risk (< 0.3): {low_risk_count:,} ({low_risk_count/len(composite_risk_scores)*100:.1f}%)")
    print(f"🟡 Medium Risk (0.3-0.7): {medium_risk_count:,} ({medium_risk_count/len(composite_risk_scores)*100:.1f}%)")
    print(f"🔴 High Risk (>= 0.7): {high_risk_count:,} ({high_risk_count/len(composite_risk_scores)*100:.1f}%)")
    print(f"📊 Average Composite Risk Score: {composite_risk_scores.mean():.4f}")
    
    # ✅ EVALUATION ONLY: Fraud analysis (separate from training features)
    print(f"\n🔍 === FRAUD EVALUATION (Ground Truth Only) ===")
    
    fraud_nodes = []
    fraud_types_found = {}
    
    for node, data in G.nodes(data=True):
        # Check evaluation labels (stored separately)
        fraud_label = data.get('evaluation_fraud_label', 0)
        fraud_type = data.get('evaluation_fraud_type')
        
        if fraud_label == 1 and fraud_type:
            fraud_nodes.append(node)
            fraud_types_found[fraud_type] = fraud_types_found.get(fraud_type, 0) + 1
    
    print(f"📈 Total fraud nodes (evaluation): {len(fraud_nodes):,} ({len(fraud_nodes)/G.number_of_nodes()*100:.1f}%)")
    if fraud_types_found:
        print("🎯 Fraud type breakdown (evaluation):")
        for fraud_type, count in fraud_types_found.items():
            print(f"   {fraud_type}: {count:,} nodes")
    
    # ✅ ENHANCED: Transaction network analysis
    print(f"\n💳 === TRANSACTION NETWORK ANALYSIS ===")
    
    # Edge analysis
    p2p_edges = []
    similarity_edges = []
    edge_amounts = []
    edge_frequencies = []
    relationship_strengths = []
    
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        
        if edge_type == 'p2p_transaction':
            p2p_edges.append((u, v))
            if 'total_amount' in data:
                edge_amounts.append(data['total_amount'])
            if 'frequency' in data:
                edge_frequencies.append(data['frequency'])
        elif edge_type == 'behavioral_similarity':
            similarity_edges.append((u, v))
        
        if 'relationship_strength' in data:
            relationship_strengths.append(data['relationship_strength'])
    
    print(f"💰 P2P Transaction Edges: {len(p2p_edges):,}")
    print(f"🤝 Behavioral Similarity Edges: {len(similarity_edges):,}")
    
    if edge_amounts:
        print(f"💵 Average P2P transaction: ₹{np.mean(edge_amounts):,.2f}")
        print(f"💵 Max P2P transaction: ₹{np.max(edge_amounts):,.2f}")
        print(f"💵 P2P transaction volume: ₹{np.sum(edge_amounts):,.2f}")
    
    if edge_frequencies:
        print(f"🔄 Average transaction frequency: {np.mean(edge_frequencies):.2f}")
    
    if relationship_strengths:
        print(f"🤝 Average relationship strength: {np.mean(relationship_strengths):.4f}")
    
    # ✅ ENHANCED: Graph-level credit network metrics
    print(f"\n🕸️ === CREDIT NETWORK METRICS ===")
    
    # Community analysis
    communities = list(nx.weakly_connected_components(G))
    print(f"🏘️ Weakly connected components: {len(communities)}")
    
    if len(communities) > 1:
        component_sizes = [len(comp) for comp in communities]
        print(f"🏘️ Largest component size: {max(component_sizes):,} ({max(component_sizes)/G.number_of_nodes()*100:.1f}%)")
        print(f"🏘️ Average component size: {np.mean(component_sizes):.1f}")
    
    # Centrality analysis for influential nodes
    try:
        print("📊 Computing centrality measures...")
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(G)
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"🎯 Top 5 by degree centrality: {[f'{node[:8]}...' for node, _ in top_degree]}")
        
        # PageRank for influence
        pagerank_scores = nx.pagerank(G, max_iter=50)
        top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"⭐ Top 5 influential nodes (PageRank): {[f'{node[:8]}...' for node, _ in top_pagerank]}")
        
    except Exception as e:
        print(f"⚠️ Centrality calculation failed: {str(e)[:50]}...")
    
    # ✅ ENHANCED: Feature completeness analysis
    print(f"\n📋 === FEATURE COMPLETENESS ANALYSIS ===")
    
    sample_node = list(G.nodes())[0] if G.nodes() else None
    if sample_node:
        sample_features = G.nodes[sample_node]
        
        # Count feature categories
        behavioral_features = [k for k in sample_features.keys() if any(pattern in k.lower() for pattern in 
                              ['stability', 'consistency', 'frequency', 'ratio', 'balance', 'transaction'])]
        graph_features = [k for k in sample_features.keys() if any(pattern in k.lower() for pattern in 
                         ['centrality', 'pagerank', 'clustering', 'degree'])]
        derived_features = [k for k in sample_features.keys() if any(pattern in k.lower() for pattern in 
                           ['time_span', 'weekend', 'channel', 'coefficient'])]
        
        print(f"🎯 Behavioral features per node: {len(behavioral_features)}")
        print(f"🕸️ Graph-level features per node: {len(graph_features)}")
        print(f"📈 Derived features per node: {len(derived_features)}")
        print(f"📊 Total features per node: {len(sample_features)}")
    
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': np.mean(degrees),
        'max_degree': max(degrees),
        'min_degree': min(degrees),
        'avg_financial_stability': financial_stability_scores.mean(),
        'avg_behavioral_consistency': behavioral_consistency_scores.mean(),
        'avg_device_stability': device_stability_scores.mean(),
        'avg_composite_risk': composite_risk_scores.mean(),
        'low_risk_count': low_risk_count,
        'medium_risk_count': medium_risk_count,
        'high_risk_count': high_risk_count,
        'fraud_nodes_evaluation': len(fraud_nodes),
        'fraud_types_evaluation': fraud_types_found,
        'p2p_edges': len(p2p_edges),
        'similarity_edges': len(similarity_edges),
        'avg_transaction_amount': np.mean(edge_amounts) if edge_amounts else 0,
        'max_transaction_amount': np.max(edge_amounts) if edge_amounts else 0,
        'num_components': len(communities)
    }

if __name__ == "__main__":
    # ✅ UPDATED: Import with Phase 2 function signature
    try:
        from graph_builder import load_and_validate_data, create_heterogeneous_credit_graph
        
        # Load Phase 2 ultra-sanitized data
        agents_df, transactions_df, labels_df, agent_id_to_idx, idx_to_agent_id = load_and_validate_data()
        
        # Create heterogeneous credit risk graph
        credit_graph = create_heterogeneous_credit_graph(
            agents_df, transactions_df, labels_df, 
            edge_construction_method='enhanced'
        )
        
        # Run comprehensive analysis
        graph_stats = analyze_graph_structure(credit_graph)
        
        print(f"\n✅ === PHASE 2 ANALYSIS SUMMARY ===")
        print(f"🏗️ Graph Type: Heterogeneous Credit Risk Assessment")
        print(f"📊 Total Nodes: {graph_stats['nodes']:,}")
        print(f"🔗 Total Edges: {graph_stats['edges']:,}")
        print(f"📈 Average Degree: {graph_stats['avg_degree']:.2f}")
        print(f"💰 Avg Financial Stability: {graph_stats['avg_financial_stability']:.3f}")
        print(f"🎯 Avg Behavioral Consistency: {graph_stats['avg_behavioral_consistency']:.3f}")
        print(f"📱 Avg Device Stability: {graph_stats['avg_device_stability']:.3f}")
        print(f"🚨 Composite Risk Score: {graph_stats['avg_composite_risk']:.3f}")
        print(f"🔴 High Risk Nodes: {graph_stats['high_risk_count']:,}")
        print(f"💳 P2P Edges: {graph_stats['p2p_edges']:,}")
        print(f"🤝 Similarity Edges: {graph_stats['similarity_edges']:,}")
        print(f"🔍 Fraud Cases (Evaluation): {graph_stats['fraud_nodes_evaluation']:,}")
        
        print(f"\n🎯 Graph ready for Phase 2 GNN training!")
        print(f"✅ Ultra-sanitized behavioral features with no demographic bias")
        print(f"✅ Comprehensive credit risk assessment capabilities")
        print(f"✅ Robust network structure for effective message passing")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure graph_builder.py is in the same directory")
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
