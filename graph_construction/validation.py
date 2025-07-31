import networkx as nx
import numpy as np
import pandas as pd

def analyze_graph_structure(G):
    """Analyze detailed graph properties for credit risk assessment"""
    print("=== Credit Risk Graph Analysis ===")
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
    
    # ✅ UPDATED: Credit risk analysis (PRIMARY FOCUS)
    risk_scores = []
    risk_profiles = []
    
    for node, data in G.nodes(data=True):
        risk_score = data.get('risk_score')
        risk_profile = data.get('risk_profile', 'Unknown')
        
        # Handle risk scores
        if risk_score is not None and pd.notna(risk_score):
            risk_scores.append(float(risk_score))
        else:
            risk_scores.append(0.5)  # Default neutral risk
            
        risk_profiles.append(risk_profile)
    
    # Credit risk statistics
    risk_scores = np.array(risk_scores)
    print(f"\n=== Credit Risk Distribution ===")
    print(f"Average risk score: {np.mean(risk_scores):.4f}")
    print(f"Risk score std deviation: {np.std(risk_scores):.4f}")
    print(f"Risk score range: [{np.min(risk_scores):.3f}, {np.max(risk_scores):.3f}]")
    
    # Risk profile categories
    risk_profile_counts = pd.Series(risk_profiles).value_counts()
    print(f"\nRisk profile distribution:")
    for profile, count in risk_profile_counts.items():
        percentage = (count / len(risk_profiles)) * 100
        print(f"  {profile}: {count} ({percentage:.1f}%)")
    
    # High-risk borrowers analysis
    high_risk_threshold = 0.7
    high_risk_nodes = [i for i, score in enumerate(risk_scores) if score > high_risk_threshold]
    print(f"\nHigh-risk borrowers (score > {high_risk_threshold}): {len(high_risk_nodes)} ({len(high_risk_nodes)/len(risk_scores)*100:.1f}%)")
    
    # ✅ SECONDARY: Fraud analysis (for validation, not primary target)
    fraud_nodes = []
    fraud_types_found = {}
    
    for node, data in G.nodes(data=True):
        fraud_val = data.get('fraud_type')
        
        # Check for valid fraud types (kept for validation purposes)
        if fraud_val and pd.notna(fraud_val) and fraud_val != '':
            if fraud_val in ['ring', 'bust_out', 'mule']:
                fraud_nodes.append(node)
                fraud_types_found[fraud_val] = fraud_types_found.get(fraud_val, 0) + 1
    
    print(f"\n=== Fraud Analysis (Reference Only) ===")
    print(f"Fraud nodes in graph: {len(fraud_nodes)} ({len(fraud_nodes)/G.number_of_nodes()*100:.1f}%)")
    if fraud_types_found:
        print("Fraud type breakdown:", fraud_types_found)
    
    # ✅ UPDATED: Transaction analysis for credit risk
    edge_weights = []
    edge_frequencies = []
    edge_consistencies = []
    
    for u, v, data in G.edges(data=True):
        edge_weights.append(data.get('total_amount', 0))
        edge_frequencies.append(data.get('frequency', 1))
        edge_consistencies.append(data.get('amount_consistency', 0))
    
    print(f"\n=== Transaction Network Analysis ===")
    print(f"Average transaction amount: ₹{np.mean(edge_weights):,.2f}")
    print(f"Max transaction amount: ₹{np.max(edge_weights):,.2f}")
    print(f"Average transaction frequency: {np.mean(edge_frequencies):.2f}")
    
    if edge_consistencies and any(edge_consistencies):
        print(f"Average amount consistency: {np.mean([c for c in edge_consistencies if c > 0]):.3f}")
    
    # ✅ NEW: Credit network metrics
    print(f"\n=== Credit Network Metrics ===")
    
    # Community risk concentration
    communities = list(nx.weakly_connected_components(G))
    print(f"Weakly connected components: {len(communities)}")
    
    # PageRank for influential borrowers
    try:
        pagerank_scores = nx.pagerank(G)
        top_influential = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 influential borrowers (PageRank): {[f'{node[:8]}...' for node, _ in top_influential]}")
    except:
        print("PageRank calculation failed (likely due to graph structure)")
    
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': np.mean(degrees),
        'avg_risk_score': np.mean(risk_scores),
        'risk_score_std': np.std(risk_scores),
        'high_risk_count': len(high_risk_nodes),
        'risk_profile_distribution': risk_profile_counts.to_dict(),
        'fraud_nodes': len(fraud_nodes),  # Secondary metric
        'fraud_types': fraud_types_found,  # Secondary metric
        'avg_transaction': np.mean(edge_weights),
        'max_transaction': np.max(edge_weights)
    }

if __name__ == "__main__":
    # ✅ UPDATED: Import with updated function signature
    from graph_builder import load_and_validate_data, create_transaction_graph
    
    # Load data with updated signature
    agents_df, transactions_df, original_agents_df, agent_id_to_idx, idx_to_agent_id = load_and_validate_data()
    transaction_graph = create_transaction_graph(agents_df, transactions_df, original_agents_df)
    
    # Run credit risk analysis
    graph_stats = analyze_graph_structure(transaction_graph)
    
    print(f"\n=== Summary ===")
    print(f"✅ Graph ready for credit risk GNN training!")
    print(f"✅ {graph_stats['nodes']} borrowers with avg risk score: {graph_stats['avg_risk_score']:.3f}")
    print(f"✅ {graph_stats['high_risk_count']} high-risk borrowers identified")
    print(f"✅ {graph_stats['edges']} transaction relationships")
    print(f"ℹ️ {graph_stats['fraud_nodes']} fraud cases for validation reference")
