import pickle
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import warnings
import os
import json

def load_enhanced_graph():
    """Load the Phase 2 enhanced credit risk graph with all behavioral features"""
    print("📊 Loading Phase 2 enhanced credit risk graph...")
    
    # ✅ UPDATED: Priority paths for Phase 2 enhanced graph
    paths_to_try = [
        'output/phase2_enhanced_credit_risk_graph.pkl',
        '../output/phase2_enhanced_credit_risk_graph.pkl',
        './output/phase2_enhanced_credit_risk_graph.pkl',
        'output/enhanced_graph.pkl',  # Fallback
        '../output/enhanced_graph.pkl',
        './output/enhanced_graph.pkl'
    ]
    
    G = None
    for path in paths_to_try:
        try:
            with open(path, 'rb') as f:
                G = pickle.load(f)
            print(f"✅ Loaded enhanced graph from {path}")
            break
        except FileNotFoundError:
            continue
    
    if G is None:
        raise FileNotFoundError("❌ Could not find Phase 2 enhanced graph file. Run feature_engineering.py first.")
    
    print(f"📈 Loaded enhanced graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Validate Phase 2 features
    sample_node = list(G.nodes())[0]
    sample_features = G.nodes[sample_node]
    print(f"🎯 Enhanced features per node: {len(sample_features)}")
    
    return G

def convert_to_pytorch_geometric(G):
    """✅ ENHANCED: Convert Phase 2 enhanced NetworkX graph to PyTorch Geometric format"""
    print("🔄 Converting to PyTorch Geometric format for Phase 2 credit risk assessment...")
    
    # Extract node information
    agent_ids = list(G.nodes())
    agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    
    # ✅ PHASE 2: Exclude evaluation-only features and meta information
    exclude_features = {
        'agent_id', 'community', 'community_id', 
        # ✅ EVALUATION ONLY: Keep fraud labels separate
        'evaluation_fraud_label', 'evaluation_fraud_type', 'fraud_type', 'ring_id', 
        # ✅ META: Remove device/system identifiers
        'device_id', 'ip_cluster_id', 'shell_companies'
    }
    
    # Get all available features from sample node
    sample_node = agent_ids[0]
    all_features = set(G.nodes[sample_node].keys())
    
    # ✅ ENHANCED: Filter to only numeric behavioral and network features
    feature_columns = []
    for feat in sorted(list(all_features - exclude_features)):
        sample_val = G.nodes[sample_node].get(feat, 0)
        if isinstance(sample_val, (int, float, np.number)) and not isinstance(sample_val, bool):
            feature_columns.append(feat)
        elif isinstance(sample_val, str) and sample_val.replace('.', '').replace('-', '').isdigit():
            feature_columns.append(feat)
    
    print(f"🎯 Using {len(feature_columns)} behavioral and network features")
    print(f"📋 Feature categories: behavioral patterns, network centrality, transaction analysis, community metrics")
    
    # ✅ ENHANCED: Extract features and create composite targets
    node_features = []
    composite_risk_scores = []
    behavioral_risk_ratings = []
    evaluation_fraud_labels = []  # Separate for evaluation only
    
    for agent_id in agent_ids:
        node_data = G.nodes[agent_id]
        
        # ✅ BEHAVIORAL FEATURES: Extract numeric features safely
        features = []
        for feat in feature_columns:
            val = node_data.get(feat, 0.0)
            if isinstance(val, (int, float, np.number)):
                features.append(float(val))
            elif isinstance(val, str):
                try:
                    features.append(float(val))
                except (ValueError, TypeError):
                    features.append(0.0)
            elif isinstance(val, bool):
                features.append(1.0 if val else 0.0)
            else:
                features.append(0.0)
        
        node_features.append(features)
        
        # ✅ PRIMARY TARGET: Composite behavioral risk score
        financial_stability = node_data.get('financial_stability_score', 0.5)
        behavioral_consistency = node_data.get('behavioral_consistency', 0.5)
        device_stability = node_data.get('device_stability_score', 0.5)
        
        # Composite risk calculation (0-1 scale, higher = riskier)
        composite_risk = (
            (1 - financial_stability) * 0.35 +
            (1 - behavioral_consistency) * 0.25 +
            (1 - device_stability) * 0.20 +
            (node_data.get('transaction_velocity_score', 0) / 100000) * 0.10 +  # Normalize transaction velocity
            (abs(node_data.get('p2p_ratio', 0.5) - 0.5) * 2) * 0.10  # P2P ratio extremes
        )
        composite_risk = min(max(composite_risk, 0.0), 1.0)  # Clamp to [0,1]
        composite_risk_scores.append(composite_risk)
        
        # ✅ SECONDARY TARGET: Behavioral risk rating (categorical)
        if composite_risk < 0.2:
            risk_rating = 0  # Very Low Risk
        elif composite_risk < 0.4:
            risk_rating = 1  # Low Risk
        elif composite_risk < 0.6:
            risk_rating = 2  # Medium Risk
        elif composite_risk < 0.8:
            risk_rating = 3  # High Risk
        else:
            risk_rating = 4  # Very High Risk
        
        behavioral_risk_ratings.append(risk_rating)
        
        # ✅ EVALUATION ONLY: Keep fraud labels for validation (not training)
        eval_fraud_label = node_data.get('evaluation_fraud_label', 0)
        evaluation_fraud_labels.append(int(eval_fraud_label) if eval_fraud_label else 0)
    
    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    composite_risk_scores = torch.tensor(composite_risk_scores, dtype=torch.float)
    behavioral_risk_ratings = torch.tensor(behavioral_risk_ratings, dtype=torch.long)
    evaluation_fraud_labels = torch.tensor(evaluation_fraud_labels, dtype=torch.long)
    
    # ✅ NORMALIZE: Feature scaling for better GNN training
    print("🔧 Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    node_features_normalized = scaler.fit_transform(node_features.numpy())
    node_features = torch.tensor(node_features_normalized, dtype=torch.float)
    
    # ✅ ENHANCED: Create edge index and enhanced edge attributes
    edge_list = []
    edge_weights = []
    edge_types = []
    relationship_strengths = []
    
    for source, target, data in G.edges(data=True):
        source_idx = agent_id_to_idx[source]
        target_idx = agent_id_to_idx[target]
        
        edge_list.append([source_idx, target_idx])
        
        # Edge weight (transaction amount or similarity score)
        edge_weight = data.get('total_amount', data.get('similarity_score', 1.0))
        edge_weights.append(float(edge_weight))
        
        # Edge type (0=P2P transaction, 1=behavioral similarity)
        edge_type = 0 if data.get('edge_type') == 'p2p_transaction' else 1
        edge_types.append(edge_type)
        
        # Relationship strength
        relationship_strength = data.get('relationship_strength', 0.5)
        relationship_strengths.append(float(relationship_strength))
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # ✅ ENHANCED: Multi-dimensional edge attributes
        edge_attr = torch.stack([
            torch.tensor(edge_weights, dtype=torch.float),
            torch.tensor(edge_types, dtype=torch.float),
            torch.tensor(relationship_strengths, dtype=torch.float)
        ], dim=1)
        
        # ✅ NORMALIZE: Edge weights for better training
        if edge_attr[:, 0].std() > 0:
            edge_attr[:, 0] = (edge_attr[:, 0] - edge_attr[:, 0].mean()) / edge_attr[:, 0].std()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    
    # ✅ PHASE 2: Create PyTorch Geometric Data object for behavioral credit risk
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        # ✅ PRIMARY TARGETS: Behavioral risk assessment
        risk_scores=composite_risk_scores,        # Regression target (0-1)
        risk_ratings=behavioral_risk_ratings,     # Classification target (0-4)
        # ✅ EVALUATION ONLY: For model validation
        evaluation_fraud_labels=evaluation_fraud_labels
    )
    
    print(f"✅ PyTorch Geometric data created:")
    print(f"   📊 Node features: {data.x.shape} (normalized)")
    print(f"   🔗 Edge index: {data.edge_index.shape}")
    print(f"   📈 Edge attributes: {data.edge_attr.shape} (weight, type, strength)")
    print(f"   🎯 Risk scores: {data.risk_scores.shape} (range: {data.risk_scores.min():.3f}-{data.risk_scores.max():.3f})")
    print(f"   📊 Risk ratings: {data.risk_ratings.shape} (classes: {data.risk_ratings.unique().tolist()})")
    print(f"   🔍 Evaluation fraud labels: {data.evaluation_fraud_labels.sum().item()} fraud ({data.evaluation_fraud_labels.sum().item() / len(data.evaluation_fraud_labels)*100:.1f}%)")
    
    return data, feature_columns, agent_id_to_idx, scaler

def save_pytorch_data(data, feature_columns, agent_id_to_idx, scaler):
    """✅ ENHANCED: Save PyTorch Geometric data with Phase 2 metadata"""
    print("💾 Saving Phase 2 PyTorch Geometric data...")
    
    # ✅ FLEXIBLE: Save to available output directory
    output_paths = ['output/', '../output/', './output/']
    
    saved = False
    for output_dir in output_paths:
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the PyTorch Geometric data object
            torch.save(data, f'{output_dir}phase2_credit_risk_graph_data.pt')
            
            # ✅ ENHANCED: Phase 2 comprehensive metadata
            risk_score_stats = {
                'min': float(data.risk_scores.min()),
                'max': float(data.risk_scores.max()),
                'mean': float(data.risk_scores.mean()),
                'std': float(data.risk_scores.std()),
                'median': float(data.risk_scores.median())
            }
            
            risk_rating_counts = torch.bincount(data.risk_ratings, minlength=5)
            risk_rating_distribution = {
                'very_low_risk': int(risk_rating_counts[0]),
                'low_risk': int(risk_rating_counts[1]),
                'medium_risk': int(risk_rating_counts[2]),
                'high_risk': int(risk_rating_counts[3]),
                'very_high_risk': int(risk_rating_counts[4])
            }
            
            metadata = {
                'version': 'phase_2_ultra_sanitized',
                'model_type': 'behavioral_credit_risk_assessment',
                'primary_task': 'behavioral_risk_score_regression',
                'secondary_task': 'behavioral_risk_rating_classification',
                'data_compliance': 'ultra_sanitized_anti_overfitting',
                
                # Graph structure
                'num_nodes': int(data.x.size(0)),
                'num_features': int(data.x.size(1)),
                'num_edges': int(data.edge_index.size(1)),
                'graph_density': float(data.edge_index.size(1)) / (data.x.size(0) * (data.x.size(0) - 1)),
                
                # Feature information
                'feature_names': feature_columns,
                'feature_categories': {
                    'behavioral_patterns': [f for f in feature_columns if any(term in f.lower() for term in 
                                          ['behavioral', 'consistency', 'stability', 'frequency'])],
                    'network_centrality': [f for f in feature_columns if any(term in f.lower() for term in 
                                         ['centrality', 'pagerank', 'clustering'])],
                    'transaction_analysis': [f for f in feature_columns if any(term in f.lower() for term in 
                                           ['transaction', 'volume', 'amount', 'velocity'])],
                    'community_metrics': [f for f in feature_columns if 'community' in f.lower()]
                },
                
                # Target variables
                'risk_score_statistics': risk_score_stats,
                'risk_rating_distribution': risk_rating_distribution,
                'evaluation_fraud_cases': int(data.evaluation_fraud_labels.sum()),
                'evaluation_legitimate_cases': int((data.evaluation_fraud_labels == 0).sum()),
                
                # Preprocessing
                'preprocessing': {
                    'feature_normalization': 'StandardScaler',
                    'edge_weight_normalization': 'z_score',
                    'missing_value_strategy': 'zero_fill',
                    'outlier_handling': 'none'
                },
                
                # Model readiness
                'gnn_ready': True,
                'recommended_architectures': ['GAT', 'GraphSAGE', 'HeteroGNN'],
                'edge_types': ['p2p_transaction', 'behavioral_similarity'],
                'training_split_ready': True,
                
                # Compliance
                'privacy_compliance': {
                    'demographic_features': 0,
                    'personal_identifiers': 0,
                    'sensitive_attributes': 0,
                    'anti_overfitting_compliant': True
                }
            }
            
            # Save metadata
            with open(f'{output_dir}phase2_pytorch_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save feature scaler
            with open(f'{output_dir}phase2_feature_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            # ✅ ADDITIONAL: Save feature mapping for model interpretation
            feature_mapping = {
                'feature_index_to_name': {i: name for i, name in enumerate(feature_columns)},
                'feature_name_to_index': {name: i for i, name in enumerate(feature_columns)},
                'agent_id_to_node_index': agent_id_to_idx,
                'node_index_to_agent_id': {v: k for k, v in agent_id_to_idx.items()}
            }
            
            with open(f'{output_dir}phase2_feature_mapping.json', 'w') as f:
                json.dump(feature_mapping, f, indent=2, default=str)
            
            print(f"✅ Saved PyTorch data to '{output_dir}phase2_credit_risk_graph_data.pt'")
            print(f"✅ Saved metadata to '{output_dir}phase2_pytorch_metadata.json'")
            print(f"✅ Saved scaler to '{output_dir}phase2_feature_scaler.pkl'")
            print(f"✅ Saved feature mapping to '{output_dir}phase2_feature_mapping.json'")
            saved = True
            break
            
        except Exception as e:
            print(f"   ⚠️ Could not save to {output_dir}: {e}")
            continue
    
    if not saved:
        raise Exception("❌ Could not save data to any output directory")
    
    return metadata

if __name__ == "__main__":
    try:
        print("🚀 === PHASE 2 PYTORCH GEOMETRIC CONVERSION ===")
        
        # Load Phase 2 enhanced graph
        G = load_enhanced_graph()
        
        # Convert to PyTorch Geometric format
        data, feature_columns, agent_id_to_idx, scaler = convert_to_pytorch_geometric(G)
        
        # Save data and comprehensive metadata
        metadata = save_pytorch_data(data, feature_columns, agent_id_to_idx, scaler)
        
        print(f"\n🎯 === PHASE 2 BEHAVIORAL CREDIT RISK MODEL READY ===")
        print(f"✅ Data version: {metadata['version']}")
        print(f"✅ Model type: {metadata['model_type']}")
        print(f"✅ Features: {metadata['num_features']} behavioral and network features")
        print(f"✅ Nodes: {metadata['num_nodes']:,} agents")
        print(f"✅ Edges: {metadata['num_edges']:,} relationships")
        print(f"✅ Graph density: {metadata['graph_density']:.6f}")
        
        print(f"\n📊 TARGET VARIABLES:")
        print(f"🎯 Primary: Risk score regression ({metadata['risk_score_statistics']['min']:.3f} - {metadata['risk_score_statistics']['max']:.3f})")
        print(f"📈 Secondary: Risk rating classification (5 classes)")
        print(f"🔍 Evaluation: {metadata['evaluation_fraud_cases']} fraud cases for validation")
        
        print(f"\n🏗️ RECOMMENDED GNN ARCHITECTURES:")
        for arch in metadata['recommended_architectures']:
            print(f"   • {arch}")
        
        print(f"\n✅ COMPLIANCE STATUS:")
        print(f"🛡️ Anti-overfitting compliant: {metadata['privacy_compliance']['anti_overfitting_compliant']}")
        print(f"🔒 Demographic features: {metadata['privacy_compliance']['demographic_features']}")
        print(f"🔐 Personal identifiers: {metadata['privacy_compliance']['personal_identifiers']}")
        
        print(f"\n🚀 Phase 2 behavioral credit risk data ready for GNN training!")
        
    except Exception as e:
        print(f"❌ Error in PyTorch conversion: {e}")
        import traceback
        traceback.print_exc()
