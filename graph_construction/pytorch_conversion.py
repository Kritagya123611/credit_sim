import pickle
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def load_enhanced_graph():
    """Load the enhanced graph with all features"""
    print("Loading enhanced graph...")
    
    # ✅ UPDATED: Flexible path handling for different execution contexts
    paths_to_try = [
        '../output/enhanced_graph.pkl',
        'output/enhanced_graph.pkl',
        './output/enhanced_graph.pkl'
    ]
    
    G = None
    for path in paths_to_try:
        try:
            with open(path, 'rb') as f:
                G = pickle.load(f)
            print(f"Loaded graph from {path}")
            break
        except FileNotFoundError:
            continue
    
    if G is None:
        raise FileNotFoundError("Could not find enhanced_graph.pkl in any expected location")
    
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def convert_to_pytorch_geometric(G):
    """Convert enhanced NetworkX graph to PyTorch Geometric format for credit risk"""
    print("Converting to PyTorch Geometric format for credit risk assessment...")
    
    # Extract node features and labels
    agent_ids = list(G.nodes())
    agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    
    # ✅ UPDATED: Exclude meta features and fraud-specific features (keep for validation only)
    exclude_features = {'agent_id', 'community', 'fraud_type', 'ring_id', 'device_id'}
    
    # Get all available features from first node
    sample_node = agent_ids[0]
    all_features = set(G.nodes[sample_node].keys())
    feature_columns = sorted(list(all_features - exclude_features))
    
    print(f"Using {len(feature_columns)} features: {feature_columns[:10]}...")
    
    # Extract node features and credit risk labels
    node_features = []
    risk_scores = []
    risk_ratings = []
    fraud_labels = []  # Keep for validation
    
    for agent_id in agent_ids:
        # ✅ UPDATED: Extract clean behavioral features (no keywords)
        features = []
        for feat in feature_columns:
            val = G.nodes[agent_id].get(feat, 0)
            # Handle any non-numeric values
            if isinstance(val, (int, float, np.number)):
                features.append(float(val))
            else:
                features.append(0.0)
        
        node_features.append(features)
        
        # ✅ UPDATED: Primary target - Credit risk score (continuous)
        risk_score = G.nodes[agent_id].get('risk_score', 0.5)
        risk_scores.append(float(risk_score))
        
        # ✅ UPDATED: Secondary target - Risk profile/rating (categorical)
        risk_profile = G.nodes[agent_id].get('risk_profile', 'Medium')
        rating_map = {
            'Very_Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4,
            'Very Low': 0, 'Low Risk': 1, 'Medium Risk': 2, 'High Risk': 3, 'Very High': 4
        }
        risk_rating = rating_map.get(risk_profile, 2)  # Default to Medium
        risk_ratings.append(risk_rating)
        
        # ✅ UPDATED: Keep fraud labels for validation (not primary training target)
        fraud_type = G.nodes[agent_id].get('fraud_type')
        fraud_label = 1 if fraud_type and pd.notna(fraud_type) and fraud_type in ['ring', 'mule', 'bust_out'] else 0
        fraud_labels.append(fraud_label)
    
    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    risk_scores = torch.tensor(risk_scores, dtype=torch.float)
    risk_ratings = torch.tensor(risk_ratings, dtype=torch.long)
    fraud_labels = torch.tensor(fraud_labels, dtype=torch.long)
    
    # ✅ UPDATED: Normalize features for better GNN training
    scaler = StandardScaler()
    node_features = torch.tensor(scaler.fit_transform(node_features), dtype=torch.float)
    
    # Create edge index and edge attributes
    edge_list = []
    edge_weights = []
    edge_consistencies = []
    
    for source, target, data in G.edges(data=True):
        source_idx = agent_id_to_idx[source]
        target_idx = agent_id_to_idx[target]
        
        edge_list.append([source_idx, target_idx])
        edge_weights.append(data.get('total_amount', 1.0))
        edge_consistencies.append(data.get('amount_consistency', 0.5))
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack([
            torch.tensor(edge_weights, dtype=torch.float),
            torch.tensor(edge_consistencies, dtype=torch.float)
        ], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)
    
    # ✅ UPDATED: Create PyTorch Geometric Data object for credit risk
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        risk_scores=risk_scores,      # Primary target (regression)
        risk_ratings=risk_ratings,    # Secondary target (classification)
        fraud_labels=fraud_labels     # Validation target
    )
    
    print(f"PyTorch Geometric data created:")
    print(f"  Node features: {data.x.shape}")
    print(f"  Edge index: {data.edge_index.shape}")
    print(f"  Edge attributes: {data.edge_attr.shape}")
    print(f"  Risk scores: {data.risk_scores.shape} (range: {data.risk_scores.min():.3f}-{data.risk_scores.max():.3f})")
    print(f"  Risk ratings: {data.risk_ratings.shape} (classes: {data.risk_ratings.unique().tolist()})")
    print(f"  Fraud validation: {data.fraud_labels.sum().item()} fraud cases ({data.fraud_labels.sum().item() / len(data.fraud_labels):.3f})")
    
    return data, feature_columns, agent_id_to_idx, scaler

def save_pytorch_data(data, feature_columns, agent_id_to_idx, scaler):
    """Save PyTorch Geometric data and metadata for credit risk model"""
    print("Saving PyTorch Geometric data...")
    
    # ✅ UPDATED: Save to multiple possible output paths
    output_paths = ['../output/', 'output/', './output/']
    
    saved = False
    for output_dir in output_paths:
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the data object
            torch.save(data, f'{output_dir}graph_data.pt')
            
            # ✅ UPDATED: Save credit risk specific metadata
            metadata = {
                'model_type': 'credit_risk_assessment',
                'primary_task': 'risk_score_regression',
                'secondary_task': 'risk_rating_classification',
                'num_nodes': data.x.size(0),
                'num_features': data.x.size(1),
                'num_edges': data.edge_index.size(1),
                'feature_names': feature_columns,
                'agent_id_to_idx': agent_id_to_idx,
                'risk_score_stats': {
                    'min': float(data.risk_scores.min()),
                    'max': float(data.risk_scores.max()),
                    'mean': float(data.risk_scores.mean()),
                    'std': float(data.risk_scores.std())
                },
                'risk_rating_distribution': {
                    int(k): int(v) for k, v in 
                    zip(*torch.unique(data.risk_ratings, return_counts=True))
                },
                'fraud_validation_count': int(data.fraud_labels.sum()),
                'legitimate_count': int((data.fraud_labels == 0).sum()),
                'preprocessing': {
                    'feature_scaling': 'StandardScaler',
                    'missing_value_fill': 0.0
                }
            }
            
            import json
            with open(f'{output_dir}pytorch_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # ✅ NEW: Save scaler for inference
            import pickle
            with open(f'{output_dir}feature_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"Saved PyTorch data to '{output_dir}graph_data.pt'")
            print(f"Saved metadata to '{output_dir}pytorch_metadata.json'")
            print(f"Saved scaler to '{output_dir}feature_scaler.pkl'")
            saved = True
            break
            
        except Exception as e:
            continue
    
    if not saved:
        raise Exception("Could not save data to any output directory")
    
    return metadata

if __name__ == "__main__":
    # ✅ UPDATED: Credit risk focused conversion pipeline
    print("=== Credit Risk PyTorch Geometric Conversion ===")
    
    # Load enhanced graph
    G = load_enhanced_graph()
    
    # Convert to PyTorch Geometric format
    data, feature_columns, agent_id_to_idx, scaler = convert_to_pytorch_geometric(G)
    
    # Save data and metadata
    metadata = save_pytorch_data(data, feature_columns, agent_id_to_idx, scaler)
    
    print(f"\n=== Credit Risk Model Ready ===")
    print(f"✅ Ready for GNN training with {metadata['num_features']} features")
    print(f"✅ Primary target: Risk score regression ({metadata['risk_score_stats']['min']:.3f} - {metadata['risk_score_stats']['max']:.3f})")
    print(f"✅ Secondary target: Risk rating classification ({len(metadata['risk_rating_distribution'])} classes)")
    print(f"✅ Graph structure: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges")
    print(f"✅ Validation: {metadata['fraud_validation_count']} fraud cases for model comparison")
    print(f"✅ Model type: {metadata['model_type']}")
