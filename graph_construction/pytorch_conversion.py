import pickle
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def load_enhanced_graph():
    """Load the enhanced graph with all features"""
    print("Loading enhanced graph...")
    with open('../output/enhanced_graph.pkl', 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def convert_to_pytorch_geometric(G):
    """Convert enhanced NetworkX graph to PyTorch Geometric format"""
    print("Converting to PyTorch Geometric format...")
    
    # Extract node features and labels
    agent_ids = list(G.nodes())
    agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    
    # Define clean feature columns (exclude identifiers and meta info)
    exclude_features = {'agent_id', 'fraud_type', 'community'}  # Meta features
    
    # Get all available features from first node
    sample_node = agent_ids[0]
    all_features = set(G.nodes[sample_node].keys())
    feature_columns = sorted(list(all_features - exclude_features))
    
    print(f"Using {len(feature_columns)} features: {feature_columns[:10]}...")
    
    # Extract node features
    node_features = []
    node_labels = []
    
    for agent_id in agent_ids:
        # Features (numerical values only)
        features = []
        for feat in feature_columns:
            val = G.nodes[agent_id].get(feat, 0)
            # Handle any non-numeric values
            if isinstance(val, (int, float)):
                features.append(float(val))
            else:
                features.append(0.0)
        
        node_features.append(features)
        
        # Labels (fraud detection)
        fraud_type = G.nodes[agent_id].get('fraud_type')
        label = 1 if fraud_type and pd.notna(fraud_type) and fraud_type in ['ring', 'mule', 'bust_out'] else 0
        node_labels.append(label)
    
    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    node_labels = torch.tensor(node_labels, dtype=torch.long)
    
    # Normalize features
    scaler = StandardScaler()
    node_features = torch.tensor(scaler.fit_transform(node_features), dtype=torch.float)
    
    # Create edge index and edge attributes
    edge_list = []
    edge_weights = []
    
    for source, target, data in G.edges(data=True):
        source_idx = agent_id_to_idx[source]
        target_idx = agent_id_to_idx[target]
        
        edge_list.append([source_idx, target_idx])
        edge_weights.append(data.get('total_amount', 1.0))
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=node_labels
    )
    
    print(f"PyTorch Geometric data created:")
    print(f"  Node features: {data.x.shape}")
    print(f"  Edge index: {data.edge_index.shape}")
    print(f"  Edge attributes: {data.edge_attr.shape}")
    print(f"  Labels: {data.y.shape}")
    print(f"  Fraud ratio: {data.y.sum().item() / len(data.y):.3f}")
    
    return data, feature_columns, agent_id_to_idx

def save_pytorch_data(data, feature_columns, agent_id_to_idx):
    """Save PyTorch Geometric data and metadata"""
    print("Saving PyTorch Geometric data...")
    
    # Save the data object
    torch.save(data, '../output/graph_data.pt')
    
    # Save metadata
    metadata = {
        'num_nodes': data.x.size(0),
        'num_features': data.x.size(1),
        'num_edges': data.edge_index.size(1),
        'feature_names': feature_columns,
        'agent_id_to_idx': agent_id_to_idx,
        'fraud_count': data.y.sum().item(),
        'legitimate_count': (data.y == 0).sum().item()
    }
    
    import json
    with open('../output/pytorch_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Saved PyTorch data to '../output/graph_data.pt'")
    print(f"Saved metadata to '../output/pytorch_metadata.json'")
    
    return metadata

if __name__ == "__main__":
    # Load enhanced graph
    G = load_enhanced_graph()
    
    # Convert to PyTorch Geometric
    data, feature_columns, agent_id_to_idx = convert_to_pytorch_geometric(G)
    
    # Save data and metadata
    metadata = save_pytorch_data(data, feature_columns, agent_id_to_idx)
    
    print(f"\n=== PyTorch Geometric Conversion Complete ===")
    print(f"✅ Ready for GNN training with {metadata['num_features']} features")
    print(f"✅ {metadata['fraud_count']} fraud nodes, {metadata['legitimate_count']} legitimate nodes")
    print(f"✅ Graph structure: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges")
