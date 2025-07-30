# config.py
import numpy as np # Import numpy for clipping

# NEW: Assign a base risk score (0 to 1) for each archetype
ARCHETYPE_BASE_RISK = {
    "Salaried Professional": 0.15,
    "Gig Worker / Freelancer": 0.45,
    "Government Employee": 0.05,
    "Student": 0.65,
    "Daily Wage Laborer": 0.85,
    "Small Business Owner": 0.50,
    "Doctor / Healthcare Worker": 0.10,
    "Tech Professional": 0.12,
    "Police / Security Personnel": 0.08,
    "Retired Senior Citizen": 0.07,
    "Delivery Agent / Rider": 0.55,
    "Lawyer / Consultant": 0.25,
    "Migrant Worker": 0.90,
    "Content Creator / Influencer": 0.60,
    "Homemaker": 0.75,
}


# UPDATE: Add a 'risk_mod' to each class. <1 reduces risk, >1 increases it.
ECONOMIC_CLASSES = {
    "Lower": {
        'multiplier': (0.6, 0.8), 
        'loan_propensity': 0.1, 
        'risk_mod': 1.20,
        'p2p_frequency_mod': 1.15,  # ✅ NEW: Lower class uses P2P more frequently
        'preferred_channels': ['UPI', 'Wallet']  # ✅ NEW: Channel preferences
    },
    "Lower_Middle": {
        'multiplier': (0.8, 1.0), 
        'loan_propensity': 0.2, 
        'risk_mod': 1.10,
        'p2p_frequency_mod': 1.10,
        'preferred_channels': ['UPI', 'IMPS']
    },
    "Middle": {
        'multiplier': (1.0, 1.2), 
        'loan_propensity': 0.4, 
        'risk_mod': 1.00,
        'p2p_frequency_mod': 1.00,
        'preferred_channels': ['UPI', 'IMPS', 'NEFT']
    },
    "Upper_Middle": {
        'multiplier': (1.2, 2.0), 
        'loan_propensity': 0.6, 
        'risk_mod': 0.85,
        'p2p_frequency_mod': 0.90,
        'preferred_channels': ['UPI', 'IMPS', 'NEFT', 'RTGS']
    },
    "High": {
        'multiplier': (2.0, 4.0), 
        'loan_propensity': 0.3, 
        'risk_mod': 0.70,
        'p2p_frequency_mod': 0.75,  # ✅ NEW: High class uses P2P less frequently
        'preferred_channels': ['IMPS', 'NEFT', 'RTGS']  # ✅ NEW: Prefer traditional channels
    }
}


# UPDATE: Add a 'risk_mod' to each personality.
FINANCIAL_PERSONALITIES = {
    "Saver": {
        'spend_chance_mod': 0.7, 
        'invest_chance_mod': 1.2,
        'investment_types': ["FD", "SIP", "LIC"],
        'risk_mod': 0.85,
        'p2p_trust_level': 0.8,  # ✅ NEW: Conservative with P2P transfers
        'p2p_amount_mod': 0.75   # ✅ NEW: Sends smaller amounts
    },
    "Over_Spender": {
        'spend_chance_mod': 1.5,
        'invest_chance_mod': 0.1,
        'investment_types': [],
        'risk_mod': 1.25,
        'p2p_trust_level': 1.3,  # ✅ NEW: More liberal with P2P
        'p2p_amount_mod': 1.4    # ✅ NEW: Sends larger amounts impulsively
    },
    "Rational_Investor": {
        'spend_chance_mod': 0.9,
        'invest_chance_mod': 1.5,
        'investment_types': ["Stocks", "Mutual_Funds", "SIP"],
        'risk_mod': 0.95,
        'p2p_trust_level': 1.0,  # ✅ NEW: Balanced P2P behavior
        'p2p_amount_mod': 1.0    # ✅ NEW: Rational P2P amounts
    },
    "Risk_Addict": {
        'spend_chance_mod': 1.2,
        'invest_chance_mod': 1.8,
        'investment_types': ["Crypto", "Stocks"],
        'risk_mod': 1.30,
        'p2p_trust_level': 1.5,  # ✅ NEW: High risk tolerance with P2P
        'p2p_amount_mod': 1.6    # ✅ NEW: Sends larger, riskier amounts
    }
}


# ✅ NEW: P2P Transaction Risk Factors
P2P_RISK_FACTORS = {
    'high_frequency_threshold': 10,      # >10 P2P transactions per day is suspicious
    'large_amount_threshold': 50000,     # >₹50K single P2P transfer is high risk
    'velocity_risk_threshold': 100000,   # >₹1L P2P volume per day is very high risk
    'network_density_threshold': 0.8,    # Dense networks (>0.8 clustering) are suspicious
    'reciprocity_anomaly_threshold': 0.1, # <0.1 reciprocity ratio is suspicious
    'time_window_minutes': 5,            # Multiple transfers within 5 minutes is suspicious
    'fraud_ring_size_threshold': 4,      # Fraud rings typically have 4+ members
    'mule_cascade_depth': 3,             # Money flows through 3+ intermediaries
}


# ✅ NEW: Agent Network Configuration
AGENT_NETWORK_CONFIG = {
    'max_family_connections': 4,         # Maximum family network size
    'max_professional_connections': 6,   # Maximum professional network size
    'max_peer_connections': 8,           # Maximum peer network size
    'connection_probability': 0.15,      # Base probability of agent connections
    'network_homophily': 0.7,           # Tendency to connect with similar agents
    'cross_class_connection_penalty': 0.3, # Reduced probability across economic classes
}


# ✅ NEW: Transaction Channel Configuration
TRANSACTION_CHANNELS = {
    'UPI': {
        'transaction_limit': 100000,     # ₹1L per transaction
        'daily_limit': 1000000,         # ₹10L per day
        'usage_percentage': 75,         # 75% of P2P transactions
        'fraud_risk_multiplier': 1.0    # Baseline fraud risk
    },
    'IMPS': {
        'transaction_limit': 500000,    # ₹5L per transaction
        'daily_limit': 1000000,        # ₹10L per day
        'usage_percentage': 15,        # 15% of P2P transactions
        'fraud_risk_multiplier': 0.8   # Lower fraud risk
    },
    'NEFT': {
        'transaction_limit': 1000000,   # ₹10L per transaction
        'daily_limit': 10000000,       # ₹1Cr per day
        'usage_percentage': 5,         # 5% of P2P transactions
        'fraud_risk_multiplier': 0.6   # Lower fraud risk (traditional)
    },
    'RTGS': {
        'transaction_limit': 10000000,  # ₹1Cr per transaction
        'daily_limit': 100000000,      # ₹10Cr per day
        'usage_percentage': 1,         # 1% of P2P transactions
        'fraud_risk_multiplier': 0.4   # Lowest fraud risk (high scrutiny)
    },
    'Wallet': {
        'transaction_limit': 50000,     # ₹50K per transaction
        'daily_limit': 200000,         # ₹2L per day
        'usage_percentage': 4,         # 4% of P2P transactions
        'fraud_risk_multiplier': 1.2   # Higher fraud risk (less regulated)
    }
}


# ✅ NEW: Fraud Detection Thresholds
FRAUD_DETECTION_THRESHOLDS = {
    'ring_detection': {
        'min_circular_transfers': 3,        # Minimum transfers in a ring pattern
        'max_time_window_hours': 24,       # Time window for detecting rings
        'amount_consistency_threshold': 0.1, # Similar amounts indicate coordination
        'member_count_threshold': 3         # Minimum members for fraud ring
    },
    'mule_detection': {
        'cash_out_ratio_threshold': 0.9,   # >90% of incoming funds cashed out
        'velocity_multiplier': 5.0,        # 5x normal transaction velocity
        'dormancy_before_activity': 30,    # Days dormant before sudden activity
        'cascade_detection_depth': 4       # Track money flows 4 levels deep
    },
    'bust_out_detection': {
        'credit_building_days': 60,        # Days of normal behavior before bust out
        'spending_spike_multiplier': 10.0, # 10x spending increase indicates bust out
        'account_age_threshold': 90,       # Minimum account age for bust out pattern
        'depletion_ratio': 0.95           # >95% balance depletion in short time
    }
}


# ✅ NEW: Graph Feature Configuration
GRAPH_FEATURES = {
    'node_features': [
        'transaction_frequency',           # Number of transactions per time period
        'balance_stability',              # Variance in account balance
        'p2p_ratio',                     # Ratio of P2P to total transactions
        'network_centrality',            # Position importance in network
        'channel_diversity',             # Number of different channels used
        'temporal_consistency',          # Regularity of transaction patterns
        'amount_patterns',               # Statistical features of transaction amounts
        'velocity_score'                 # Speed of money movement
    ],
    'edge_features': [
        'transaction_frequency',         # How often agents transfer to each other
        'amount_consistency',           # Consistency of transfer amounts
        'reciprocity_score',            # Bidirectional transfer ratio
        'temporal_patterns',            # Time-based transfer patterns
        'channel_consistency',          # Consistency of channels used
        'amount_growth_rate'            # Change in transfer amounts over time
    ],
    'graph_features': [
        'clustering_coefficient',       # Network clustering density
        'shortest_path_lengths',        # Average shortest paths
        'community_detection',          # Community structure strength
        'centrality_distribution',     # Distribution of node centralities
        'assortativity',               # Tendency of similar nodes to connect
        'transitivity'                 # Triangle formation tendency
    ]
}


# NEW: A helper function to convert the final score back to a category
def get_risk_profile_from_score(score):
    """
    Converts numerical risk score to categorical risk profile.
    
    Args:
        score: Risk score between 0 and 1
        
    Returns:
        Risk profile category string
    """
    score = np.clip(score, 0.0, 1.0)  # Ensure score is within bounds
    
    if score < 0.10: return "Very_Low"
    if score < 0.25: return "Low"
    if score < 0.60: return "Medium"
    if score < 0.80: return "High"
    return "Very_High"


def get_preferred_p2p_channel(economic_class, financial_personality=None):
    """
    ✅ NEW: Returns preferred P2P channel based on agent profile.
    
    Args:
        economic_class: Agent's economic class
        financial_personality: Agent's financial personality (optional)
        
    Returns:
        Preferred P2P channel string
    """
    class_config = ECONOMIC_CLASSES.get(economic_class, ECONOMIC_CLASSES['Middle'])
    preferred_channels = class_config.get('preferred_channels', ['UPI'])
    
    # Adjust based on personality if provided
    if financial_personality == 'Risk_Addict':
        # Risk addicts prefer newer, faster channels
        if 'UPI' in preferred_channels:
            return 'UPI'
        elif 'Wallet' in preferred_channels:
            return 'Wallet'
    elif financial_personality == 'Saver':
        # Savers prefer traditional, secure channels
        if 'NEFT' in preferred_channels:
            return 'NEFT'
        elif 'IMPS' in preferred_channels:
            return 'IMPS'
    
    # Default to most common channel in class
    return preferred_channels[0] if preferred_channels else 'UPI'


def calculate_p2p_risk_modifier(agent_profile):
    """
    ✅ NEW: Calculates P2P-specific risk modifier based on agent profile.
    
    Args:
        agent_profile: Dictionary containing agent's profile attributes
        
    Returns:
        P2P risk modifier (multiplier for base risk score)
    """
    base_modifier = 1.0
    
    # Economic class impact
    economic_class = agent_profile.get('economic_class', 'Middle')
    class_config = ECONOMIC_CLASSES.get(economic_class, ECONOMIC_CLASSES['Middle'])
    p2p_freq_mod = class_config.get('p2p_frequency_mod', 1.0)
    
    # Personality impact
    personality = agent_profile.get('financial_personality', 'Rational_Investor')
    personality_config = FINANCIAL_PERSONALITIES.get(personality, FINANCIAL_PERSONALITIES['Rational_Investor'])
    trust_level = personality_config.get('p2p_trust_level', 1.0)
    
    # Calculate combined modifier
    p2p_risk_modifier = base_modifier * (p2p_freq_mod * 0.5 + trust_level * 0.5)
    
    return np.clip(p2p_risk_modifier, 0.5, 2.0)  # Keep within reasonable bounds


def get_transaction_limits(channel, economic_class):
    """
    ✅ NEW: Returns transaction limits based on channel and economic class.
    
    Args:
        channel: Payment channel (UPI, IMPS, etc.)
        economic_class: Agent's economic class
        
    Returns:
        Dictionary with transaction and daily limits
    """
    channel_config = TRANSACTION_CHANNELS.get(channel, TRANSACTION_CHANNELS['UPI'])
    base_txn_limit = channel_config['transaction_limit']
    base_daily_limit = channel_config['daily_limit']
    
    # Adjust limits based on economic class
    class_multiplier = {
        'Lower': 0.5,
        'Lower_Middle': 0.7,
        'Middle': 1.0,
        'Upper_Middle': 1.5,
        'High': 2.0
    }.get(economic_class, 1.0)
    
    return {
        'transaction_limit': int(base_txn_limit * class_multiplier),
        'daily_limit': int(base_daily_limit * class_multiplier),
        'fraud_risk_multiplier': channel_config['fraud_risk_multiplier']
    }


def validate_agent_configuration(agent_profile):
    """
    ✅ NEW: Validates agent configuration for consistency and realism.
    
    Args:
        agent_profile: Dictionary containing agent's profile attributes
        
    Returns:
        Boolean indicating if configuration is valid, and list of issues
    """
    issues = []
    
    # Check required fields
    required_fields = ['economic_class', 'financial_personality', 'archetype_name']
    for field in required_fields:
        if field not in agent_profile or agent_profile[field] is None:
            issues.append(f"Missing required field: {field}")
    
    # Validate economic class
    if agent_profile.get('economic_class') not in ECONOMIC_CLASSES:
        issues.append(f"Invalid economic class: {agent_profile.get('economic_class')}")
    
    # Validate financial personality
    if agent_profile.get('financial_personality') not in FINANCIAL_PERSONALITIES:
        issues.append(f"Invalid financial personality: {agent_profile.get('financial_personality')}")
    
    # Validate archetype
    if agent_profile.get('archetype_name') not in ARCHETYPE_BASE_RISK:
        issues.append(f"Invalid archetype: {agent_profile.get('archetype_name')}")
    
    # Check risk score bounds
    risk_score = agent_profile.get('risk_score', 0)
    if not (0 <= risk_score <= 1):
        issues.append(f"Risk score out of bounds: {risk_score}")
    
    return len(issues) == 0, issues


# ✅ NEW: Configuration validation on import
def _validate_configuration():
    """Validates the configuration for internal consistency."""
    # Ensure all archetypes have corresponding risk scores
    for archetype in ARCHETYPE_BASE_RISK:
        if archetype not in ARCHETYPE_BASE_RISK:
            print(f"Warning: {archetype} missing from ARCHETYPE_BASE_RISK")
    
    # Ensure channel percentages add up to 100%
    total_percentage = sum(config['usage_percentage'] for config in TRANSACTION_CHANNELS.values())
    if abs(total_percentage - 100) > 1:  # Allow 1% tolerance
        print(f"Warning: Channel usage percentages sum to {total_percentage}%, not 100%")
    
    print("Configuration validation completed.")

class RealisticP2PStructure:
    @staticmethod
    def select_realistic_channel():
        """Select channel based on your TRANSACTION_CHANNELS configuration."""
        import random
        channels = list(TRANSACTION_CHANNELS.keys())
        weights = [TRANSACTION_CHANNELS[ch]['usage_percentage'] for ch in channels]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        return random.choices(channels, weights=normalized_weights, k=1)[0]
    
# Run validation on import
_validate_configuration()
