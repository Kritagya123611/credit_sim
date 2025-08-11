# config.py
import numpy as np  # Import numpy for clipping

# ===================================================================
# ARCHETYPE BASE RISK SCORES  
# ===================================================================

# Assign a base risk score (0 to 1) for each archetype
ARCHETYPE_BASE_RISK = {
    "Salaried Professional": 0.15,
    "Gig Worker": 0.45,
    "Government Employee": 0.05,
    "Student": 0.65,
    "Daily Wage Laborer": 0.85,
    "Small Business Owner": 0.50,
    "Doctor": 0.10,
    "Tech Professional": 0.12,
    "Police": 0.08,
    "Senior Citizen": 0.07,
    "Delivery Agent": 0.55,
    "Lawyer": 0.25,
    "Migrant Worker": 0.90,
    "Content Creator": 0.60,
    "Homemaker": 0.75,
}

# ===================================================================
# ECONOMIC CLASSES & FINANCIAL PERSONALITIES
# ===================================================================

# Economic classes with risk modifiers and P2P behavior
ECONOMIC_CLASSES = {
    "Lower": {
        'multiplier': (0.6, 0.8), 
        'loan_propensity': 0.1, 
        'risk_mod': 1.20,
        'p2p_frequency_mod': 1.15,
        'preferred_channels': ['UPI', 'Wallet']
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
        'p2p_frequency_mod': 0.75,
        'preferred_channels': ['IMPS', 'NEFT', 'RTGS']
    }
}

# Financial personalities with risk and P2P behavior modifiers
FINANCIAL_PERSONALITIES = {
    "Saver": {
        'spend_chance_mod': 0.7, 
        'invest_chance_mod': 1.2,
        'investment_types': ["FD", "SIP", "LIC"],
        'risk_mod': 0.85,
        'p2p_trust_level': 0.8,
        'p2p_amount_mod': 0.75
    },
    "Over_Spender": {
        'spend_chance_mod': 1.5,
        'invest_chance_mod': 0.1,
        'investment_types': [],
        'risk_mod': 1.25,
        'p2p_trust_level': 1.3,
        'p2p_amount_mod': 1.4
    },
    "Rational_Investor": {
        'spend_chance_mod': 0.9,
        'invest_chance_mod': 1.5,
        'investment_types': ["Stocks", "Mutual_Funds", "SIP"],
        'risk_mod': 0.95,
        'p2p_trust_level': 1.0,
        'p2p_amount_mod': 1.0
    },
    "Risk_Addict": {
        'spend_chance_mod': 1.2,
        'invest_chance_mod': 1.8,
        'investment_types': ["Crypto", "Stocks"],
        'risk_mod': 1.30,
        'p2p_trust_level': 1.5,
        'p2p_amount_mod': 1.6
    }
}

# ===================================================================
# P2P & NETWORK CONFIGURATIONS  
# ===================================================================

# P2P Transaction Risk Factors
P2P_RISK_FACTORS = {
    'high_frequency_threshold': 10,
    'large_amount_threshold': 50000,
    'velocity_risk_threshold': 100000,
    'network_density_threshold': 0.8,
    'reciprocity_anomaly_threshold': 0.1,
    'time_window_minutes': 5,
    'fraud_ring_size_threshold': 4,
    'mule_cascade_depth': 3,
}

# Agent Network Configuration for heterogeneous graph
AGENT_NETWORK_CONFIG = {
    'max_family_connections': 4,
    'max_professional_connections': 6,
    'max_peer_connections': 8,
    # ✅ NEW: Phase 2 enhanced network limits
    'max_employer_connections': 3,
    'max_merchant_connections': 15,
    'max_business_partner_connections': 5,
    'max_educational_institution_connections': 2,
    'connection_probability': 0.15,
    'network_homophily': 0.7,
    'cross_class_connection_penalty': 0.3,
}

# ✅ NEW: Employer Relationship Configuration for Phase 2
EMPLOYER_RELATIONSHIP_CONFIG = {
    'default_config': {
        'max_employers': 3,
        'min_tenure_months': 1,
        'max_tenure_months': 240,
        'salary_consistency_threshold': 0.8,
        'salary_payment_reliability': {
            'corporate_employer': 0.98,
            'government_employer': 0.99,
            'contractor_employer': 0.75,
            'law_firm_employer': 0.92,
            'pension_provider': 0.99
        }
    },
    'archetype_specific': {
        'Salaried Professional': {'typical_employers': ['corporate_employer'], 'max_employers': 2},
        'Government Employee': {'typical_employers': ['government_employer'], 'max_employers': 1},
        'Police': {'typical_employers': ['government_employer'], 'max_employers': 1},
        'Lawyer': {'typical_employers': ['law_firm_employer'], 'max_employers': 2},
        'Migrant Worker': {'typical_employers': ['contractor_employer'], 'max_employers': 3},
        'Senior Citizen': {'typical_employers': ['pension_provider'], 'max_employers': 2},
        'Small Business Owner': {'typical_employers': ['business_entity'], 'max_employers': 1}
    }
}

# ✅ NEW: Salary Source Validation for Phase 2
SALARY_SOURCE_VALIDATION = {
    'rules': {
        'min_monthly_salary': 5000,
        'max_monthly_salary': 1000000,
        'salary_growth_rate_limit': 0.3,  # 30% max increase per year
        'payment_frequency_patterns': {
            'corporate_employer': ['monthly'],
            'government_employer': ['monthly'],
            'contractor_employer': ['weekly', 'bi_weekly', 'monthly'],
            'law_firm_employer': ['monthly', 'quarterly'],
            'pension_provider': ['monthly'],
            'business_entity': ['daily', 'weekly', 'monthly']
        }
    },
    'anomaly_detection': {
        'sudden_salary_spike_threshold': 3.0,  # 3x normal salary
        'payment_gap_threshold_days': 45,     # No salary for 45+ days
        'multiple_employer_same_day_flag': True,
        'round_number_salary_suspicion': 0.8  # 80% round numbers is suspicious
    }
}

# ✅ NEW: Device Consistency Ranges for Phase 2
DEVICE_CONSISTENCY_RANGES = {
    'Salaried Professional': {'min': 0.87, 'max': 0.98, 'precision': 3},
    'Government Employee': {'min': 0.89, 'max': 0.98, 'precision': 3},
    'Police': {'min': 0.89, 'max': 0.98, 'precision': 3},
    'Doctor': {'min': 0.85, 'max': 0.96, 'precision': 3},
    'Lawyer': {'min': 0.85, 'max': 0.95, 'precision': 3},
    'Tech Professional': {'min': 0.82, 'max': 0.94, 'precision': 3},
    'Senior Citizen': {'min': 0.96, 'max': 0.99, 'precision': 3},
    'Small Business Owner': {'min': 0.78, 'max': 0.92, 'precision': 3},
    'Student': {'min': 0.68, 'max': 0.87, 'precision': 3},
    'Gig Worker': {'min': 0.72, 'max': 0.88, 'precision': 3},
    'Delivery Agent': {'min': 0.70, 'max': 0.86, 'precision': 3},
    'Content Creator': {'min': 0.75, 'max': 0.90, 'precision': 3},
    'Migrant Worker': {'min': 0.35, 'max': 0.65, 'precision': 3},
    'Daily Wage Laborer': {'min': 0.45, 'max': 0.70, 'precision': 3},
    'Homemaker': {'min': 0.85, 'max': 0.95, 'precision': 3}
}

# Transaction Channel Configuration with enhanced Phase 2 support
TRANSACTION_CHANNELS = {
    'UPI': {
        'transaction_limit': 100000,
        'daily_limit': 1000000,
        'usage_percentage': 75,
        'fraud_risk_multiplier': 1.0,
        # ✅ NEW: Salary support
        'supports_salary': True,
        'salary_limit': 500000
    },
    'IMPS': {
        'transaction_limit': 500000,
        'daily_limit': 1000000,
        'usage_percentage': 15,
        'fraud_risk_multiplier': 0.8,
        'supports_salary': True,
        'salary_limit': 1000000
    },
    'NEFT': {
        'transaction_limit': 1000000,
        'daily_limit': 10000000,
        'usage_percentage': 5,
        'fraud_risk_multiplier': 0.6,
        'supports_salary': True,
        'salary_limit': 5000000
    },
    'RTGS': {
        'transaction_limit': 10000000,
        'daily_limit': 100000000,
        'usage_percentage': 1,
        'fraud_risk_multiplier': 0.4,
        'supports_salary': True,
        'salary_limit': 50000000
    },
    'Wallet': {
        'transaction_limit': 50000,
        'daily_limit': 200000,
        'usage_percentage': 4,
        'fraud_risk_multiplier': 1.2,
        'supports_salary': False,
        'salary_limit': 0
    },
    # ✅ NEW: Bank Transfer channel
    'Bank_Transfer': {
        'transaction_limit': 5000000,
        'daily_limit': 20000000,
        'usage_percentage': 0,  # Not counted in P2P but used for salary
        'fraud_risk_multiplier': 0.5,
        'supports_salary': True,
        'salary_limit': 10000000
    }
}

# ===================================================================
# FRAUD DETECTION & HETEROGENEOUS GRAPH FEATURES
# ===================================================================

# Fraud Detection Thresholds
FRAUD_DETECTION_THRESHOLDS = {
    'ring_detection': {
        'min_circular_transfers': 3,
        'max_time_window_hours': 24,
        'amount_consistency_threshold': 0.1,
        'member_count_threshold': 3
    },
    'mule_detection': {
        'cash_out_ratio_threshold': 0.9,
        'velocity_multiplier': 5.0,
        'dormancy_before_activity': 30,
        'cascade_detection_depth': 4
    },
    'bust_out_detection': {
        'credit_building_days': 60,
        'spending_spike_multiplier': 10.0,
        'account_age_threshold': 90,
        'depletion_ratio': 0.95
    },
    # ✅ NEW: Salary source fraud detection
    'salary_source_fraud': {
        'fake_employer_threshold': 5,  # 5+ employees with same employer is suspicious
        'salary_pattern_deviation': 0.3,  # 30% deviation from expected pattern
        'employment_verification_score': 0.7,  # Below 0.7 is suspicious
        'multiple_concurrent_employers': 3  # 3+ simultaneous employers is suspicious
    }
}

# Graph Feature Configuration for heterogeneous GNN
GRAPH_FEATURES = {
    'node_features': [
        'transaction_frequency',
        'balance_stability',
        'p2p_ratio',
        'network_centrality',
        'channel_diversity',
        'temporal_consistency',
        'amount_patterns',
        'velocity_score',
        'heterogeneous_connections',
        'employment_stability',  # ✅ ENHANCED
        'merchant_diversity',
        'device_consistency',
        # ✅ NEW: Phase 2 features
        'salary_source_consistency',
        'employer_relationship_strength',
        'income_pattern_regularity',
        'behavioral_diversity_score'
    ],
    'edge_features': [
        'transaction_frequency',
        'amount_consistency',
        'reciprocity_score',
        'temporal_patterns',
        'channel_consistency',
        'amount_growth_rate',
        'relationship_type',
        'relationship_duration',
        # ✅ NEW: Phase 2 edge features
        'employment_relationship_indicator',
        'salary_payment_indicator',
        'merchant_relationship_strength'
    ],
    'graph_features': [
        'clustering_coefficient',
        'shortest_path_lengths',
        'community_detection',
        'centrality_distribution',
        'assortativity',
        'transitivity',
        'heterogeneous_mixing',
        'temporal_evolution',
        # ✅ NEW: Phase 2 graph features
        'employer_network_structure',
        'salary_flow_patterns',
        'cross_archetype_connections'
    ]
}

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def get_risk_profile_from_score(score):
    """
    Converts numerical risk score to categorical risk profile.
    
    Args:
        score: Risk score between 0 and 1
        
    Returns:
        Risk profile category string
    """
    score = np.clip(score, 0.0, 1.0)
    
    if score < 0.10: return "Very_Low"
    if score < 0.25: return "Low"
    if score < 0.60: return "Medium"
    if score < 0.80: return "High"
    return "Very_High"

def get_preferred_p2p_channel(economic_class, financial_personality=None):
    """
    Returns preferred P2P channel based on agent profile.
    
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
        if 'UPI' in preferred_channels:
            return 'UPI'
        elif 'Wallet' in preferred_channels:
            return 'Wallet'
    elif financial_personality == 'Saver':
        if 'NEFT' in preferred_channels:
            return 'NEFT'
        elif 'IMPS' in preferred_channels:
            return 'IMPS'
    
    return preferred_channels[0] if preferred_channels else 'UPI'

def calculate_p2p_risk_modifier(agent_profile):
    """
    Calculates P2P-specific risk modifier based on agent profile.
    
    Args:
        agent_profile: Dictionary containing agent's profile attributes
        
    Returns:
        P2P risk modifier (multiplier for base risk score)
    """
    base_modifier = 1.0
    
    economic_class = agent_profile.get('economic_class', 'Middle')
    class_config = ECONOMIC_CLASSES.get(economic_class, ECONOMIC_CLASSES['Middle'])
    p2p_freq_mod = class_config.get('p2p_frequency_mod', 1.0)
    
    personality = agent_profile.get('financial_personality', 'Rational_Investor')
    personality_config = FINANCIAL_PERSONALITIES.get(personality, FINANCIAL_PERSONALITIES['Rational_Investor'])
    trust_level = personality_config.get('p2p_trust_level', 1.0)
    
    p2p_risk_modifier = base_modifier * (p2p_freq_mod * 0.5 + trust_level * 0.5)
    
    return np.clip(p2p_risk_modifier, 0.5, 2.0)

def get_transaction_limits(channel, economic_class):
    """
    Returns transaction limits based on channel and economic class.
    
    Args:
        channel: Payment channel (UPI, IMPS, etc.)
        economic_class: Agent's economic class
        
    Returns:
        Dictionary with transaction and daily limits
    """
    channel_config = TRANSACTION_CHANNELS.get(channel, TRANSACTION_CHANNELS['UPI'])
    base_txn_limit = channel_config['transaction_limit']
    base_daily_limit = channel_config['daily_limit']
    
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
        'fraud_risk_multiplier': channel_config['fraud_risk_multiplier'],
        # ✅ NEW: Salary support info
        'supports_salary': channel_config.get('supports_salary', False),
        'salary_limit': channel_config.get('salary_limit', 0)
    }

def validate_agent_configuration(agent_profile):
    """
    Validates agent configuration for consistency and realism.
    
    Args:
        agent_profile: Dictionary containing agent's profile attributes
        
    Returns:
        Boolean indicating if configuration is valid, and list of issues
    """
    issues = []
    
    required_fields = ['economic_class', 'financial_personality', 'archetype_name']
    for field in required_fields:
        if field not in agent_profile or agent_profile[field] is None:
            issues.append(f"Missing required field: {field}")
    
    if agent_profile.get('economic_class') not in ECONOMIC_CLASSES:
        issues.append(f"Invalid economic class: {agent_profile.get('economic_class')}")
    
    if agent_profile.get('financial_personality') not in FINANCIAL_PERSONALITIES:
        issues.append(f"Invalid financial personality: {agent_profile.get('financial_personality')}")
    
    if agent_profile.get('archetype_name') not in ARCHETYPE_BASE_RISK:
        issues.append(f"Invalid archetype: {agent_profile.get('archetype_name')}")
    
    risk_score = agent_profile.get('risk_score', 0)
    if not (0 <= risk_score <= 1):
        issues.append(f"Risk score out of bounds: {risk_score}")
    
    return len(issues) == 0, issues

# ✅ NEW: Phase 2 helper functions
def get_employer_relationship_config(archetype_name=None):
    """Get employer relationship configuration for archetype"""
    default_config = EMPLOYER_RELATIONSHIP_CONFIG['default_config']
    
    if archetype_name and archetype_name in EMPLOYER_RELATIONSHIP_CONFIG['archetype_specific']:
        specific_config = EMPLOYER_RELATIONSHIP_CONFIG['archetype_specific'][archetype_name]
        return {**default_config, **specific_config}
    
    return default_config

def validate_salary_source(employer_id, agent_id, amount, date):
    """Validate salary source transaction"""
    validation_rules = SALARY_SOURCE_VALIDATION['rules']
    
    if amount <= 0:
        return False, "Invalid salary amount"
    
    min_salary = validation_rules['min_monthly_salary']
    max_salary = validation_rules['max_monthly_salary']
    
    if not (min_salary <= amount <= max_salary):
        return False, f"Salary amount out of range: {min_salary}-{max_salary}"
    
    return True, "Valid salary source"

def get_device_consistency_range(archetype_name):
    """Get device consistency range for archetype"""
    return DEVICE_CONSISTENCY_RANGES.get(archetype_name, {
        'min': 0.70,
        'max': 0.95,
        'precision': 3
    })

# ===================================================================
# REALISTIC P2P STRUCTURE CLASS
# ===================================================================

class RealisticP2PStructure:
    """Enhanced P2P transaction structure for heterogeneous graphs"""
    
    @staticmethod
    def select_realistic_channel(amount=None, economic_class=None, transaction_context='p2p'):
        """Enhanced channel selection with context awareness."""
        import random
        
        # Filter channels based on context
        if transaction_context == 'salary':
            valid_channels = {k: v for k, v in TRANSACTION_CHANNELS.items() 
                            if v.get('supports_salary', False)}
        else:
            valid_channels = TRANSACTION_CHANNELS
        
        # Adjust weights based on amount and class
        base_weights = {}
        for channel, config in valid_channels.items():
            weight = config['usage_percentage']
            
            # Amount-based adjustments
            if amount:
                if amount > 100000 and channel in ['NEFT', 'RTGS']:
                    weight *= 2.0
                elif amount < 1000 and channel in ['UPI', 'Wallet']:
                    weight *= 1.5
            
            # Economic class adjustments
            if economic_class in ['High', 'Upper_Middle'] and channel in ['NEFT', 'RTGS']:
                weight *= 1.5
            elif economic_class in ['Lower', 'Lower_Middle'] and channel in ['UPI', 'Wallet']:
                weight *= 1.3
            
            base_weights[channel] = weight
        
        # Normalize and select
        total_weight = sum(base_weights.values())
        if total_weight == 0:
            return 'UPI'
        
        channels = list(base_weights.keys())
        weights = [base_weights[c] / total_weight for c in channels]
        
        return random.choices(channels, weights=weights, k=1)[0]
    
    @staticmethod
    def format_p2p_transaction(sender_id, recipient_id, channel='UPI', 
                             transaction_type='DEBIT', include_reference=False,
                             transaction_context='p2p'):
        """Enhanced P2P transaction formatting with context awareness."""
        
        if transaction_context == 'salary':
            if transaction_type == "CREDIT":
                desc = f"SAL CREDIT FROM {sender_id[:8]} via {channel}"
            else:
                desc = f"SAL CREDIT TO {recipient_id[:8]} via {channel}"
        else:
            if transaction_type == "CREDIT":
                desc = f"P2P Transfer from {sender_id[:8]} via {channel}"
            else:
                desc = f"P2P Transfer to {recipient_id[:8]} via {channel}"
        
        if include_reference:
            desc += f" Ref:{sender_id[:4]}{recipient_id[:4]}"
        
        return desc
    
    @staticmethod
    def is_p2p_transaction(description):
        """Enhanced P2P detection."""
        p2p_indicators = ["P2P Transfer", "UPI Transfer", "IMPS Transfer", "Personal Transfer"]
        return any(indicator in description for indicator in p2p_indicators)
    
    @staticmethod
    def is_salary_transaction(description):
        """Detect salary transactions."""
        salary_indicators = ["SAL CREDIT", "SALARY", "WAGE", "PAY CREDIT"]
        return any(indicator in description.upper() for indicator in salary_indicators)
    
    @staticmethod
    def extract_counterparty_id(description, transaction_type):
        """Enhanced counterparty extraction."""
        if not (RealisticP2PStructure.is_p2p_transaction(description) or 
               RealisticP2PStructure.is_salary_transaction(description)):
            return None
        
        if transaction_type == "CREDIT" and "from " in description:
            return description.split("from ")[1].split(" via")[0]
        elif transaction_type == "DEBIT" and "to " in description:
            return description.split("to ")[1].split(" via")[0]
        
        return None
    
    @staticmethod
    def validate_channel(channel):
        """Enhanced channel validation."""
        return channel in TRANSACTION_CHANNELS

# ===================================================================
# CONFIGURATION VALIDATION
# ===================================================================

def _validate_configuration():
    """Enhanced configuration validation with Phase 2 checks."""
    issues = []
    
    # Validate archetype risk scores
    for archetype in ARCHETYPE_BASE_RISK:
        if ARCHETYPE_BASE_RISK[archetype] < 0 or ARCHETYPE_BASE_RISK[archetype] > 1:
            issues.append(f"Risk score for {archetype} is out of bounds")
    
    # Validate channel percentages (excluding Bank_Transfer which is 0%)
    p2p_channels = {k: v for k, v in TRANSACTION_CHANNELS.items() if v['usage_percentage'] > 0}
    total_percentage = sum(config['usage_percentage'] for config in p2p_channels.values())
    if abs(total_percentage - 100) > 1:
        issues.append(f"P2P channel usage percentages sum to {total_percentage}%, not 100%")
    
    # Validate economic class multipliers
    for econ_class, config in ECONOMIC_CLASSES.items():
        if config['multiplier'][0] >= config['multiplier'][1]:
            issues.append(f"Invalid multiplier range for {econ_class}")
    
    # ✅ NEW: Validate device consistency ranges
    for archetype, ranges in DEVICE_CONSISTENCY_RANGES.items():
        if ranges['min'] >= ranges['max']:
            issues.append(f"Invalid device consistency range for {archetype}")
        if ranges['min'] < 0 or ranges['max'] > 1:
            issues.append(f"Device consistency out of bounds for {archetype}")
    
    if issues:
        print("Configuration validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration validation completed successfully.")

# Run validation on import
_validate_configuration()
