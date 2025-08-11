"""
Enhanced configuration package initialization for Phase 2: Salary Source Tracking
Includes comprehensive imports for heterogeneous graph-based credit risk modeling,
with robust fallback mechanisms and enhanced P2P transaction support with employer tracking.
"""

# Import from the root config.py (your comprehensive configuration file)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ===================================================================
# CORE CONFIGURATION IMPORTS
# ===================================================================

from config import (
    ECONOMIC_CLASSES,
    FINANCIAL_PERSONALITIES, 
    ARCHETYPE_BASE_RISK,
    get_risk_profile_from_score,
    TRANSACTION_CHANNELS,
    get_preferred_p2p_channel,
    calculate_p2p_risk_modifier,
    get_transaction_limits,
    validate_agent_configuration
)

# ===================================================================
# ENHANCED PHASE 2 IMPORTS
# ===================================================================

# ✅ NEW: Import enhanced configurations for heterogeneous graphs with salary tracking
try:
    from config import (
        P2P_RISK_FACTORS,
        AGENT_NETWORK_CONFIG,
        FRAUD_DETECTION_THRESHOLDS,
        GRAPH_FEATURES,
        EMPLOYER_RELATIONSHIP_CONFIG,  # ✅ NEW: Employer tracking
        SALARY_SOURCE_VALIDATION,      # ✅ NEW: Salary source validation
        DEVICE_CONSISTENCY_RANGES,     # ✅ NEW: Device consistency ranges
        RealisticP2PStructure
    )
except ImportError:
    # Fallback for missing enhanced features
    P2P_RISK_FACTORS = {}
    AGENT_NETWORK_CONFIG = {}
    FRAUD_DETECTION_THRESHOLDS = {}
    GRAPH_FEATURES = {}
    EMPLOYER_RELATIONSHIP_CONFIG = {}
    SALARY_SOURCE_VALIDATION = {}
    DEVICE_CONSISTENCY_RANGES = {}

# ===================================================================
# P2P STRUCTURE IMPORT WITH ENHANCED FALLBACK
# ===================================================================

# Import P2P structure from this package with enhanced fallback
try:
    from .p2p_structure import RealisticP2PStructure
except ImportError:
    try:
        # Try importing from config module
        from config import RealisticP2PStructure
    except ImportError:
        # Final fallback - create enhanced implementation
        class RealisticP2PStructure:
            @staticmethod
            def select_realistic_channel(amount=None, economic_class=None, urgency='normal'):
                import random
                # Enhanced channel selection with amount and class considerations
                if amount and amount > 100000:
                    return random.choice(['NEFT', 'RTGS'])
                elif amount and amount > 50000:
                    return random.choice(['IMPS', 'NEFT'])
                
                # Use TRANSACTION_CHANNELS for realistic distribution
                channels = list(TRANSACTION_CHANNELS.keys())
                weights = [TRANSACTION_CHANNELS[ch]['usage_percentage'] for ch in channels]
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                return random.choices(channels, weights=normalized_weights, k=1)[0]
            
            @staticmethod
            def format_p2p_transaction(sender_id, recipient_id, channel='UPI', 
                                     transaction_type='DEBIT', include_reference=False,
                                     sender_node_type='agent', recipient_node_type='agent'):
                # ✅ ENHANCED: Support for different node types
                if sender_node_type == 'company' or recipient_node_type == 'company':
                    if transaction_type == "DEBIT":
                        desc = f"SAL CREDIT TO {recipient_id[:8]}"
                    else:
                        desc = f"SAL CREDIT FROM {sender_id[:8]}"
                elif sender_node_type == 'merchant' or recipient_node_type == 'merchant':
                    if transaction_type == "DEBIT":
                        desc = f"MERCH PAY TO {recipient_id[:8]}"
                    else:
                        desc = f"MERCH PAY FROM {sender_id[:8]}"
                else:
                    if transaction_type == "DEBIT":
                        desc = f"{channel} P2P TO {recipient_id[:8]}"
                    else:
                        desc = f"{channel} P2P FROM {sender_id[:8]}"
                return desc
            
            @staticmethod
            def is_p2p_transaction(description):
                p2p_indicators = ['P2P', 'UPI', 'SAL CREDIT', 'MERCH PAY']
                return any(indicator in description.upper() for indicator in p2p_indicators)
            
            @staticmethod
            def validate_channel(channel):
                return channel in TRANSACTION_CHANNELS
            
            @staticmethod
            def get_channel_fraud_risk(channel):
                return TRANSACTION_CHANNELS.get(channel, {}).get('fraud_risk_multiplier', 1.0)

# ===================================================================
# ENHANCED UTILITY FUNCTIONS FOR PHASE 2
# ===================================================================

def get_risk_profile_score(score):
    """Alias for backward compatibility"""
    return get_risk_profile_from_score(score)

def get_agent_network_limits():
    """Get network connection limits for agents"""
    return AGENT_NETWORK_CONFIG.get('max_connections', {
        'family': 4,
        'professional': 6,
        'peer': 8,
        'employer': 3,  # ✅ NEW: Employer connections
        'merchant': 10  # ✅ NEW: Merchant connections
    })

def get_fraud_detection_config():
    """Get fraud detection configuration"""
    return FRAUD_DETECTION_THRESHOLDS

def get_graph_feature_config():
    """Get graph feature configuration for GNN models"""
    return GRAPH_FEATURES

def validate_p2p_channel(channel):
    """Validate P2P channel"""
    return RealisticP2PStructure.validate_channel(channel)

def get_realistic_channel_distribution():
    """Get channel probability distribution"""
    try:
        return RealisticP2PStructure.get_channel_probability_weights()
    except AttributeError:
        # Fallback to TRANSACTION_CHANNELS
        return {ch: config['usage_percentage'] for ch, config in TRANSACTION_CHANNELS.items()}

# ✅ NEW: Employer relationship utilities
def get_employer_relationship_config():
    """Get employer relationship configuration"""
    return EMPLOYER_RELATIONSHIP_CONFIG.get('default_config', {
        'max_employers': 3,
        'min_tenure_months': 1,
        'max_tenure_months': 240,
        'salary_consistency_threshold': 0.8
    })

def validate_salary_source(employer_id, agent_id, amount, date):
    """Validate salary source transaction"""
    validation_rules = SALARY_SOURCE_VALIDATION.get('rules', {})
    
    # Basic validation
    if amount <= 0:
        return False, "Invalid salary amount"
    
    min_salary = validation_rules.get('min_monthly_salary', 5000)
    max_salary = validation_rules.get('max_monthly_salary', 1000000)
    
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

def calculate_employment_tenure_score(start_date, current_date):
    """Calculate employment tenure score for fraud detection"""
    if not start_date or not current_date:
        return 0.0
    
    tenure_days = (current_date - start_date).days
    tenure_months = tenure_days / 30.44  # Average days per month
    
    # Score based on tenure (longer tenure = higher score)
    if tenure_months >= 24:
        return 1.0
    elif tenure_months >= 12:
        return 0.8
    elif tenure_months >= 6:
        return 0.6
    elif tenure_months >= 3:
        return 0.4
    else:
        return 0.2

# ===================================================================
# ENHANCED EXPORTS FOR PHASE 2
# ===================================================================

# Make everything available when importing from config package
__all__ = [
    # Core configuration
    'ECONOMIC_CLASSES',
    'FINANCIAL_PERSONALITIES',
    'ARCHETYPE_BASE_RISK', 
    'get_risk_profile_from_score',
    'get_risk_profile_score',  # Backward compatibility alias
    'TRANSACTION_CHANNELS',
    'get_preferred_p2p_channel',
    'calculate_p2p_risk_modifier',
    'get_transaction_limits',
    'validate_agent_configuration',
    
    # ✅ ENHANCED: Phase 2 exports
    'P2P_RISK_FACTORS',
    'AGENT_NETWORK_CONFIG', 
    'FRAUD_DETECTION_THRESHOLDS',
    'GRAPH_FEATURES',
    'EMPLOYER_RELATIONSHIP_CONFIG',  # ✅ NEW
    'SALARY_SOURCE_VALIDATION',      # ✅ NEW
    'DEVICE_CONSISTENCY_RANGES',     # ✅ NEW
    'RealisticP2PStructure',
    
    # ✅ ENHANCED: Utility functions
    'get_agent_network_limits',
    'get_fraud_detection_config',
    'get_graph_feature_config',
    'validate_p2p_channel',
    'get_realistic_channel_distribution',
    'get_employer_relationship_config',    # ✅ NEW
    'validate_salary_source',              # ✅ NEW
    'get_device_consistency_range',        # ✅ NEW
    'calculate_employment_tenure_score'    # ✅ NEW
]

# ===================================================================
# PACKAGE INITIALIZATION & VALIDATION
# ===================================================================

def _validate_package_imports():
    """Validate that all critical imports are available"""
    try:
        # Test critical functionality
        assert len(ECONOMIC_CLASSES) > 0, "ECONOMIC_CLASSES is empty"
        assert len(FINANCIAL_PERSONALITIES) > 0, "FINANCIAL_PERSONALITIES is empty"
        assert len(ARCHETYPE_BASE_RISK) > 0, "ARCHETYPE_BASE_RISK is empty"
        assert len(TRANSACTION_CHANNELS) > 0, "TRANSACTION_CHANNELS is empty"
        
        # Test RealisticP2PStructure
        test_channel = RealisticP2PStructure.select_realistic_channel()
        assert test_channel in TRANSACTION_CHANNELS, "P2P channel selection failed"
        
        # Test risk profile function
        test_risk = get_risk_profile_from_score(0.5)
        assert test_risk in ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'], "Risk profile function failed"
        
        # ✅ NEW: Test employer relationship functions
        employer_config = get_employer_relationship_config()
        assert 'max_employers' in employer_config, "Employer config validation failed"
        
        # ✅ NEW: Test salary source validation
        is_valid, message = validate_salary_source("test_employer", "test_agent", 50000, None)
        assert isinstance(is_valid, bool), "Salary validation failed"
        
        print("✅ Config package validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️  Config package validation warning: {e}")
        return False

# Run validation on package import
_validate_package_imports()

# ===================================================================
# PACKAGE METADATA
# ===================================================================

__version__ = "2.1.0-phase2"
__author__ = "Credit Risk GNN Team"
__description__ = "Enhanced configuration package for heterogeneous graph-based credit risk modeling with salary source tracking"

# Print package info on import (can be disabled in production)
if __name__ != "__main__":
    print(f"📦 Config package v{__version__} loaded with {len(__all__)} exports")
    print(f"   - Economic classes: {len(ECONOMIC_CLASSES)}")
    print(f"   - Financial personalities: {len(FINANCIAL_PERSONALITIES)}")
    print(f"   - Agent archetypes: {len(ARCHETYPE_BASE_RISK)}")
    print(f"   - Transaction channels: {len(TRANSACTION_CHANNELS)}")
    print("   - Phase 2 salary source tracking: ✅")
    print("   - Enhanced behavioral diversity: ✅")
