# config/__init__.py

# Import from the root config.py (your comprehensive configuration file)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

# Import P2P structure from this package
try:
    from .p2p_structure import RealisticP2PStructure
except ImportError:
    # Fallback if p2p_structure doesn't exist
    class RealisticP2PStructure:
        @staticmethod
        def select_realistic_channel():
            import random
            # Use your TRANSACTION_CHANNELS for realistic distribution
            channels = list(TRANSACTION_CHANNELS.keys())
            weights = [TRANSACTION_CHANNELS[ch]['usage_percentage'] for ch in channels]
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            return random.choices(channels, weights=normalized_weights, k=1)[0]

# Make everything available when importing from config package
__all__ = [
    'ECONOMIC_CLASSES',
    'FINANCIAL_PERSONALITIES',
    'ARCHETYPE_BASE_RISK', 
    'get_risk_profile_from_score',
    'TRANSACTION_CHANNELS',
    'get_preferred_p2p_channel',
    'calculate_p2p_risk_modifier',
    'get_transaction_limits',
    'validate_agent_configuration',
    'RealisticP2PStructure'
]
