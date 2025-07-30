"""
Realistic P2P Transaction Structure Configuration

This module defines what banks actually see in P2P/UPI transactions in the real world.
Banks don't see user-defined descriptions - they only see standardized transaction metadata
based on the payment channel used.

This ensures the graph-based credit risk model trains on realistic banking data
where P2P transfers are identified by sender/recipient IDs and transaction patterns
rather than descriptive text that banks don't have access to.
"""

import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

class RealisticP2PStructure:
    """
    Defines what banks actually see in P2P/UPI transactions.
    
    This class ensures that the simulation reflects real-world banking data
    where P2P transactions have standardized descriptions based on the
    payment channel, not user-defined descriptions.
    """
    
    # What banks actually see based on payment channel
    BANK_VISIBLE_DESCRIPTIONS = {
        'UPI': 'UPI P2P',
        'IMPS': 'IMPS TXN',
        'NEFT': 'NEFT CR',
        'RTGS': 'RTGS CR',
        'Wallet': 'WALLET TXN',
        'P2P': 'P2P TXN',
        'Mobile_Banking': 'MB TXN',
        'Net_Banking': 'NB TXN'
    }
    
    # Transaction reference patterns banks use
    REFERENCE_PATTERNS = {
        'UPI': 'UPI/{ref_id}',
        'IMPS': 'IMPS-{ref_id}',
        'NEFT': 'NEFT-{ref_id}',
        'RTGS': 'RTGS-{ref_id}',
        'Wallet': 'WLT-{ref_id}',
        'P2P': 'P2P-{ref_id}'
    }
    
    # Standard channel codes used by Indian banks
    CHANNEL_CODES = {
        'UPI': ['UPI', 'UP'],
        'IMPS': ['IMP', 'IM'],
        'NEFT': ['NFT', 'NF'],
        'RTGS': ['RTG', 'RT'],
        'Wallet': ['WLT', 'WL'],
        'P2P': ['P2P', 'PP']
    }
    
    # What banks DON'T see (user-defined descriptions that are hidden)
    HIDDEN_FROM_BANK = [
        'Family Support', 'Emergency Help', 'Fuel Share', 
        'Collaboration Fee', 'Service Payment', 'Friend Loan',
        'Rent Payment', 'Food Expenses', 'Medical Help',
        'Equipment Share', 'Creator Support', 'Freelancer Payment',
        'Brand Advance', 'Worker Support', 'Peer Help',
        'Community Support', 'Mutual Aid', 'Crisis Support'
    ]
    
    @staticmethod
    def get_bank_visible_desc(channel: str) -> str:
        """Returns what the bank actually sees based on payment channel."""
        return RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS.get(
            channel, 'P2P TXN'
        )
    
    @staticmethod
    def generate_transaction_reference(channel: str, 
                                     transaction_date: datetime = None) -> str:
        """Generates realistic transaction reference number based on channel."""
        if transaction_date is None:
            transaction_date = datetime.now()
        
        # Generate reference based on date and random component
        date_str = transaction_date.strftime("%y%m%d")
        random_component = random.randint(100000, 999999)
        
        pattern = RealisticP2PStructure.REFERENCE_PATTERNS.get(
            channel, 'TXN-{ref_id}'
        )
        
        ref_id = f"{date_str}{random_component}"
        return pattern.format(ref_id=ref_id)
    
    @staticmethod
    def format_p2p_transaction(sender_id: str, 
                             recipient_id: str, 
                             channel: str = 'UPI',
                             transaction_type: str = 'DEBIT',
                             amount: float = None,
                             include_reference: bool = True) -> str:
        """Formats P2P transaction as banks see it in account statements."""
        bank_desc = RealisticP2PStructure.get_bank_visible_desc(channel)
        
        # Truncate agent IDs to realistic length (banks typically show 6-8 chars)
        sender_short = sender_id[:8] if sender_id else "UNKNOWN"
        recipient_short = recipient_id[:8] if recipient_id else "UNKNOWN"
        
        if transaction_type == "DEBIT":
            # Outgoing transfer format
            base_desc = f"{bank_desc} TO {recipient_short}"
        else:  # CREDIT
            # Incoming transfer format  
            base_desc = f"{bank_desc} FROM {sender_short}"
        
        if include_reference:
            ref_id = RealisticP2PStructure.generate_transaction_reference(channel)
            base_desc += f" REF:{ref_id}"
        
        return base_desc
    
    @staticmethod
    def validate_channel(channel: str) -> bool:
        """Validates if the channel is supported in Indian banking system."""
        return channel in RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS
    
    @staticmethod
    def get_channel_code(channel: str) -> str:
        """Returns abbreviated channel code used in bank systems."""
        codes = RealisticP2PStructure.CHANNEL_CODES.get(channel, ['UNK'])
        return random.choice(codes)
    
    @staticmethod
    def is_p2p_transaction(description: str) -> bool:
        """Determines if a transaction description represents a P2P transfer."""
        p2p_indicators = ['UPI P2P', 'IMPS TXN', 'P2P TXN', 'TO ', 'FROM ']
        return any(indicator in description.upper() for indicator in p2p_indicators)
    
    @staticmethod
    def extract_counterparty_id(description: str, 
                               transaction_type: str) -> Optional[str]:
        """Extracts counterparty agent ID from bank transaction description."""
        if transaction_type == "DEBIT":
            # Look for "TO XXXXXXXX" pattern
            match = re.search(r'TO ([A-Za-z0-9]+)', description)
        else:  # CREDIT
            # Look for "FROM XXXXXXXX" pattern
            match = re.search(r'FROM ([A-Za-z0-9]+)', description)
        
        return match.group(1) if match else None
    
    @staticmethod
    def get_realistic_transaction_channels() -> List[str]:
        """Returns list of realistic transaction channels for P2P transfers."""
        return list(RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS.keys())
    
    @staticmethod
    def get_channel_probability_weights() -> Dict[str, float]:
        """Returns probability weights for different P2P channels based on
        Indian market usage patterns."""
        return {
            'UPI': 0.75,      # Dominant in Indian P2P market
            'IMPS': 0.15,     # Popular for instant transfers
            'NEFT': 0.05,     # Less common for P2P, more for business
            'RTGS': 0.01,     # Rare for P2P (high-value only)
            'Wallet': 0.03,   # Mobile wallets
            'P2P': 0.01       # Generic P2P
        }
    
    @staticmethod
    def select_realistic_channel() -> str:
        """Selects a realistic P2P channel based on Indian market usage patterns."""
        channels = list(RealisticP2PStructure.get_channel_probability_weights().keys())
        weights = list(RealisticP2PStructure.get_channel_probability_weights().values())
        
        return random.choices(channels, weights=weights, k=1)[0]

# Utility functions for integration with existing codebase
def get_standardized_p2p_description(sender_id: str, 
                                   recipient_id: str, 
                                   channel: str = 'UPI',
                                   transaction_type: str = 'DEBIT') -> str:
    """Convenience function to get standardized P2P description."""
    return RealisticP2PStructure.format_p2p_transaction(
        sender_id, recipient_id, channel, transaction_type, include_reference=False
    )

def is_realistic_p2p_channel(channel: str) -> bool:
    """Convenience function to validate P2P channels."""
    return RealisticP2PStructure.validate_channel(channel)

def get_random_p2p_channel() -> str:
    """Convenience function to get a random realistic P2P channel."""
    return RealisticP2PStructure.select_realistic_channel()

# Configuration constants that can be imported directly
REALISTIC_P2P_CHANNELS = RealisticP2PStructure.get_realistic_transaction_channels()
P2P_CHANNEL_WEIGHTS = RealisticP2PStructure.get_channel_probability_weights()
BANK_VISIBLE_FORMATS = RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS
