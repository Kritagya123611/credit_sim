"""
Enhanced P2P Transaction Structure Configuration for Phase 2: Salary Source Tracking

This module defines what banks actually see in P2P/UPI transactions in the real world,
enhanced for Phase 2: Salary source tracking and enhanced behavioral diversity.

Banks don't see user-defined descriptions - they only see standardized transaction metadata
based on the payment channel used. This ensures the graph-based credit risk model trains 
on realistic banking data where P2P transfers are identified by sender/recipient IDs and 
transaction patterns rather than descriptive text that banks don't have access to.

Phase 2 Updates:
- Enhanced salary source tracking from employer companies
- Improved device consistency and behavioral diversity
- Better fraud detection integration
- Enhanced channel selection algorithms
"""

import random
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import re
import hashlib
import json

class RealisticP2PStructure:
    """
    Enhanced P2P transaction structure for Phase 2: Salary source tracking.
    
    Defines what banks actually see in P2P/UPI transactions with support for:
    - Salary source tracking from employer companies
    - Enhanced behavioral diversity prevention
    - Improved fraud detection features
    - Alternative data integration
    """
    
    # ✅ ENHANCED: Bank-visible descriptions with salary source tracking
    BANK_VISIBLE_DESCRIPTIONS = {
        'UPI': 'UPI P2P',
        'IMPS': 'IMPS TXN', 
        'NEFT': 'NEFT CR',
        'RTGS': 'RTGS CR',
        'Wallet': 'WALLET TXN',
        'P2P': 'P2P TXN',
        'Mobile_Banking': 'MB TXN',
        'Net_Banking': 'NB TXN',
        # ✅ ENHANCED: Company-to-agent salary transactions
        'SALARY_CREDIT': 'SAL CREDIT',
        'COMPANY_TXN': 'COMP TXN',
        'EMPLOYER_PAY': 'EMP PAY',
        'CORPORATE_SAL': 'CORP SAL',
        # ✅ ENHANCED: Merchant transactions
        'MERCHANT_PAY': 'MERCH PAY',
        'ECOMMERCE': 'ECOM TXN',
        'SERVICE_PAY': 'SVC PAY',
        # ✅ NEW: Specialized salary source types
        'GOVERNMENT_SAL': 'GOVT SAL',
        'PENSION_PAY': 'PEN PAY',
        'CONTRACTOR_PAY': 'CNTR PAY'
    }
    
    # ✅ ENHANCED: Reference patterns with better security
    REFERENCE_PATTERNS = {
        'UPI': 'UPI/{ref_id}',
        'IMPS': 'IMPS-{ref_id}', 
        'NEFT': 'NEFT-{ref_id}',
        'RTGS': 'RTGS-{ref_id}',
        'Wallet': 'WLT-{ref_id}',
        'P2P': 'P2P-{ref_id}',
        'SALARY_CREDIT': 'SAL-{ref_id}',
        'MERCHANT_PAY': 'MRC-{ref_id}',
        'GOVERNMENT_SAL': 'GOV-{ref_id}',
        'PENSION_PAY': 'PEN-{ref_id}',
        'CONTRACTOR_PAY': 'CNT-{ref_id}'
    }
    
    # ✅ ENHANCED: Channel codes with fraud risk indicators and salary source support
    CHANNEL_CODES = {
        'UPI': {'codes': ['UPI', 'UP'], 'fraud_risk': 1.0, 'speed': 'instant', 'salary_support': True},
        'IMPS': {'codes': ['IMP', 'IM'], 'fraud_risk': 0.8, 'speed': 'instant', 'salary_support': True},
        'NEFT': {'codes': ['NFT', 'NF'], 'fraud_risk': 0.6, 'speed': 'batch', 'salary_support': True},
        'RTGS': {'codes': ['RTG', 'RT'], 'fraud_risk': 0.4, 'speed': 'instant', 'salary_support': True},
        'Wallet': {'codes': ['WLT', 'WL'], 'fraud_risk': 1.2, 'speed': 'instant', 'salary_support': False},
        'P2P': {'codes': ['P2P', 'PP'], 'fraud_risk': 1.0, 'speed': 'instant', 'salary_support': False},
        'Bank_Transfer': {'codes': ['BT', 'TXN'], 'fraud_risk': 0.5, 'speed': 'batch', 'salary_support': True}
    }
    
    # ✅ ENHANCED: Transaction limits by channel with salary considerations
    CHANNEL_LIMITS = {
        'UPI': {'per_txn': 100000, 'daily': 1000000, 'salary_max': 500000},
        'IMPS': {'per_txn': 500000, 'daily': 1000000, 'salary_max': 1000000},
        'NEFT': {'per_txn': 1000000, 'daily': 10000000, 'salary_max': 5000000},
        'RTGS': {'per_txn': 10000000, 'daily': 100000000, 'salary_max': 50000000},
        'Bank_Transfer': {'per_txn': 5000000, 'daily': 20000000, 'salary_max': 10000000},
        'Wallet': {'per_txn': 50000, 'daily': 200000, 'salary_max': 0},  # No salary support
        'P2P': {'per_txn': 100000, 'daily': 500000, 'salary_max': 0}     # No salary support
    }
    
    # ✅ ENHANCED: Node type indicators for heterogeneous graphs with salary tracking
    NODE_TYPE_PREFIXES = {
        'agent': 'AGT',
        'company': 'CMP', 
        'merchant': 'MRC',
        'device': 'DEV',
        'ip_cluster': 'IPC',
        'family_unit': 'FAM',
        'bank_branch': 'BBR',
        # ✅ NEW: Employer-specific prefixes
        'corporate_employer': 'CRP',
        'government_employer': 'GOV',
        'contractor_employer': 'CNT',
        'law_firm_employer': 'LAW',
        'pension_provider': 'PEN'
    }
    
    # ✅ NEW: Salary source type mapping
    SALARY_SOURCE_TYPES = {
        'corporate_employer': 'CORPORATE_SAL',
        'government_employer': 'GOVERNMENT_SAL',
        'contractor_employer': 'CONTRACTOR_PAY',
        'law_firm_employer': 'SALARY_CREDIT',
        'pension_provider': 'PENSION_PAY',
        'company': 'COMPANY_TXN'
    }
    
    # ✅ FIX: Add the missing methods that caused the AttributeError
    @staticmethod
    def get_realistic_transaction_channels() -> List[str]:
        """✅ FIXED: Returns list of realistic transaction channels."""
        return list(RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS.keys())
    
    @staticmethod
    def get_channel_probability_weights() -> Dict[str, float]:
        """✅ FIXED: Enhanced probability weights with salary channels."""
        return {
            'UPI': 0.70,           # Still dominant
            'IMPS': 0.12,          # Popular for salary/business
            'NEFT': 0.08,          # Important for salary
            'RTGS': 0.02,          # High-value only  
            'Wallet': 0.05,        # Mobile wallets
            'P2P': 0.02,           # Generic P2P
            'Bank_Transfer': 0.01  # Traditional banking
        }
    
    @staticmethod
    def get_bank_visible_desc(channel: str, node_type: str = 'agent', 
                             transaction_context: str = 'p2p') -> str:
        """
        ✅ ENHANCED: Returns bank-visible description with salary source context.
        
        Args:
            channel: Payment channel used
            node_type: Type of node (agent, company, merchant, etc.)
            transaction_context: Context (p2p, salary, merchant, etc.)
            
        Returns:
            Bank-visible transaction description
        """
        # ✅ ENHANCED: Salary source handling
        if transaction_context == 'salary':
            salary_type = RealisticP2PStructure.SALARY_SOURCE_TYPES.get(node_type, 'SALARY_CREDIT')
            return RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS.get(salary_type, 'SAL CREDIT')
        
        # Special handling for other node types
        if node_type in ['company', 'corporate_employer']:
            return RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS.get('COMPANY_TXN', 'COMP TXN')
        elif node_type == 'merchant':
            return RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS.get('MERCHANT_PAY', 'MERCH PAY')
        elif node_type == 'government_employer':
            return RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS.get('GOVERNMENT_SAL', 'GOVT SAL')
        elif node_type == 'pension_provider':
            return RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS.get('PENSION_PAY', 'PEN PAY')
        
        return RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS.get(channel, 'P2P TXN')
    
    @staticmethod
    def generate_transaction_reference(channel: str, 
                                     transaction_date: datetime = None,
                                     sender_id: str = None,
                                     recipient_id: str = None,
                                     transaction_context: str = 'p2p') -> str:
        """
        ✅ ENHANCED: Generates realistic transaction reference with salary context.
        """
        if transaction_date is None:
            transaction_date = datetime.now()
        
        # Generate reference based on date, channel, and parties
        date_str = transaction_date.strftime("%y%m%d")
        time_str = transaction_date.strftime("%H%M%S")
        
        # Create unique component using hash if parties provided
        if sender_id and recipient_id:
            unique_str = f"{sender_id[:4]}{recipient_id[:4]}{time_str}{transaction_context}"
            hash_component = hashlib.md5(unique_str.encode()).hexdigest()[:6].upper()
        else:
            hash_component = f"{random.randint(100000, 999999):06d}"
        
        # ✅ ENHANCED: Context-aware pattern selection
        if transaction_context == 'salary':
            pattern = RealisticP2PStructure.REFERENCE_PATTERNS.get('SALARY_CREDIT', 'SAL-{ref_id}')
        else:
            pattern = RealisticP2PStructure.REFERENCE_PATTERNS.get(channel, 'TXN-{ref_id}')
        
        ref_id = f"{date_str}{hash_component}"
        return pattern.format(ref_id=ref_id)
    
    @staticmethod
    def format_p2p_transaction(sender_id: str, 
                             recipient_id: str, 
                             channel: str = 'UPI',
                             transaction_type: str = 'DEBIT',
                             amount: float = None,
                             include_reference: bool = True,
                             sender_node_type: str = 'agent',
                             recipient_node_type: str = 'agent',
                             transaction_context: str = 'p2p') -> str:
        """
        ✅ ENHANCED: Formats transactions with salary source tracking support.
        
        Args:
            sender_id: Sender node ID
            recipient_id: Recipient node ID  
            channel: Payment channel
            transaction_type: DEBIT or CREDIT
            amount: Transaction amount (for validation)
            include_reference: Include transaction reference
            sender_node_type: Type of sender node
            recipient_node_type: Type of recipient node
            transaction_context: Context (p2p, salary, merchant, etc.)
            
        Returns:
            Bank-visible transaction description
        """
        # ✅ ENHANCED: Validate transaction limits with salary considerations
        if amount and not RealisticP2PStructure.validate_transaction_amount(channel, amount, transaction_context):
            # Use alternative channel for large amounts or salary transactions
            channel = RealisticP2PStructure._get_alternative_channel(amount, transaction_context)
        
        # ✅ ENHANCED: Get appropriate description based on context and node types
        if transaction_context == 'salary':
            bank_desc = RealisticP2PStructure.get_bank_visible_desc(channel, sender_node_type, 'salary')
        elif sender_node_type in ['company', 'corporate_employer'] or recipient_node_type in ['company', 'corporate_employer']:
            bank_desc = RealisticP2PStructure.get_bank_visible_desc(channel, 'company', transaction_context)
        elif sender_node_type == 'merchant' or recipient_node_type == 'merchant':
            bank_desc = RealisticP2PStructure.get_bank_visible_desc(channel, 'merchant', transaction_context)
        else:
            bank_desc = RealisticP2PStructure.get_bank_visible_desc(channel, 'agent', transaction_context)
        
        # Format node IDs with type prefixes for heterogeneous graph clarity
        sender_short = RealisticP2PStructure._format_node_id(sender_id, sender_node_type)
        recipient_short = RealisticP2PStructure._format_node_id(recipient_id, recipient_node_type)
        
        if transaction_type == "DEBIT":
            base_desc = f"{bank_desc} TO {recipient_short}"
        else:  # CREDIT
            base_desc = f"{bank_desc} FROM {sender_short}"
        
        if include_reference:
            ref_id = RealisticP2PStructure.generate_transaction_reference(
                channel, datetime.now(), sender_id, recipient_id, transaction_context
            )
            base_desc += f" REF:{ref_id}"
        
        return base_desc
    
    @staticmethod
    def _format_node_id(node_id: str, node_type: str) -> str:
        """✅ ENHANCED: Formats node ID with enhanced type prefix support."""
        prefix = RealisticP2PStructure.NODE_TYPE_PREFIXES.get(node_type, 'UNK')
        node_short = node_id[:6] if node_id else "UNKNOWN"
        return f"{prefix}{node_short}"
    
    @staticmethod
    def _get_alternative_channel(amount: float, transaction_context: str = 'p2p') -> str:
        """✅ ENHANCED: Gets alternative channel based on amount and context."""
        # ✅ ENHANCED: Salary transactions prefer secure channels
        if transaction_context == 'salary':
            if amount > 500000:
                return 'RTGS'
            elif amount > 100000:
                return 'NEFT'
            else:
                return 'IMPS'
        
        # Regular P2P logic
        if amount > 100000:
            return random.choice(['NEFT', 'RTGS'])
        elif amount > 50000:
            return random.choice(['IMPS', 'NEFT'])
        else:
            return 'UPI'
    
    @staticmethod
    def validate_transaction_amount(channel: str, amount: float, 
                                  transaction_context: str = 'p2p') -> bool:
        """✅ ENHANCED: Validates transaction amount with salary context."""
        limits = RealisticP2PStructure.CHANNEL_LIMITS.get(channel)
        if not limits:
            return True
        
        # ✅ ENHANCED: Check salary-specific limits
        if transaction_context == 'salary':
            salary_max = limits.get('salary_max', 0)
            if salary_max == 0:  # Channel doesn't support salary
                return False
            return amount <= salary_max
        
        return amount <= limits['per_txn']
    
    @staticmethod
    def validate_channel(channel: str) -> bool:
        """✅ ENHANCED: Enhanced channel validation."""
        return channel in RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS
    
    @staticmethod
    def supports_salary_transactions(channel: str) -> bool:
        """✅ NEW: Check if channel supports salary transactions."""
        channel_info = RealisticP2PStructure.CHANNEL_CODES.get(channel, {})
        return channel_info.get('salary_support', False)
    
    @staticmethod
    def get_channel_code(channel: str) -> str:
        """✅ ENHANCED: Returns abbreviated channel code with metadata."""
        channel_info = RealisticP2PStructure.CHANNEL_CODES.get(channel, {'codes': ['UNK']})
        return random.choice(channel_info['codes'])
    
    @staticmethod
    def get_channel_fraud_risk(channel: str) -> float:
        """✅ ENHANCED: Returns fraud risk multiplier for channel."""
        channel_info = RealisticP2PStructure.CHANNEL_CODES.get(channel, {'fraud_risk': 1.0})
        return channel_info.get('fraud_risk', 1.0)
    
    @staticmethod
    def is_p2p_transaction(description: str) -> bool:
        """✅ ENHANCED: Enhanced P2P detection with salary source awareness."""
        p2p_indicators = [
            'UPI P2P', 'IMPS TXN', 'P2P TXN', 'TO AGT', 'FROM AGT', 
            'TO CMP', 'FROM CMP', 'TO MRC', 'FROM MRC'
        ]
        return any(indicator in description.upper() for indicator in p2p_indicators)
    
    @staticmethod
    def is_salary_transaction(description: str) -> bool:
        """✅ NEW: Detect salary transactions."""
        salary_indicators = [
            'SAL CREDIT', 'GOVT SAL', 'PEN PAY', 'CNTR PAY', 'CORP SAL', 'EMP PAY'
        ]
        return any(indicator in description.upper() for indicator in salary_indicators)
    
    @staticmethod
    def extract_counterparty_id(description: str, 
                               transaction_type: str) -> Optional[str]:
        """✅ ENHANCED: Enhanced counterparty extraction with salary support."""
        if transaction_type == "DEBIT":
            # Look for "TO NODEXXXXXXXX" pattern
            match = re.search(r'TO ([A-Z]{3}[A-Za-z0-9]+)', description)
        else:  # CREDIT
            # Look for "FROM NODEXXXXXXXX" pattern
            match = re.search(r'FROM ([A-Z]{3}[A-Za-z0-9]+)', description)
        
        if match:
            # Extract actual ID without node type prefix
            full_id = match.group(1)
            return full_id[3:] if len(full_id) > 3 else full_id
        
        return None
    
    @staticmethod
    def extract_node_type(description: str, transaction_type: str) -> Optional[str]:
        """✅ ENHANCED: Enhanced node type extraction with salary sources."""
        if transaction_type == "DEBIT":
            match = re.search(r'TO ([A-Z]{3})', description)
        else:  # CREDIT
            match = re.search(r'FROM ([A-Z]{3})', description)
        
        if match:
            prefix = match.group(1)
            # Reverse lookup in NODE_TYPE_PREFIXES
            for node_type, type_prefix in RealisticP2PStructure.NODE_TYPE_PREFIXES.items():
                if type_prefix == prefix:
                    return node_type
        
        return 'agent'  # Default to agent
    
    @staticmethod
    def select_realistic_channel(amount: float = None, 
                               economic_class: str = None,
                               urgency: str = 'normal',
                               transaction_context: str = 'p2p',
                               node_type: str = 'agent') -> str:
        """
        ✅ ENHANCED: Enhanced channel selection with salary source awareness.
        
        Args:
            amount: Transaction amount (influences channel choice)
            economic_class: Sender's economic class
            urgency: Transaction urgency (normal, high, emergency)
            transaction_context: Context (p2p, salary, merchant, etc.)
            node_type: Node type for context-aware selection
            
        Returns:
            Selected payment channel
        """
        base_weights = RealisticP2PStructure.get_channel_probability_weights()
        
        # ✅ ENHANCED: Salary transaction preferences
        if transaction_context == 'salary':
            # Salary transactions prefer secure, reliable channels
            base_weights['NEFT'] *= 2.0
            base_weights['IMPS'] *= 1.5
            base_weights['Bank_Transfer'] = base_weights.get('Bank_Transfer', 0.1) * 3.0
            base_weights['UPI'] *= 1.2  # Still popular for salaries
            base_weights['Wallet'] *= 0.1  # Rarely used for salaries
            base_weights['P2P'] *= 0.1   # Rarely used for salaries
        
        # Adjust weights based on amount
        if amount:
            if amount > 100000:  # High-value transactions prefer secure channels
                base_weights['RTGS'] *= 3.0
                base_weights['NEFT'] *= 2.0
                base_weights['UPI'] *= 0.5
            elif amount < 1000:  # Small amounts prefer instant channels
                base_weights['UPI'] *= 1.5
                base_weights['Wallet'] *= 1.3
        
        # Adjust weights based on economic class
        if economic_class:
            if economic_class in ['High', 'Upper_Middle']:
                base_weights['RTGS'] *= 2.0
                base_weights['NEFT'] *= 1.5
            elif economic_class in ['Lower', 'Lower_Middle']:
                base_weights['UPI'] *= 1.3
                base_weights['Wallet'] *= 1.2
        
        # Adjust for urgency
        if urgency == 'high':
            base_weights['UPI'] *= 1.5
            base_weights['IMPS'] *= 1.3
        elif urgency == 'emergency':
            base_weights['UPI'] *= 2.0
            base_weights['IMPS'] *= 1.8
        
        # ✅ ENHANCED: Node type preferences
        if node_type in ['corporate_employer', 'government_employer']:
            base_weights['NEFT'] *= 2.0
            base_weights['RTGS'] *= 1.5
            base_weights['Bank_Transfer'] = base_weights.get('Bank_Transfer', 0.1) * 4.0
        
        # Normalize weights and filter valid channels
        valid_channels = {k: v for k, v in base_weights.items() 
                         if k in RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS}
        
        total_weight = sum(valid_channels.values())
        if total_weight == 0:
            return 'UPI'  # Fallback
        
        normalized_weights = {k: v/total_weight for k, v in valid_channels.items()}
        
        channels = list(normalized_weights.keys())
        weights = list(normalized_weights.values())
        
        return random.choices(channels, weights=weights, k=1)[0]
    
    @staticmethod
    def get_transaction_metadata(description: str) -> Dict[str, str]:
        """✅ ENHANCED: Enhanced metadata extraction with salary source awareness."""
        metadata = {
            'is_p2p': RealisticP2PStructure.is_p2p_transaction(description),
            'is_salary': RealisticP2PStructure.is_salary_transaction(description),
            'channel': 'unknown',
            'node_type': 'unknown',
            'counterparty_id': None,
            'reference_id': None,
            'transaction_context': 'p2p'
        }
        
        # Determine transaction context
        if metadata['is_salary']:
            metadata['transaction_context'] = 'salary'
        elif any(merchant_ind in description.upper() for merchant_ind in ['MERCH', 'ECOM', 'SVC']):
            metadata['transaction_context'] = 'merchant'
        
        # Extract channel
        for channel in RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS:
            if channel in description or RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS[channel] in description:
                metadata['channel'] = channel
                break
        
        # Extract reference ID
        ref_match = re.search(r'REF:([A-Za-z0-9\-/]+)', description)
        if ref_match:
            metadata['reference_id'] = ref_match.group(1)
        
        # Extract node type and counterparty for relevant transactions
        if metadata['is_p2p'] or metadata['is_salary']:
            if 'TO ' in description:
                metadata['counterparty_id'] = RealisticP2PStructure.extract_counterparty_id(description, 'DEBIT')
                metadata['node_type'] = RealisticP2PStructure.extract_node_type(description, 'DEBIT')
            elif 'FROM ' in description:
                metadata['counterparty_id'] = RealisticP2PStructure.extract_counterparty_id(description, 'CREDIT')
                metadata['node_type'] = RealisticP2PStructure.extract_node_type(description, 'CREDIT')
        
        return metadata

# ===================================================================
# ENHANCED UTILITY FUNCTIONS FOR PHASE 2
# ===================================================================

def get_standardized_p2p_description(sender_id: str, 
                                   recipient_id: str, 
                                   channel: str = 'UPI',
                                   transaction_type: str = 'DEBIT',
                                   sender_node_type: str = 'agent',
                                   recipient_node_type: str = 'agent',
                                   transaction_context: str = 'p2p') -> str:
    """✅ ENHANCED: Enhanced standardized descriptions with salary support."""
    return RealisticP2PStructure.format_p2p_transaction(
        sender_id, recipient_id, channel, transaction_type, 
        include_reference=False, sender_node_type=sender_node_type,
        recipient_node_type=recipient_node_type, transaction_context=transaction_context
    )

def get_standardized_salary_description(employer_id: str,
                                      employee_id: str,
                                      channel: str = 'NEFT',
                                      employer_type: str = 'corporate_employer') -> str:
    """✅ NEW: Standardized salary transaction descriptions."""
    return RealisticP2PStructure.format_p2p_transaction(
        employer_id, employee_id, channel, 'CREDIT',
        include_reference=False, sender_node_type=employer_type,
        recipient_node_type='agent', transaction_context='salary'
    )

def is_realistic_p2p_channel(channel: str) -> bool:
    """✅ Enhanced channel validation."""
    return RealisticP2PStructure.validate_channel(channel)

def get_random_p2p_channel(amount: float = None, 
                          economic_class: str = None,
                          transaction_context: str = 'p2p') -> str:
    """✅ ENHANCED: Enhanced random channel with context awareness."""
    return RealisticP2PStructure.select_realistic_channel(
        amount, economic_class, transaction_context=transaction_context
    )

def get_channel_limits(channel: str) -> Dict[str, int]:
    """✅ ENHANCED: Enhanced channel limits with salary support."""
    return RealisticP2PStructure.CHANNEL_LIMITS.get(channel, {
        'per_txn': 0, 'daily': 0, 'salary_max': 0
    })

def calculate_channel_risk_score(channel: str, amount: float, 
                               frequency: int, transaction_context: str = 'p2p') -> float:
    """✅ ENHANCED: Enhanced risk calculation with salary context."""
    base_risk = RealisticP2PStructure.get_channel_fraud_risk(channel)
    
    # ✅ ENHANCED: Salary transactions are generally lower risk
    if transaction_context == 'salary':
        base_risk *= 0.7  # Reduce risk for salary transactions
    
    # Amount risk factor
    limits = get_channel_limits(channel)
    if transaction_context == 'salary' and limits['salary_max'] > 0:
        amount_risk = min(amount / limits['salary_max'], 1.0)
    else:
        amount_risk = min(amount / limits['per_txn'], 1.0) if limits['per_txn'] > 0 else 0
    
    # Frequency risk factor
    frequency_risk = min(frequency / 10.0, 1.0)  # 10+ transactions is high risk
    
    # Combined risk score
    risk_score = (base_risk * 0.5) + (amount_risk * 0.3) + (frequency_risk * 0.2)
    
    return min(risk_score, 1.0)  # Cap at 1.0

# ===================================================================
# ✅ FIXED: Configuration constants that caused the AttributeError
# ===================================================================

REALISTIC_P2P_CHANNELS = RealisticP2PStructure.get_realistic_transaction_channels()
P2P_CHANNEL_WEIGHTS = RealisticP2PStructure.get_channel_probability_weights()
BANK_VISIBLE_FORMATS = RealisticP2PStructure.BANK_VISIBLE_DESCRIPTIONS
CHANNEL_LIMITS = RealisticP2PStructure.CHANNEL_LIMITS
NODE_TYPE_PREFIXES = RealisticP2PStructure.NODE_TYPE_PREFIXES
SALARY_SOURCE_TYPES = RealisticP2PStructure.SALARY_SOURCE_TYPES

# ===================================================================
# ENHANCED VALIDATION & TESTING
# ===================================================================

def _validate_p2p_structure():
    """✅ ENHANCED: Enhanced validation with salary source testing."""
    print("✅ Validating RealisticP2PStructure configuration...")
    
    # Test basic functionality
    test_desc = RealisticP2PStructure.format_p2p_transaction(
        "test_sender_123", "test_recipient_456", "UPI", "DEBIT"
    )
    assert RealisticP2PStructure.is_p2p_transaction(test_desc), "P2P detection failed"
    
    # ✅ NEW: Test salary transactions
    salary_desc = RealisticP2PStructure.format_p2p_transaction(
        "company_001", "agent_123", "NEFT", "CREDIT", 
        sender_node_type="corporate_employer", recipient_node_type="agent",
        transaction_context="salary"
    )
    assert RealisticP2PStructure.is_salary_transaction(salary_desc), "Salary transaction detection failed"
    
    # Test heterogeneous nodes
    company_desc = RealisticP2PStructure.format_p2p_transaction(
        "company_001", "agent_123", "NEFT", "CREDIT", 
        sender_node_type="company", recipient_node_type="agent"
    )
    assert "COMP TXN" in company_desc or "SAL CREDIT" in company_desc, "Company transaction formatting failed"
    
    # ✅ FIXED: Test the methods that were missing
    channels = RealisticP2PStructure.get_realistic_transaction_channels()
    assert len(channels) > 0, "No channels returned"
    
    weights = RealisticP2PStructure.get_channel_probability_weights()
    assert sum(weights.values()) > 0, "No channel weights"
    
    # ✅ NEW: Test channel selection with salary context
    salary_channel = RealisticP2PStructure.select_realistic_channel(
        amount=75000, economic_class="Middle", transaction_context="salary"
    )
    assert salary_channel in REALISTIC_P2P_CHANNELS, "Salary channel selection failed"
    assert RealisticP2PStructure.supports_salary_transactions(salary_channel), "Selected channel doesn't support salary"
    
    # ✅ NEW: Test salary amount validation
    is_valid = RealisticP2PStructure.validate_transaction_amount("NEFT", 500000, "salary")
    assert is_valid, "Salary amount validation failed"
    
    print("✅ RealisticP2PStructure validation completed successfully!")

# Run validation on import
if __name__ == "__main__":
    _validate_p2p_structure()
else:
    # Quick validation for import
    try:
        _validate_p2p_structure()
    except Exception as e:
        print(f"⚠️  RealisticP2PStructure validation warning: {e}")
