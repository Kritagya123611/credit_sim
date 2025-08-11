# agents/base_agent.py

import uuid
import random
from faker import Faker
from datetime import datetime, timedelta
import numpy as np

# Import realistic P2P structure configuration
from config_pkg.p2p_structure import RealisticP2PStructure, get_standardized_p2p_description

# Initialize Faker once to be used by all agents
fake = Faker('en_IN')

class BaseAgent:
    """
    Enhanced parent agent class supporting heterogeneous multi-layer graphs,
    temporal behavioral tracking, and advanced P2P transaction handling.
    Updated for Phase 1: Advanced Graph Structure Enhancements.
    
    ✅ NEW FEATURES ADDED:
    - Balance history tracking for volatility calculation
    - Realistic device count distribution (1-5 devices)
    - Improved employment tenure handling
    - Derived feature calculation with noise to prevent identical values
    """
    
    STANDARDIZED_FIELDS = {
        # Core Profile
        "archetype_name": "Generic",
        "risk_profile": "N/A", 
        "risk_score": 0.0,
        "economic_class": "N/A",
        "financial_personality": "N/A",
        "employment_status": "N/A",
        "employment_verification": "N/A",
        "income_type": "N/A",
        "avg_monthly_income_range": "N/A",
        "income_pattern": "N/A",
        
        # Financial Habits
        "savings_retention_rate": "N/A",
        "has_investment_activity": False,
        "investment_types": [],
        "has_loan_emi": False,
        "loan_emi_payment_status": "N/A",
        "has_insurance_payments": False,
        "insurance_types": [],
        
        # Utility Payments
        "utility_payment_status": "N/A",
        "mobile_plan_type": "N/A",
        
        # Digital Footprint
        "device_consistency_score": 0.5,  # ✅ CHANGED: Default to 0.5 instead of 0.0
        "ip_consistency_score": 0.0,
        "sim_churn_rate": "N/A",
        "primary_digital_channels": [],
        "login_pattern": "N/A",
        
        # E-commerce
        "ecommerce_activity_level": "N/A",
        "ecommerce_avg_ticket_size": "N/A",
        
        # ✅ NEW: Heterogeneous Graph Node Connections
        "company_id": None,
        "primary_bank_branch_id": None,
        "family_unit_id": None,
        "device_fingerprints": [],
        "ip_cluster_ids": [],
        "frequent_merchants": [],
        
        # ✅ NEW: Employment & Company Relationship Fields
        "employment_start_date": None,
        "employment_tenure_months": 0,
        "employer_payment_consistency": 1.0,
        "industry_sector": "Unknown",
        "company_size": "Unknown",
        
        # ✅ NEW: Temporal Behavior Tracking
        "behavioral_change_points": [],
        "relationship_start_dates": {},
        "transaction_velocity_windows": {
            'daily': [],
            'weekly': [],
            'monthly': [],
            'quarterly': []
        },
    }

    def __init__(self, **kwargs):
        # Universal Identity Attributes
        self.agent_id = str(uuid.uuid4())
        self.name = fake.name()
        self.email = fake.email()
        self.account_no = fake.bban()
        self.balance = kwargs.get("starting_balance", 0.0)
        self.txn_log = []
        
        # ✅ NEW: Balance history tracking for volatility calculation
        self.balance_history = [self.balance]
        
        # Standardized Fields from schema
        for field, default in self.STANDARDIZED_FIELDS.items():
            setattr(self, field, kwargs.get(field, default))
        
        # ✅ NEW: Realistic device count distribution
        self._initialize_realistic_device_count()
        
        # ✅ NEW: Realistic employment tenure distribution
        self._initialize_employment_tenure()
        
        # Initialize network connections
        self._initialize_p2p_networks()
        self._initialize_heterogeneous_connections()
        self._initialize_temporal_tracking()

    def _initialize_realistic_device_count(self):
        """✅ NEW: Initialize realistic device count based on agent type"""
        # Override in subclasses for agent-specific patterns
        device_count = self.get_realistic_device_count()
        
        # Create device fingerprints
        self.device_fingerprints = []
        for _ in range(device_count):
            device_id = str(uuid.uuid4())
            self.device_fingerprints.append(device_id)
        
        # Update device consistency score with some randomness
        base_score = random.uniform(0.2, 0.9)
        noise = (hash(self.agent_id) % 100) / 100.0 * 0.2 - 0.1  # -0.1 to +0.1
        self.device_consistency_score = max(0.0, min(1.0, base_score + noise))

    def get_realistic_device_count(self):
        """Get realistic device count - override in subclasses"""
        # Default distribution: 1-5 devices with realistic probabilities
        device_options = [1, 2, 3, 4, 5]
        weights = [0.1, 0.4, 0.3, 0.15, 0.05]  # Most people have 2-3 devices
        return random.choices(device_options, weights=weights)[0]

    def _initialize_employment_tenure(self):
        """✅ NEW: Initialize realistic employment tenure"""
        if self.employment_tenure_months == 0:  # Only if not already set
            # Realistic tenure distribution
            tenure_options = [0, 3, 6, 12, 18, 24, 36, 48, 60, 72]
            weights = [0.15, 0.1, 0.15, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02, 0.01]
            self.employment_tenure_months = random.choices(tenure_options, weights=weights)[0]
            
            # Set employment start date if tenure > 0
            if self.employment_tenure_months > 0:
                start_date = datetime.now().date() - timedelta(days=self.employment_tenure_months * 30)
                self.employment_start_date = start_date

    def update_balance(self, new_balance):
        """✅ NEW: Update balance and track history for volatility calculation"""
        self.balance = new_balance
        self.balance_history.append(new_balance)
        
        # Keep only last 30 balance records to prevent memory issues
        if len(self.balance_history) > 30:
            self.balance_history = self.balance_history[-30:]

    def calculate_balance_volatility(self):
        """✅ NEW: Calculate actual balance volatility over time"""
        if len(self.balance_history) < 2:
            return 0.0
        
        balances = np.array(self.balance_history)
        mean_balance = np.mean(balances)
        
        if mean_balance == 0:
            return 0.0
        
        std_balance = np.std(balances)
        volatility = std_balance / mean_balance
        
        # Cap volatility at reasonable max (2.0) for outlier prevention
        return min(volatility, 2.0)

    def calculate_p2p_ratio_with_noise(self):
        """✅ NEW: Calculate P2P ratio with noise to prevent identical values"""
        sent = getattr(self, 'p2p_sent_count', 0)
        received = getattr(self, 'p2p_received_count', 0)
        
        if sent + received == 0:
            return 1.0
        
        # Add agent-specific noise based on agent_id hash
        agent_hash = hash(self.agent_id) % 1000 / 1000.0
        noise = (agent_hash - 0.5) * 0.02  # -0.01 to +0.01 variation
        
        base_ratio = (received + 1) / (sent + received + 2)
        final_ratio = base_ratio + noise
        
        return max(0.0, min(1.0, final_ratio))

    def calculate_credit_debit_ratio_with_noise(self):
        """✅ NEW: Calculate credit/debit ratio with variation"""
        p2p_sent = getattr(self, 'p2p_sent_count', 0)
        p2p_received = getattr(self, 'p2p_received_count', 0)
        total_transactions = getattr(self, 'total_transactions_count', 0)
        
        if total_transactions == 0:
            base_ratio = 0.5
        else:
            # Calculate based on P2P patterns
            if p2p_sent + p2p_received == 0:
                base_ratio = 0.5
            else:
                base_ratio = (p2p_received + 1) / (p2p_sent + p2p_received + 2)
        
        # Add agent-specific randomness
        agent_hash = hash(self.agent_id) % 1000 / 1000.0
        noise = (agent_hash - 0.5) * 0.04  # -0.02 to +0.02 variation
        
        final_ratio = base_ratio + noise
        return max(0.0, min(1.0, final_ratio))

    def calculate_financial_stability_with_variation(self):
        """✅ NEW: Calculate financial stability with realistic variation"""
        # Balance component
        balance = getattr(self, 'balance', 0)
        balance_score = min(balance / 10000, 1.0)
        
        # Consistency component
        behavioral_changes = len(getattr(self, 'behavioral_change_points', []))
        consistency_score = max(0.0, 1.0 - (behavioral_changes * 0.1))
        
        # Tenure component
        tenure_score = min(self.employment_tenure_months / 36, 1.0)
        
        # Device consistency component
        device_score = self.device_consistency_score
        
        # Weighted combination
        stability = (balance_score * 0.3) + (consistency_score * 0.25) + \
                   (tenure_score * 0.25) + (device_score * 0.2)
        
        # Add agent-specific variation
        agent_hash = hash(self.agent_id) % 1000 / 1000.0
        noise = (agent_hash - 0.5) * 0.1  # -0.05 to +0.05 variation
        
        final_stability = stability + noise
        return max(0.0, min(1.0, final_stability))

    def _initialize_p2p_networks(self):
        """Initialize common P2P network attributes that may be used by child classes."""
        if not hasattr(self, 'contacts'):
            self.contacts = []
        if not hasattr(self, 'family_members'):
            self.family_members = []
        if not hasattr(self, 'professional_network'):
            self.professional_network = []

    def _initialize_heterogeneous_connections(self):
        """✅ NEW: Initialize connections to non-agent node types"""
        # Company/employer relationship
        if self.company_id is None:
            self.company_id = None
        if self.employment_start_date is None:
            self.employment_start_date = None
        
        # Device and IP tracking
        if not isinstance(self.device_fingerprints, list):
            self.device_fingerprints = []
        if not isinstance(self.ip_cluster_ids, list):
            self.ip_cluster_ids = []
        
        # Merchant relationships
        if not isinstance(self.frequent_merchants, list):
            self.frequent_merchants = []
        
        # Family and banking
        if self.family_unit_id is None:
            self.family_unit_id = None
        if self.primary_bank_branch_id is None:
            self.primary_bank_branch_id = None

        # Relationship tracking
        if not isinstance(self.relationship_start_dates, dict):
            self.relationship_start_dates = {}

    def _initialize_temporal_tracking(self):
        """✅ NEW: Initialize temporal behavior tracking for multi-graph snapshots"""
        if not isinstance(self.transaction_velocity_windows, dict):
            self.transaction_velocity_windows = {
                'daily': [],
                'weekly': [],
                'monthly': [],
                'quarterly': []
            }
        
        if not isinstance(self.behavioral_change_points, list):
            self.behavioral_change_points = []

    def __repr__(self):
        return f"<Agent {self.agent_id[:6]} - {self.archetype_name}, ₹{self.balance:.2f}>"

    def to_dict(self):
        """Enhanced: Exports agent data including heterogeneous relationships"""
        profile_dict = self.__dict__.copy()
        
        # Remove non-serializable data
        fields_to_remove = ['txn_log', 'transaction_velocity_windows', 'balance_history']
        
        # Remove P2P network object references but keep IDs
        p2p_fields = [
            'contacts', 'family_members', 'professional_network', 'collaborators',
            'service_providers', 'dependents', 'employees', 'family_member_recipient',
            'fellow_agents', 'family_back_home', 'peer_network', 'social_circle',
            'extended_family', 'children_contacts', 'worker_network', 'family_recipient',
            'creator_network', 'freelancer_network', 'brand_contacts', 'junior_associate',
            'family_dependents'
        ]
        
        for field in fields_to_remove + p2p_fields:
            profile_dict.pop(field, None)
        
        # ✅ NEW: Add heterogeneous relationship summary
        profile_dict.update(self.get_heterogeneous_connections_summary())
        
        return profile_dict

    def log_transaction(self, txn_type, description, amount, date, channel="Other", recipient_id=None):
        """Enhanced transaction logging with balance history tracking"""
        # Enhanced validation
        if amount <= 0:
            return None
        
        old_balance = self.balance
        
        if txn_type == "CREDIT":
            self.balance += amount
        elif txn_type == "DEBIT":
            if self.balance < amount:
                return None
            self.balance -= amount
        else:
            return None
        
        # ✅ NEW: Update balance history
        self.update_balance(self.balance)
        
        # Enhanced recipient validation
        if recipient_id == self.agent_id and ("to " in description.lower() or "from " in description.lower()):
            print(f"Warning: Potential recipient_id error for {txn_type} transaction by {self.agent_id[:6]}")
        
        # Realistic description handling for P2P transfers
        if recipient_id and recipient_id != self.agent_id:
            try:
                bank_description = RealisticP2PStructure.format_p2p_transaction(
                    sender_id=self.agent_id,
                    recipient_id=recipient_id,
                    channel=channel,
                    transaction_type=txn_type,
                    include_reference=False
                )
            except:
                bank_description = description
        else:
            bank_description = description
        
        txn = {
            "agent_id": self.agent_id,
            "date": date.strftime("%Y-%m-%d"),
            "type": txn_type,
            "channel": channel,
            "description": bank_description,
            "amount": round(amount, 2),
            "balance_after_txn": round(self.balance, 2),
            "recipient_id": recipient_id
        }
        
        self.txn_log.append(txn)
        
        # ✅ NEW: Update temporal velocity tracking
        self.update_transaction_velocity(date, amount, txn_type)
        
        return txn

    def get_employment_tenure_months(self, current_date=None):
        """✅ ENHANCED: Calculate employment tenure for company relationship strength"""
        if not self.employment_start_date:
            return self.employment_tenure_months  # Use pre-set value
        
        current_date = current_date or datetime.now().date()
        if isinstance(current_date, datetime):
            current_date = current_date.date()
        
        delta = current_date - self.employment_start_date
        calculated_tenure = max(0, delta.days // 30)
        
        # Update stored value
        self.employment_tenure_months = calculated_tenure
        return calculated_tenure

    def get_heterogeneous_connections_summary(self):
        """Get summary of all heterogeneous node connections"""
        return {
            'has_employer': self.company_id is not None,
            'employment_tenure_months': self.get_employment_tenure_months(),
            'device_count': len(self.device_fingerprints),
            'ip_cluster_count': len(self.ip_cluster_ids),
            'frequent_merchant_count': len(self.frequent_merchants),
            'has_family_unit': self.family_unit_id is not None,
            'has_bank_branch': self.primary_bank_branch_id is not None,
            'total_heterogeneous_connections': (
                (1 if self.company_id else 0) +
                len(self.device_fingerprints) +
                len(self.ip_cluster_ids) +
                len(self.frequent_merchants) +
                (1 if self.family_unit_id else 0) +
                (1 if self.primary_bank_branch_id else 0)
            )
        }

    def update_transaction_velocity(self, date, amount, transaction_type):
        """✅ NEW: Update velocity tracking for different temporal windows"""
        if not isinstance(date, datetime):
            return
        
        for window in self.transaction_velocity_windows:
            self.transaction_velocity_windows[window].append({
                'date': date,
                'amount': amount,
                'type': transaction_type
            })
            
            # Keep only relevant time window
            cutoff_days = {'daily': 1, 'weekly': 7, 'monthly': 30, 'quarterly': 90}[window]
            cutoff_date = date - timedelta(days=cutoff_days)
            
            self.transaction_velocity_windows[window] = [
                txn for txn in self.transaction_velocity_windows[window] 
                if txn['date'] >= cutoff_date
            ]

    def get_balance(self):
        """Getter method for current balance."""
        return self.balance

    def has_sufficient_balance(self, amount):
        """Check if agent has sufficient balance for a transaction."""
        return self.balance >= amount

    def get_total_network_connections(self):
        """Get total number of P2P connections across all networks."""
        network_attrs = [
            'contacts', 'family_members', 'professional_network', 'collaborators',
            'service_providers', 'dependents', 'employees', 'fellow_agents',
            'family_back_home', 'peer_network', 'social_circle', 'extended_family',
            'children_contacts', 'worker_network', 'creator_network', 'freelancer_network',
            'brand_contacts', 'family_dependents'
        ]
        
        total = 0
        unique_connections = set()
        
        for attr in network_attrs:
            network = getattr(self, attr, [])
            if network:
                if isinstance(network, list):
                    for connection in network:
                        if hasattr(connection, 'agent_id'):
                            unique_connections.add(connection.agent_id)
                elif hasattr(network, 'agent_id'):  # Single connection
                    unique_connections.add(network.agent_id)
        
        return len(unique_connections)

    def can_send_p2p_transfer(self, amount, min_balance_buffer=100):
        """Check if agent can afford to send a P2P transfer while maintaining minimum balance."""
        return self.balance >= (amount + min_balance_buffer)

    def act(self, date: datetime, **context):
        """Override this method in child classes to define agent behavior."""
        return []
