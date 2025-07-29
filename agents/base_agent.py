# agents/base_agent.py

import uuid
import random
from faker import Faker
from datetime import datetime

# Initialize Faker once to be used by all agents
fake = Faker('en_IN')

class BaseAgent:
    """
    The parent agent class containing standardized fields,
    transaction logging, and common behavior across all agent archetypes.
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
        "device_consistency_score": 0.0,
        "ip_consistency_score": 0.0,
        "sim_churn_rate": "N/A",
        "primary_digital_channels": [],
        "login_pattern": "N/A",
        # E-commerce
        "ecommerce_activity_level": "N/A",
        "ecommerce_avg_ticket_size": "N/A",
    }

    def __init__(self, **kwargs):
        # Universal Identity Attributes
        self.agent_id = str(uuid.uuid4())
        self.name = fake.name()
        self.email = fake.email()
        self.account_no = fake.bban()
        self.balance = kwargs.get("starting_balance", 0.0)
        self.txn_log = []
        
        # Standardized Fields from schema
        for field, default in self.STANDARDIZED_FIELDS.items():
            setattr(self, field, kwargs.get(field, default))
        
        # ✅ NEW: Initialize common P2P network attributes (to be populated by simulation engine)
        # These are common across many agent types and will be initialized as empty lists
        self._initialize_p2p_networks()

    def _initialize_p2p_networks(self):
        """✅ NEW: Initialize common P2P network attributes that may be used by child classes."""
        # Only initialize if not already set by child class
        if not hasattr(self, 'contacts'):
            self.contacts = []
        if not hasattr(self, 'family_members'):
            self.family_members = []
        if not hasattr(self, 'professional_network'):
            self.professional_network = []

    def __repr__(self):
        return f"<Agent {self.agent_id[:6]} - {self.archetype_name}, ₹{self.balance:.2f}>"

    def to_dict(self):
        """✅ Enhanced: Converts the agent's profile attributes to a dictionary for CSV/JSON export."""
        profile_dict = self.__dict__.copy()
        
        # Remove non-serializable or sensitive data
        fields_to_remove = ['balance', 'txn_log']
        
        # ✅ NEW: Remove P2P network lists from export (they contain object references)
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
        
        return profile_dict

    def get_transaction_log(self):
        """Returns the list of all transactions performed by the agent."""
        return self.txn_log

    def log_transaction(self, txn_type, description, amount, date, channel="Other", recipient_id=None):
        """✅ Enhanced: Logs a validated transaction and updates agent's balance with improved validation."""
        # ✅ Enhanced validation
        if amount <= 0:
            return None  # Reject invalid amounts
        
        if txn_type == "CREDIT":
            self.balance += amount
        elif txn_type == "DEBIT":
            if self.balance < amount:
                return None  # Reject transaction - insufficient funds
            self.balance -= amount
        else:
            return None  # Reject invalid transaction type
        
        # ✅ Enhanced: Ensure recipient_id is not the same as agent_id for transfers
        if recipient_id == self.agent_id and ("to " in description.lower() or "from " in description.lower()):
            print(f"Warning: Potential recipient_id error for {txn_type} transaction by {self.agent_id[:6]}")
        
        txn = {
            "agent_id": self.agent_id,
            "date": date.strftime("%Y-%m-%d"),
            "type": txn_type,
            "channel": channel,
            "description": description,
            "amount": round(amount, 2),
            "balance_after_txn": round(self.balance, 2),
            "recipient_id": recipient_id
        }
        self.txn_log.append(txn)
        return txn

    def _handle_daily_living_expenses(self, date, events, daily_spend_chance=0.4):
        """✅ Enhanced: Simulates generic daily expenses with improved error handling and variety."""
        personality_chance_mod = {
            "Saver": 0.5, "Over_Spender": 1.5,
            "Rational_Investor": 0.8, "Risk_Addict": 1.2
        }.get(self.financial_personality, 1.0)

        if random.random() < (daily_spend_chance * personality_chance_mod):
            try:
                # ✅ Enhanced: Better handling of income range parsing
                income_range = str(self.avg_monthly_income_range)
                if '-' in income_range and income_range != "N/A":
                    min_inc, max_inc = map(int, income_range.split('-'))
                    avg_monthly_income = (min_inc + max_inc) / 2
                    daily_slice = avg_monthly_income / 30
                    spend_amount = daily_slice * random.uniform(0.5, 2.0)
                    
                    # ✅ Enhanced: Minimum spend threshold and balance check
                    if spend_amount >= 10 and self.balance > spend_amount:
                        # ✅ Enhanced: More varied expense descriptions
                        description = random.choice([
                            "Food & Refreshments", "Local Commute", 
                            "General Store Purchase", "Daily Necessities",
                            "Snacks & Beverages", "Transportation",
                            "Household Items", "Personal Care"
                        ])
                        channel = random.choice(["UPI", "Card", "Cash"])
                        txn = self.log_transaction("DEBIT", description, spend_amount, date, channel=channel)
                        if txn:
                            events.append(txn)
            except (ValueError, ZeroDivisionError, AttributeError, TypeError):
                # ✅ Enhanced: More specific error handling
                pass

    def get_balance(self):
        """✅ NEW: Getter method for current balance."""
        return self.balance

    def has_sufficient_balance(self, amount):
        """✅ NEW: Check if agent has sufficient balance for a transaction."""
        return self.balance >= amount

    def get_network_size(self, network_name):
        """✅ NEW: Get the size of a specific P2P network."""
        network = getattr(self, network_name, [])
        return len(network) if network else 0

    def get_total_network_connections(self):
        """✅ NEW: Get total number of P2P connections across all networks."""
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
        """✅ NEW: Check if agent can afford to send a P2P transfer while maintaining minimum balance."""
        return self.balance >= (amount + min_balance_buffer)

    def act(self, date: datetime, **context):
        """Override this method in child classes to define agent behavior."""
        return []
