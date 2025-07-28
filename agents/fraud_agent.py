# agents/fraud_agent.py

import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent

class FraudAgent(BaseAgent):
    """
    A versatile agent class to simulate different types of fraudulent behavior.
    The behavior is determined by the `fraud_type` parameter during initialization.
    - 'ring': Participates in circular transactions with other fraud agents.
    - 'bust_out': Builds a good credit history and then vanishes after maxing out credit.
    - 'mule': Passively receives funds from many sources and then cashes out.
    """
    def __init__(self, fraud_type: str, **fraud_params):
        # 1. Define base profile attributes for a generic fraudster
        profile_attributes = {
            "archetype_name": "Fraudulent Agent",
            "risk_profile": "Fraud", # A distinct category beyond "Very_High"
            "employment_status": "Self-Employed", # Often used to obscure income source
            "employment_verification": "Not_Verified",
            "income_type": "Unclassified",
            "avg_monthly_income_range": "0-10000",
            "income_pattern": "Erratic_High_Variance",
            "savings_retention_rate": "Near_Zero",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "N/A",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.1, 0.4), 2),
            "ip_consistency_score": round(random.uniform(0.1, 0.4), 2),
            "sim_churn_rate": "Very_High",
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Anomalous",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Low",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Configure specific fraud behaviors
        self.fraud_type = fraud_type
        self.is_active = True # A flag to stop behavior after busting out or cashing out

        if self.fraud_type == 'ring':
            # Ring members need a shared identifier and digital footprint
            self.ring_id = fraud_params.get('ring_id', None)
            shared_footprint = fraud_params.get('shared_footprint', {})
            # Overwrite unique device/IP with the shared one
            self.device_id = shared_footprint.get('device_id')
            self.ip_address = shared_footprint.get('ip_address')
            self.balance = random.uniform(5000, 10000) # Start with funds to move around

        elif self.fraud_type == 'bust_out':
            self.behavior_state = 'building_credit' # Initial state
            self.creation_date = fraud_params.get('creation_date', datetime.now().date())
            self.bust_out_day_threshold = random.randint(60, 90) # Days to wait before bust-out
            self.balance = random.uniform(10000, 20000) # Start with a good balance

        elif self.fraud_type == 'mule':
            self.cash_out_threshold = random.uniform(30000, 50000)
            self.balance = random.uniform(0, 1000) # Mules start with low balance

    def act(self, date: datetime, **context):
        if not self.is_active:
            return []

        events = []
        if self.fraud_type == 'ring':
            ring_members = context.get('ring_members', [])
            if len(ring_members) > 1 and random.random() < 0.3:
                my_index = ring_members.index(self)
                recipient_index = (my_index + 1) % len(ring_members)
                recipient = ring_members[recipient_index]
                
                amount = self.balance * random.uniform(0.1, 0.3)
                
                debit_txn = self.log_transaction("DEBIT", f"P2P to Ring Member {recipient.agent_id[:6]}", amount, date)
                if debit_txn:
                    events.append(debit_txn)
                    context['p2p_transfers'].append({'recipient': recipient, 'amount': amount, 'date': date, 'sender_id': self.agent_id})
                    
        elif self.fraud_type == 'bust_out':
            # --- FIX: Convert datetime to date for subtraction ---
            days_active = (date.date() - self.creation_date).days
            
            if self.behavior_state == 'building_credit':
                if days_active < self.bust_out_day_threshold:
                    if random.random() < 0.1: 
                        txn = self.log_transaction("DEBIT", "Utility Bill Payment", random.uniform(500, 1500), date)
                        if txn: events.append(txn)
                else:
                    self.behavior_state = 'busting_out'

            if self.behavior_state == 'busting_out':
                for _ in range(random.randint(3, 6)):
                    amount = self.balance * random.uniform(0.1, 0.25)
                    txn = self.log_transaction("DEBIT", "High-Value E-commerce/Withdrawal", amount, date)
                    if txn: events.append(txn)
                self.is_active = False

        elif self.fraud_type == 'mule':
            if self.balance > self.cash_out_threshold:
                cash_out_amount = self.balance * random.uniform(0.9, 1.0)
                txn = self.log_transaction("DEBIT", "Large Cash Withdrawal", cash_out_amount, date)
                if txn:
                    events.append(txn)
                    self.is_active = False
        
        return events