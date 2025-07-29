# agents/fraud_agent.py

import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent

# --- UPDATED IMPORTS: All 15 legitimate agent classes are now available to be mimicked ---
from agents.salaried import SalariedProfessional
from agents.gig_worker import GigWorker
from agents.gov_employee import GovernmentEmployee
from agents.student import Student
from agents.laborer import DailyWageLaborer
from agents.small_business import SmallBusinessOwner
from agents.doctor import Doctor
from agents.it_employee import TechProfessional
from agents.police import PoliceOfficer
from agents.senior import SeniorCitizen
from agents.delivery_agent import DeliveryAgent
from agents.lawyer import Lawyer
from agents.migrant import MigrantWorker
from agents.content_creator import ContentCreator
from agents.homemaker import Homemaker

class FraudAgent(BaseAgent):
    """
    An enhanced fraud agent that mimics a legitimate agent profile to hide,
    forcing detection to rely on graph patterns rather than profile features.
    """
    def __init__(self, fraud_type: str, **fraud_params):
        
        mimic_class = fraud_params.get('mimic_agent_class', GigWorker)
        mimic_agent = mimic_class()
        profile_attributes = mimic_agent.to_dict()

        profile_attributes['archetype_name'] = f"Fraudulent Agent ({fraud_type})"
        profile_attributes['risk_profile'] = "Fraud"
        profile_attributes['risk_score'] = 0.99
        profile_attributes['device_consistency_score'] = round(random.uniform(0.1, 0.4), 2)
        profile_attributes['ip_consistency_score'] = round(random.uniform(0.1, 0.4), 2)
        profile_attributes['sim_churn_rate'] = "Very_High"

        super().__init__(**profile_attributes)

        self.fraud_type = fraud_type
        self.is_active = True

        if self.fraud_type == 'ring':
            self.ring_id = fraud_params.get('ring_id', None)
            shared_footprint = fraud_params.get('shared_footprint', {})
            self.device_id = shared_footprint.get('device_id')
            self.ip_address = shared_footprint.get('ip_address')
            self.balance = random.uniform(5000, 10000)
        elif self.fraud_type == 'bust_out':
            self.behavior_state = 'building_credit'
            self.creation_date = fraud_params.get('creation_date', datetime.now().date())
            self.bust_out_day_threshold = random.randint(60, 90)
            self.balance = random.uniform(10000, 20000)
        elif self.fraud_type == 'mule':
            self.cash_out_threshold = random.uniform(30000, 50000)
            self.balance = random.uniform(0, 1000)

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
                
                # The agent NO LONGER logs its own debit.
                # It ONLY adds a request to the P2P transfer queue for the engine to process.
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': amount, 
                    'desc': 'P2P Ring Transfer'
                })
                    
        elif self.fraud_type == 'bust_out':
            days_active = (date.date() - self.creation_date).days
            
            if self.behavior_state == 'building_credit':
                if days_active < self.bust_out_day_threshold:
                    if random.random() < 0.2: 
                        desc = random.choice(["UPI Spend - Food", "Prepaid Mobile Recharge"])
                        txn = self.log_transaction("DEBIT", desc, random.uniform(200, 500), date, channel="UPI")
                        if txn: events.append(txn)
                else:
                    self.behavior_state = 'busting_out'

            if self.behavior_state == 'busting_out':
                for _ in range(random.randint(3, 6)):
                    amount = self.balance * random.uniform(0.1, 0.25)
                    desc = random.choice(["E-commerce Purchase", "ATM Withdrawal"])
                    channel = "Card" if "E-commerce" in desc else "ATM"
                    txn = self.log_transaction("DEBIT", desc, amount, date, channel=channel)
                    if txn: events.append(txn)
                self.is_active = False

        elif self.fraud_type == 'mule':
            if self.balance > self.cash_out_threshold:
                cash_out_amount = self.balance * random.uniform(0.9, 1.0)
                txn = self.log_transaction("DEBIT", "ATM Withdrawal", cash_out_amount, date, channel="ATM")
                if txn:
                    events.append(txn)
                    self.is_active = False
        
        return events