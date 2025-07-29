# agents/delivery_agent.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np


class DeliveryAgent(BaseAgent):
    """
    A multi-dimensional profile for a Delivery Agent.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower', financial_personality='Over_Spender'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Delivery Agent / Rider"

        # --- RISK SCORE CALCULATION ---
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "15000-25000"
        min_monthly, max_monthly = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_monthly * income_multiplier)}-{int(max_monthly * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Gig_Work_Contractor",
            "employment_verification": "Not_Verified",
            "income_type": "Platform_Payout",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Daily",
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "Mostly_On_Time",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.88, 0.95), 2),
            "ip_consistency_score": round(random.uniform(0.30, 0.50), 2),
            "sim_churn_rate": "Medium",
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Geographically_Dynamic",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Low",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.base_daily_payout = random.uniform(min_mod, max_mod) / 26

        self.loan_emi_amount = (min_mod + max_mod) / 2 * 0.15
        self.cod_settlement_chance = 0.60
        self.cod_balance = 0.0

        self.fuel_spend_chance = 0.90 * personality_config['spend_chance_mod']
        self.recharge_chance = 0.10 * personality_config['spend_chance_mod']

        # ✅ Added P2P attributes - Delivery agents have tight-knit networks
        self.fellow_agents = []  # To be populated by simulation engine
        self.family_back_home = []  # Family members for remittances
        self. peer_network = []  # Other gig workers and similar economic group
        
        self.p2p_transfer_chance = 0.20 * personality_config.get('spend_chance_mod', 1.0)  # Moderate to high frequency
        self.agent_support_chance = 0.15  # Support within delivery agent community
        self.family_remittance_chance = 0.12  # Sending money home
        self.emergency_help_chance = 0.08  # Emergency mutual aid
        
        # Operational sharing patterns common in delivery agent community
        self.fuel_sharing_chance = 0.10  # Sharing fuel costs
        self.vehicle_maintenance_sharing = 0.05  # Sharing repair costs
        
        # Track last remittance to manage frequency
        self.last_family_remittance_date = None

        self.balance = random.uniform(500, 2000)

    def _handle_income_and_settlements(self, date, events):
        """Simulates daily payouts and the critical COD settlement cycle."""
        daily_payout = self.base_daily_payout * random.uniform(0.7, 1.4)
        txn = self.log_transaction("CREDIT", "Platform Payout", daily_payout, date, channel="Bank Transfer")
        if txn: events.append(txn)

        if random.random() < self.cod_settlement_chance:
            num_cod_orders = random.randint(3, 10)
            for _ in range(num_cod_orders):
                cod_amount = random.uniform(100, 800)
                txn = self.log_transaction("CREDIT", "Cash on Delivery Deposit", cod_amount, date, channel="Cash Deposit")
                if txn:
                    events.append(txn)
                    self.cod_balance += cod_amount

            if self.cod_balance > 0:
                txn = self.log_transaction("DEBIT", "COD Settlement to Platform", self.cod_balance, date, channel="UPI")
                if txn:
                    events.append(txn)
                    self.cod_balance = 0.0

    def _handle_fixed_debits(self, date, events):
        """Handles loan payments."""
        if self.has_loan_emi and date.day == 10:
            if random.random() > 0.15:
                txn = self.log_transaction("DEBIT", "Two-Wheeler Loan EMI", self.loan_emi_amount, date, channel="Auto_Debit")
                if txn: events.append(txn)

    def _handle_operational_spending(self, date, events):
        """Simulates daily operational costs like fuel and mobile recharges."""
        if random.random() < self.fuel_spend_chance:
            fuel_amount = random.uniform(150, 400)
            txn = self.log_transaction("DEBIT", "Fuel", fuel_amount, date, channel="UPI")
            if txn: events.append(txn)

        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([99, 149, 199])
            txn = self.log_transaction("DEBIT", "Prepaid Data Recharge", recharge_amount, date, channel="UPI")
            if txn: events.append(txn)

    def _handle_agent_community_transfers(self, date, events, context):
        """✅ NEW: Handles transfers within delivery agent network."""
        if self.fellow_agents and random.random() < self.p2p_transfer_chance:
            recipient = random.choice(self.fellow_agents)
            
            # Small to moderate amounts typical for delivery agents
            base_amount = random.uniform(150, 1000)
            
            # Adjust based on current balance (can't send what you don't have)
            max_sendable = self.balance * 0.25  # Don't send more than 25% of balance
            amount = min(base_amount, max_sendable)
            
            if amount >= 50:  # Minimum viable transfer
                transfer_desc = random.choice([
                    'Fuel Share',
                    'Agent Support',
                    'Repair Help',
                    'Friend Loan',
                    'Meal Share',
                    'Emergency Advance',
                    'Peer Support'
                ])
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(amount, 2), 
                    'desc': transfer_desc,
                    'channel': 'UPI'
                })

    def _handle_family_remittances(self, date, events, context):
        """✅ NEW: Handles sending money back home to family."""
        # Send money home weekly or bi-weekly
        current_week = date.isocalendar()[1]
        send_this_week = current_week % 2 == 0  # Every other week
        
        if (self.family_back_home and 
            send_this_week and
            date.weekday() == 0 and  # Monday remittances
            random.random() < self.family_remittance_chance and
            self.balance > 1000):  # Ensure sufficient balance
            
            current_date_key = date.strftime("%Y-%m-%d")
            if self.last_family_remittance_date != current_date_key:
                recipient = random.choice(self.family_back_home)
                
                # Family remittances are typically 20-40% of available balance
                remittance_amount = self.balance * random.uniform(0.20, 0.40)
                
                # Adjust based on personality
                if self.financial_personality == 'Saver':
                    remittance_amount *= random.uniform(1.2, 1.5)  # Savers send more home
                elif self.financial_personality == 'Over_Spender':
                    remittance_amount *= random.uniform(0.7, 1.0)  # Over-spenders send less
                
                if remittance_amount >= 500:  # Minimum meaningful remittance
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': recipient, 
                        'amount': round(remittance_amount, 2), 
                        'desc': 'Family Remittance',
                        'channel': 'UPI'
                    })
                    self.last_family_remittance_date = current_date_key

    def _handle_operational_sharing(self, date, events, context):
        """✅ NEW: Handles operational cost sharing common in delivery community."""
        # Fuel sharing
        if (self.fellow_agents and 
            random.random() < self.fuel_sharing_chance and
            self.balance > 300):
            
            recipient = random.choice(self.fellow_agents)
            fuel_share_amount = random.uniform(100, 300)
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(fuel_share_amount, 2), 
                'desc': 'Fuel Cost Share',
                'channel': 'UPI'
            })
        
        # Vehicle maintenance sharing
        if (self.fellow_agents and 
            random.random() < self.vehicle_maintenance_sharing and
            self.balance > 500):
            
            recipient = random.choice(self.fellow_agents)
            maintenance_share = random.uniform(200, 800)
            
            # Ensure we don't send more than we can afford
            max_share = self.balance * 0.3
            final_amount = min(maintenance_share, max_share)
            
            if final_amount >= 100:
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': random.choice(['Vehicle Repair Share', 'Maintenance Help', 'Bike Service Share']),
                    'channel': 'UPI'
                })

    def _handle_emergency_support(self, date, events, context):
        """✅ NEW: Handles emergency support within delivery agent community."""
        if (self.fellow_agents and 
            random.random() < self.emergency_help_chance and
            self.balance > 800):  # Need decent balance for emergency help
            
            recipient = random.choice(self.fellow_agents)
            
            # Emergency support amounts
            emergency_amount = random.uniform(300, 1200)
            
            # Can't give more than 40% of balance in emergency
            max_emergency = self.balance * 0.4
            final_amount = min(emergency_amount, max_emergency)
            
            if final_amount >= 200:  # Minimum for meaningful emergency help
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': random.choice([
                        'Agent Emergency',
                        'Medical Help',
                        'Urgent Support',
                        'Crisis Help',
                        'Emergency Advance'
                    ]),
                    'channel': 'UPI'
                })

    def _handle_peer_network_transfers(self, date, events, context):
        """✅ NEW: Handles transfers with broader peer network (other gig workers)."""
        if (self.peer_network and 
            random.random() < self.agent_support_chance and
            self.balance > 400):
            
            recipient = random.choice(self.peer_network)
            
            # Peer network transfers are typically smaller
            peer_amount = random.uniform(100, 600)
            
            # Adjust based on available balance
            max_peer = self.balance * 0.2  # Maximum 20% for peer transfers
            final_amount = min(peer_amount, max_peer)
            
            if final_amount >= 75:  # Minimum for peer transfers
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': random.choice([
                        'Gig Worker Support',
                        'Peer Help',
                        'Worker Solidarity',
                        'Community Support',
                        'Mutual Aid'
                    ]),
                    'channel': 'UPI'
                })

    def act(self, date: datetime, **context):
        """
        ✅ Updated: Simulates the daily high-velocity financial life of a delivery agent with P2P transfers.
        """
        events = []
        self._handle_income_and_settlements(date, events)
        self._handle_fixed_debits(date, events)
        self._handle_operational_spending(date, events)
        self._handle_agent_community_transfers(date, events, context)  # ✅ Agent network transfers
        self._handle_family_remittances(date, events, context)         # ✅ Family remittances
        self._handle_operational_sharing(date, events, context)        # ✅ Operational cost sharing
        self._handle_emergency_support(date, events, context)          # ✅ Emergency support
        self._handle_peer_network_transfers(date, events, context)     # ✅ Broader peer network
        self._handle_daily_living_expenses(date, events, daily_spend_chance=0.6)
        return events
