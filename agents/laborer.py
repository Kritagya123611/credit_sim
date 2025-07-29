# agents/daily_wage_laborer.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np


class DailyWageLaborer(BaseAgent):
    """
    A multi-dimensional profile for a Daily Wage Laborer.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Daily Wage Laborer"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "7000-15000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Informal_Labor",
            "employment_verification": "Not_Verified",
            "income_type": "Cash_Deposit, Wages",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Daily",
            "savings_retention_rate": "Near_Zero",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "N/A",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.30, 0.55), 2),
            "ip_consistency_score": round(random.uniform(0.20, 0.40), 2),
            "sim_churn_rate": "High",
            "primary_digital_channels": ["Cash", "UPI"],
            "login_pattern": "Irregular",
            "ecommerce_activity_level": "None",
            "ecommerce_avg_ticket_size": "N/A",
        }
        
        super().__init__(**profile_attributes)

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        avg_monthly_income = random.uniform(min_mod, max_mod)

        self.daily_work_chance = 0.75 * (1 + (class_config['loan_propensity'] * 0.2))
        self.daily_wage_amount = avg_monthly_income / 22
        
        self.remittance_percentage = random.uniform(0.6, 0.8) * (1.1 if financial_personality == 'Saver' else 1)
        self.recharge_chance = 0.05
        
        # ✅ Updated P2P attributes - Daily wage laborers have community networks
        self.worker_network = []  # To be populated by simulation engine
        self.family_recipient = None  # Single family member for regular remittances
        self.p2p_transfer_chance = 0.15 * personality_config.get('spend_chance_mod', 1.0)
        
        # Emergency and community support patterns
        self.emergency_help_chance = 0.03  # 3% chance of emergency help
        self.community_support_chance = 0.08  # 8% chance of community support
        
        # Track last remittance to manage frequency
        self.last_remittance_day = None

        self.balance = random.uniform(50, 200)

    def _handle_daily_income(self, date, events):
        """✅ Separated income handling from P2P transfers."""
        if random.random() < self.daily_work_chance:
            wage_txn = self.log_transaction("CREDIT", "Cash Wage Deposit", self.daily_wage_amount, date, channel="Cash Deposit")
            if wage_txn:
                events.append(wage_txn)

    def _handle_cash_withdrawals(self, date, events):
        """Handles cash withdrawals for daily expenses."""
        if self.balance > 50 and random.random() < 0.4:
            # Keep minimal cash for survival, withdraw rest for daily use
            cash_out_amount = self.balance * random.uniform(0.7, 0.9)
            if cash_out_amount > 20:
                cash_txn = self.log_transaction("DEBIT", "Cash Withdrawal", cash_out_amount, date, channel="ATM")
                if cash_txn:
                    events.append(cash_txn)

    def _handle_family_remittances(self, date, events, context):
        """✅ NEW: Handles regular family remittances after work."""
        # Send money home after getting paid (but not every day to avoid over-sending)
        current_day_key = date.strftime("%Y-%m-%d")
        
        if (self.family_recipient and 
            self.balance >= self.daily_wage_amount and  # Ensure we just got paid
            self.last_remittance_day != current_day_key and
            random.random() < 0.4):  # 40% chance of sending money after work
            
            remittance_amount = self.daily_wage_amount * self.remittance_percentage
            
            if remittance_amount >= 20:  # Minimum threshold
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': self.family_recipient, 
                    'amount': round(remittance_amount, 2), 
                    'desc': 'Family Remittance',
                    'channel': 'UPI'
                })
                self.last_remittance_day = current_day_key

    def _handle_p2p_transfers(self, date, events, context):
        """✅ NEW: Handles transfers within worker community."""
        if self.worker_network and random.random() < self.p2p_transfer_chance:
            recipient = random.choice(self.worker_network)
            
            # Small amounts typical for this economic group
            base_amount = random.uniform(50, 500)
            
            # Adjust based on current balance (can't send what you don't have)
            max_sendable = self.balance * 0.3  # Don't send more than 30% of balance
            amount = min(base_amount, max_sendable)
            
            if amount >= 20:  # Minimum viable transfer
                transfer_desc = random.choice([
                    'Worker Loan', 
                    'Shared Meal', 
                    'Emergency Help', 
                    'Tool Sharing',
                    'Transport Share',
                    'Tea/Snacks',
                    'Mutual Aid'
                ])
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(amount, 2), 
                    'desc': transfer_desc,
                    'channel': 'UPI'
                })

    def _handle_community_support(self, date, events, context):
        """✅ NEW: Handles emergency community support transfers."""
        if (self.worker_network and 
            random.random() < self.community_support_chance and 
            self.balance > 100):  # Only if some balance available
            
            recipient = random.choice(self.worker_network)
            
            # Emergency/community support amounts
            support_amount = random.uniform(100, 300)
            
            # Ensure we don't send more than we can afford
            max_support = self.balance * 0.5  # Maximum 50% for emergency help
            final_amount = min(support_amount, max_support)
            
            if final_amount >= 50:  # Minimum for meaningful support
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': random.choice([
                        'Emergency Support', 
                        'Medical Help', 
                        'Urgent Need',
                        'Community Aid',
                        'Worker Emergency'
                    ]),
                    'channel': 'UPI'
                })

    def _handle_emergency_help(self, date, events, context):
        """✅ NEW: Handles urgent emergency help for fellow workers."""
        if (self.worker_network and 
            random.random() < self.emergency_help_chance and 
            self.balance > 200):  # Need significant balance for emergency help
            
            recipient = random.choice(self.worker_network)
            
            # Emergency amounts - larger but still constrained by economic reality
            emergency_amount = random.uniform(200, 600)
            
            # Can't give more than 60% of balance in emergency
            max_emergency = self.balance * 0.6
            final_amount = min(emergency_amount, max_emergency)
            
            if final_amount >= 100:  # Minimum for emergency
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Worker Emergency Fund',
                    'channel': 'UPI'
                })

    def _handle_recharge(self, date, events):
        """Simulates small, infrequent mobile recharges."""
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([10, 20, 49, 99])
            txn = self.log_transaction("DEBIT", "Sachet Mobile Recharge", recharge_amount, date, channel="UPI")
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling."""
        events = []
        self._handle_daily_income(date, events)
        self._handle_family_remittances(date, events, context)    # ✅ Primary family support
        self._handle_p2p_transfers(date, events, context)         # ✅ Worker community transfers
        self._handle_community_support(date, events, context)     # ✅ Community mutual aid
        self._handle_emergency_help(date, events, context)        # ✅ Emergency support
        self._handle_cash_withdrawals(date, events)
        self._handle_recharge(date, events)
        self._handle_daily_living_expenses(date, events)
        return events
