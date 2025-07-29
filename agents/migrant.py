# agents/migrant.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np


class MigrantWorker(BaseAgent):
    """
    A multi-dimensional profile for a Migrant Worker.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Migrant Worker"

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
            "income_type": "Wages",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Weekly_or_Monthly",
            "savings_retention_rate": "Near_Zero",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "N/A",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.40, 0.60), 2),
            "ip_consistency_score": round(random.uniform(0.30, 0.50), 2),
            "sim_churn_rate": "High",
            "primary_digital_channels": ["UPI", "IMPS"],
            "login_pattern": "Remittance_Cycle",
            "ecommerce_activity_level": "None",
            "ecommerce_avg_ticket_size": "N/A",
        }
        
        super().__init__(**profile_attributes)

        self.home_state = random.choice(["Uttar Pradesh", "Bihar", "Odisha", "Rajasthan"])
        self.work_city = random.choice(["Mumbai", "Delhi", "Bengaluru", "Surat"])

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_mod, max_mod)
        
        self.pay_cycle = "monthly" if economic_class == 'Lower_Middle' else "weekly"
        
        self.weekly_wage = self.monthly_income / 4
        self.monthly_pay_day = random.randint(1, 5)
        self.weekly_pay_day = 6
        
        self.remittance_percentage = random.uniform(0.6, 0.85) * (1.1 if financial_personality == 'Saver' else 1)
        self.recharge_chance = 0.07

        # ✅ Updated P2P attributes - Migrant workers primarily send money home
        self.family_back_home = []  # To be populated by simulation engine
        self.p2p_transfer_chance = 0.30 * personality_config.get('spend_chance_mod', 1.0)  # High frequency for remittances
        
        # Emergency and festival transfer patterns
        self.emergency_transfer_chance = 0.05  # 5% chance of emergency transfers
        self.festival_months = [3, 10, 11]  # Holi, Diwali, etc.
        
        # Track last remittance to avoid over-sending
        self.last_remittance_date = None

        self.balance = random.uniform(100, 500)

    def _handle_income(self, date, events):
        """✅ Separated income handling from P2P transfers."""
        is_payday = False
        wage_amount = 0

        if self.pay_cycle == "weekly" and date.weekday() == self.weekly_pay_day:
            is_payday = True
            wage_amount = self.weekly_wage
        elif self.pay_cycle == "monthly" and date.day == self.monthly_pay_day:
            is_payday = True
            wage_amount = self.monthly_income

        if is_payday:
            wage_txn = self.log_transaction("CREDIT", f"Wage ({self.work_city})", wage_amount, date, channel="Bank Transfer")
            if wage_txn:
                events.append(wage_txn)

    def _handle_cash_withdrawals(self, date, events):
        """Handles cash withdrawals for daily expenses."""
        # Cash out most of remaining balance after remittances
        if self.balance > 200 and random.random() < 0.3:
            cash_out_amount = self.balance * random.uniform(0.6, 0.9)
            cash_txn = self.log_transaction("DEBIT", f"Cash Withdrawal ({self.work_city})", cash_out_amount, date, channel="ATM")
            if cash_txn:
                events.append(cash_txn)

    def _handle_regular_remittances(self, date, events, context):
        """✅ NEW: Handles regular scheduled remittances home."""
        # Send money home after payday (within 2 days of receiving wage)
        is_remittance_window = False
        
        if self.pay_cycle == "weekly":
            # Send within 2 days of weekly payday
            days_since_payday = (date.weekday() - self.weekly_pay_day) % 7
            is_remittance_window = days_since_payday <= 2
        elif self.pay_cycle == "monthly":
            # Send within 3 days of monthly payday
            if date.day >= self.monthly_pay_day and date.day <= self.monthly_pay_day + 3:
                is_remittance_window = True

        if is_remittance_window and self.family_back_home:
            # Check if already sent this cycle
            if self.last_remittance_date != date.strftime("%Y-%m-%d"):
                recipient = random.choice(self.family_back_home)
                
                # Calculate remittance based on available balance and percentage
                available_for_remittance = self.balance * self.remittance_percentage
                
                if available_for_remittance >= 500:  # Minimum threshold
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': recipient, 
                        'amount': available_for_remittance, 
                        'desc': f'Family Remittance to {self.home_state}',
                        'channel': 'UPI'
                    })
                    self.last_remittance_date = date.strftime("%Y-%m-%d")

    def _handle_p2p_transfers(self, date, events, context):
        """✅ NEW: Handles additional P2P transfers beyond regular remittances."""
        if self.family_back_home and random.random() < self.p2p_transfer_chance:
            recipient = random.choice(self.family_back_home)
            
            # Base amount for non-remittance transfers
            base_amount = random.uniform(300, 1500)
            
            # Increase during festival months
            if date.month in self.festival_months:
                base_amount *= random.uniform(1.5, 2.5)
                transfer_desc = random.choice([
                    'Festival Money', 
                    'Religious Celebration', 
                    'Holiday Support', 
                    'Special Occasion'
                ])
            else:
                transfer_desc = random.choice([
                    'Emergency Support', 
                    'Medical Help', 
                    'Additional Support',
                    'Urgent Need',
                    'Family Emergency'
                ])
            
            # Only send if sufficient balance
            if self.balance > base_amount + 200:  # Keep some buffer
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(base_amount, 2), 
                    'desc': transfer_desc,
                    'channel': 'UPI'
                })

    def _handle_emergency_transfers(self, date, events, context):
        """✅ NEW: Handles urgent emergency transfers home."""
        if (self.family_back_home and 
            random.random() < self.emergency_transfer_chance and 
            self.balance > 1000):  # Need significant balance for emergency
            
            recipient = random.choice(self.family_back_home)
            # Emergency transfers are typically larger
            emergency_amount = self.balance * random.uniform(0.4, 0.7)
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(emergency_amount, 2), 
                'desc': 'Family Emergency',
                'channel': 'UPI'
            })

    def _handle_recharge(self, date, events):
        """Simulates small, infrequent mobile recharges."""
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([49, 99, 149])
            txn = self.log_transaction("DEBIT", "Prepaid Mobile Recharge", recharge_amount, date, channel="UPI")
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling."""
        events = []
        self._handle_income(date, events)
        self._handle_regular_remittances(date, events, context)  # ✅ Primary remittance logic
        self._handle_p2p_transfers(date, events, context)        # ✅ Additional transfers
        self._handle_emergency_transfers(date, events, context)  # ✅ Emergency transfers
        self._handle_cash_withdrawals(date, events)
        self._handle_recharge(date, events)
        self._handle_daily_living_expenses(date, events)
        return events
