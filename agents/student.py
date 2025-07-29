import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np


class Student(BaseAgent):
    def __init__(self, economic_class='Lower_Middle', financial_personality='Over_Spender'):
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Student"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "3000-10000"
        min_allowance, max_allowance = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_allowance * income_multiplier)}-{int(max_allowance * income_multiplier)}"
        
        profile_attributes = {
            "archetype_name": archetype_name, 
            "risk_profile": risk_profile_category, 
            "risk_score": risk_score,
            "economic_class": economic_class, 
            "financial_personality": financial_personality,
            "employment_status": "Not_Applicable", 
            "employment_verification": "Not_Applicable",
            "income_type": "Allowance", 
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Irregular", 
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False, 
            "investment_types": [],
            "has_loan_emi": False, 
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False, 
            "insurance_types": [],
            "utility_payment_status": "N/A", 
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.70, 0.85), 2),
            "ip_consistency_score": round(random.uniform(0.40, 0.60), 2),
            "sim_churn_rate": "Medium", 
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Late_Night_Activity", 
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "Low",
        }
        
        super().__init__(**profile_attributes)

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.allowance_amount = random.uniform(min_mod, max_mod)
        self.allowance_days = sorted(random.sample(range(2, 28), random.randint(1, 2)))

        self.daily_spend_chance = 0.75 * personality_config['spend_chance_mod']
        self.recharge_chance = 0.08
        self.bnpl_chance = 0.15 if financial_personality == 'Over_Spender' else 0.05
        
        # ✅ Updated P2P logic - separated from spending
        self.p2p_transfer_chance = 0.25 * personality_config['spend_chance_mod']  # Higher chance for students

        self.bnpl_repayments = {}
        # ✅ To be populated by simulation engine
        self.contacts = []

        self.balance = random.uniform(100, 500)

    def _handle_income(self, date, events):
        """Handles irregular allowance/family support income."""
        if date.day in self.allowance_days:
            txn = self.log_transaction("CREDIT", "Allowance/Family Support", self.allowance_amount, date, channel="P2P")
            if txn: events.append(txn)

    def _handle_spending(self, date, events):
        """Handles regular student spending patterns."""
        # Handle BNPL repayments
        if date.date() in self.bnpl_repayments:
            amount_due = self.bnpl_repayments.pop(date.date())
            txn = self.log_transaction("DEBIT", "BNPL Repayment", amount_due, date, channel="UPI")
            if txn: events.append(txn)

        # Handle mobile recharge
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([99, 149, 199, 239])
            txn = self.log_transaction("DEBIT", "Prepaid Mobile Recharge", recharge_amount, date, channel="UPI")
            if txn: events.append(txn)

        # Handle regular spending
        if random.random() < self.daily_spend_chance:
            spend_category = random.choice([
                "Food_Delivery", "Cab_Service", "OTT_Subscription", "Groceries", "Gaming_Purchase", "Stationery"
            ])
            
            if spend_category == "Food_Delivery" and random.random() < self.bnpl_chance:
                # BNPL transaction - no immediate debit
                spend_amount = random.uniform(100, 600)
                repayment_date = date.date() + timedelta(days=15)
                self.bnpl_repayments[repayment_date] = self.bnpl_repayments.get(repayment_date, 0) + spend_amount
                # Log as a BNPL transaction (optional - could track differently)
                txn = self.log_transaction("DEBIT", f"BNPL - {spend_category}", 0, date, channel="BNPL")
                if txn: events.append(txn)
            else:
                spend_amount = random.uniform(100, 600)
                txn = self.log_transaction("DEBIT", f"UPI Spend - {spend_category}", spend_amount, date, channel="UPI")
                if txn: events.append(txn)

    def _handle_p2p_transfers(self, date, events, context):
        """✅ NEW: Dedicated P2P transfer method for student group expenses."""
        if self.contacts and random.random() < self.p2p_transfer_chance:
            recipient = random.choice(self.contacts)
            # Students typically send smaller amounts
            amount = random.uniform(100, 800)
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': amount, 
                'desc': random.choice([
                    'Group Study Materials', 
                    'Shared Food Order', 
                    'Project Expense', 
                    'Friend Loan',
                    'Event Contribution',
                    'Textbook Sharing'
                ]),
                'channel': 'UPI'
            })

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes dedicated P2P transfer handling."""
        events = []
        self._handle_income(date, events)
        self._handle_spending(date, events)
        self._handle_p2p_transfers(date, events, context)  # ✅ Added this line
        self._handle_daily_living_expenses(date, events)
        return events
