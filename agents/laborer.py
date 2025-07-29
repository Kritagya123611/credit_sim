# agents/daily_wage_laborer.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES

class DailyWageLaborer(BaseAgent):
    """
    A multi-dimensional profile for a Daily Wage Laborer.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower', financial_personality='Saver'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        # Note: These dimensions have a smaller effect on this archetype due to economic constraints
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])

        base_income_range = "7000-15000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Daily Wage Laborer",
            "risk_profile": "Very_High",
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

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        avg_monthly_income = random.uniform(min_mod, max_mod)

        # Higher economic class means slightly more consistent work
        self.daily_work_chance = 0.75 * (1 + (class_config['loan_propensity'] * 0.2)) # Use loan_propensity as proxy for stability
        self.daily_wage_amount = avg_monthly_income / 22 # Assume ~22 working days
        
        # "Saver" personality remits a higher percentage
        self.remittance_percentage = random.uniform(0.6, 0.8) * (1.1 if financial_personality == 'Saver' else 1)
        
        self.recharge_chance = 0.05

        self.balance = random.uniform(50, 200)

    def _handle_work_and_remittance(self, date, events):
        """Simulates getting paid for a day's work and immediately sending money home."""
        if random.random() < self.daily_work_chance:
            wage_txn = self.log_transaction("CREDIT", "Cash Wage Deposit", self.daily_wage_amount, date)
            if wage_txn:
                events.append(wage_txn)

                remittance_amount = self.daily_wage_amount * self.remittance_percentage
                remit_txn = self.log_transaction("DEBIT", "Family Remittance (P2P)", remittance_amount, date)
                if remit_txn:
                    events.append(remit_txn)
                
                if self.balance > 0:
                    cash_out_amount = self.balance * random.uniform(0.8, 1.0)
                    if cash_out_amount > 10:
                        cash_txn = self.log_transaction("DEBIT", "Cash Withdrawal for Expenses", cash_out_amount, date)
                        if cash_txn:
                            events.append(cash_txn)

    def _handle_recharge(self, date, events):
        """Simulates small, infrequent mobile recharges."""
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([10, 20, 49, 99])
            txn = self.log_transaction("DEBIT", "Sachet Mobile Recharge", recharge_amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the daily "spike and drain" financial cycle of a daily wage laborer.
        """
        events = []
        self._handle_work_and_remittance(date, events)
        self._handle_recharge(date, events)
        return events