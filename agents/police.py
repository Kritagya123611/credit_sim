# agents/police.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np

class PoliceOfficer(BaseAgent):
    """
    A multi-dimensional profile for Police / Security Staff.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Saver'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Police / Security Personnel"

        # --- RISK SCORE CALCULATION ---
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "30000-50000"
        min_sal, max_sal = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_sal * income_multiplier)}-{int(max_sal * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Salaried",
            "employment_verification": "GOVT_Verified",
            "income_type": "Uniformed_Services_Salary",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "Medium",
            "has_investment_activity": True,
            "investment_types": ["LIC", "FD"],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["LIC", "Government_Schemes"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid",
            "device_consistency_score": round(random.uniform(0.92, 0.98), 2),
            "ip_consistency_score": round(random.uniform(0.75, 0.85), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Mobile_Banking", "UPI"],
            "login_pattern": "Shift_Work_Night_Activity",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Medium",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.salary_day = 1
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_mod, max_mod)

        self.remittance_percentage = 0.30 * (1.2 if financial_personality == 'Saver' else 1)
        self.emi_percentage = 0.20
        self.investment_percentage = 0.10 * personality_config['invest_chance_mod']
        self.insurance_percentage = 0.08
        
        self.daily_spend_chance = 0.60 * personality_config['spend_chance_mod']
        
        self.balance = random.uniform(self.salary_amount * 0.15, self.salary_amount * 0.4)

    def _handle_fixed_monthly_events(self, date, events):
        """Handles salary, remittances, and other fixed-date debits."""
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Salary Deposit", self.salary_amount, date)
            if txn: 
                events.append(txn)
                remittance_amount = self.salary_amount * self.remittance_percentage
                remit_txn = self.log_transaction("DEBIT", "Family Remittance (P2P)", remittance_amount, date)
                if remit_txn: events.append(remit_txn)

        if self.has_loan_emi and date.day == 10:
            emi_amount = self.salary_amount * self.emi_percentage
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", emi_amount, date)
            if txn: events.append(txn)

        if self.has_insurance_payments and date.day == 15:
            insurance_total = self.salary_amount * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "Insurance Premium (LIC)", insurance_total, date)
            if txn: events.append(txn)

        if self.has_investment_activity and date.day == 20:
            invest_amt = self.salary_amount * self.investment_percentage
            txn = self.log_transaction("DEBIT", "Investment - FD", invest_amt, date)
            if txn: events.append(txn)

    def _handle_daily_spending(self, date, events):
        """
        Simulates daily spending that can occur 24/7 due to shift work.
        """
        if random.random() < self.daily_spend_chance:
            is_night_shift = random.random() < 0.4
            
            if is_night_shift:
                description = "Night Duty - Food/Tea"
                amount = random.uniform(100, 300)
            else:
                description = "Daily Expense - Groceries/Misc"
                amount = random.uniform(200, 800)
                
            txn = self.log_transaction("DEBIT", description, amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the agent's disciplined financial life.
        """
        events = []
        self._handle_fixed_monthly_events(date, events)
        self._handle_daily_spending(date, events)
        return events