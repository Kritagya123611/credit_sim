# agents/homemaker.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np

class Homemaker(BaseAgent):
    """
    A multi-dimensional profile for a Homemaker.
    Behavior is modified by the household's economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Saver'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Homemaker"

        # --- RISK SCORE CALCULATION ---
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "10000-30000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Not_Applicable",
            "employment_verification": "Not_Applicable",
            "income_type": "Family_Support",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Child_Education_Plan"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.60, 0.80), 2),
            "ip_consistency_score": 0.98,
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["UPI", "Mobile_Banking"],
            "login_pattern": "Infrequent",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Medium",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_allowance = random.uniform(min_mod, max_mod)

        self.loan_emi_amount = self.monthly_allowance * 0.30
        self.insurance_premium = self.monthly_allowance * 0.10
        self.utility_bill_amount = self.monthly_allowance * 0.15
        self.weekly_grocery_day = 5 # Saturday
        self.school_fee_months = [1, 4, 7, 10]

        self.occasional_spend_chance = 0.08 * personality_config['spend_chance_mod']
        self.shared_device_id = None 

        self.balance = random.uniform(self.monthly_allowance * 0.05, self.monthly_allowance * 0.2)

    def _handle_monthly_income_and_fixed_costs(self, date, events):
        """Handles the monthly allowance and fixed, recurring household payments."""
        if date.day == 1:
            txn = self.log_transaction("CREDIT", "Family Support Transfer", self.monthly_allowance, date)
            if txn: events.append(txn)

        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Home/Car Loan EMI (Co-payment)", self.loan_emi_amount, date)
            if txn: events.append(txn)
            
        if date.day == 15:
            txn = self.log_transaction("DEBIT", "Utility Bill Payment (Gas/DTH)", self.utility_bill_amount, date)
            if txn: events.append(txn)

        if self.has_insurance_payments and date.day == 20:
            txn = self.log_transaction("DEBIT", "Child Education Plan Premium", self.insurance_premium, date)
            if txn: events.append(txn)

    def _handle_household_spending(self, date, events):
        """Simulates structured and occasional spending for the household."""
        if date.weekday() == self.weekly_grocery_day:
            grocery_amount = self.monthly_allowance * random.uniform(0.1, 0.15)
            txn = self.log_transaction("DEBIT", "UPI - Weekly Groceries", grocery_amount, date)
            if txn: events.append(txn)

        if date.month in self.school_fee_months and date.day == 5:
            fee_amount = self.monthly_allowance * random.uniform(0.5, 1.5)
            txn = self.log_transaction("DEBIT", "Netbanking - School Fees", fee_amount, date)
            if txn: events.append(txn)

        if random.random() < self.occasional_spend_chance:
            amount = self.monthly_allowance * random.uniform(0.05, 0.1)
            category = random.choice(["Kids Clothing", "Home Goods", "Online Pharmacy"])
            txn = self.log_transaction("DEBIT", f"E-commerce - {category}", amount, date)
            if txn: events.append(txn)
    
    def act(self, date: datetime, **context):
        """
        Simulates the Homemaker's role as a household financial manager.
        """
        events = []
        self._handle_monthly_income_and_fixed_costs(date, events)
        self._handle_household_spending(date, events)
        return events