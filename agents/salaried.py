# agents/salaried_professional.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np

class SalariedProfessional(BaseAgent):
    """
    A multi-dimensional profile for a Salaried Professional.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Middle', financial_personality='Rational_Investor'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Salaried Professional"

        # --- RISK SCORE CALCULATION ---
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "40000-80000"
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
            "employment_verification": "EPFO_Verified",
            "income_type": "Salary",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "High" if financial_personality == "Saver" else "Medium",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid",
            "device_consistency_score": round(random.uniform(0.90, 0.98), 2),
            "ip_consistency_score": round(random.uniform(0.85, 0.95), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Mobile_Banking"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Medium",
        }
        
        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.salary_day = random.randint(1, 5)
        min_sal_mod, max_sal_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal_mod, max_sal_mod)
        
        self.emi_percentage = 0.25
        self.investment_percentage = 0.15 * personality_config['invest_chance_mod']
        self.insurance_percentage = 0.05
        self.utility_bill_percentage = 0.05
        
        self.ecommerce_spend_chance = 0.15 * personality_config['spend_chance_mod']
        self.weekday_spend_chance = 0.50 * personality_config['spend_chance_mod']
        self.weekend_spend_chance = 0.70 * personality_config['spend_chance_mod']
        
        self.annual_bonus_month = 3
        self.has_received_bonus_this_year = False

        self.balance = random.uniform(self.salary_amount * 0.2, self.salary_amount * 0.5)

    def _handle_monthly_credits(self, date, events):
        """Handles salary and annual bonus credits."""
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Salary Deposit", self.salary_amount, date)
            if txn: events.append(txn)
            if date.month == 1:
                self.has_received_bonus_this_year = False

        if date.month == self.annual_bonus_month and date.day == self.salary_day and not self.has_received_bonus_this_year:
            bonus_amount = self.salary_amount * random.uniform(1.5, 3.0)
            txn = self.log_transaction("CREDIT", "Annual Bonus", bonus_amount, date)
            if txn: events.append(txn)
            self.has_received_bonus_this_year = True

    def _handle_recurring_debits(self, date, events):
        """Handles fixed monthly payments like EMIs, insurance, and bills."""
        if self.has_loan_emi and date.day == 10:
            emi_amount = self.salary_amount * self.emi_percentage
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", emi_amount, date)
            if txn: events.append(txn)

        if self.has_insurance_payments and date.day == 15:
            insurance_total = self.salary_amount * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "Insurance Premium", insurance_total, date)
            if txn: events.append(txn)

        if self.has_investment_activity and date.day == 20:
            invest_amt = self.salary_amount * self.investment_percentage
            txn = self.log_transaction("DEBIT", "SIP Investment", invest_amt, date)
            if txn: events.append(txn)

        if date.day == 25:
            bill = self.salary_amount * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", bill, date)
            if txn: events.append(txn)

    def _handle_daily_spending(self, date, events):
        """Handles probabilistic daily spending which varies by day type."""
        ecommerce_chance = self.ecommerce_spend_chance * (2.5 if self.has_received_bonus_this_year else 1)
        if random.random() < ecommerce_chance:
            ecommerce_amt = random.uniform(1000, 5000)
            txn = self.log_transaction("DEBIT", "E-commerce Purchase", ecommerce_amt, date)
            if txn: events.append(txn)

        is_weekend = date.weekday() >= 5
        if is_weekend:
            if random.random() < self.weekend_spend_chance:
                spend = random.uniform(500, 2500)
                txn = self.log_transaction("DEBIT", "Weekend Entertainment/Dining", spend, date)
                if txn: events.append(txn)
        else:
            if random.random() < self.weekday_spend_chance:
                spend_type = random.choice(["Transport", "Groceries", "Lunch"])
                spend_amount = random.uniform(150, 800)
                txn = self.log_transaction("DEBIT", f"UPI Spend - {spend_type}", spend_amount, date)
                if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the agent's daily financial actions by calling helper methods.
        """
        events = []
        self._handle_monthly_credits(date, events)
        self._handle_recurring_debits(date, events)
        self._handle_daily_spending(date, events)
        # --- ADDED: Universal daily spending ---
        self._handle_daily_living_expenses(date, events)
        return events