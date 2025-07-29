# agents/government_employee.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES

class GovernmentEmployee(BaseAgent):
    """
    A multi-dimensional profile for a Government Employee.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Middle', financial_personality='Saver'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])

        base_income_range = "35000-70000"
        min_sal, max_sal = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_sal * income_multiplier)}-{int(max_sal * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Government Employee",
            "risk_profile": "Very_Low",
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Salaried",
            "employment_verification": "GOVT_Verified",
            "income_type": "Government_Salary",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "High",
            "has_investment_activity": True,
            "investment_types": personality_config['investment_types'], # Use personality-defined investments
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["LIC", "Government_Schemes"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid",
            "device_consistency_score": round(random.uniform(0.95, 0.99), 2),
            "ip_consistency_score": round(random.uniform(0.92, 0.98), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Medium",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.salary_day = 1
        min_sal_mod, max_sal_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal_mod, max_sal_mod)
        
        # Define conservative expense percentages, modified by personality
        self.emi_percentage = 0.30
        self.investment_percentage = 0.10 * personality_config['invest_chance_mod']
        self.insurance_percentage = 0.08
        self.utility_bill_percentage = 0.05
        
        # Very low probability for discretionary spending, modified by personality
        self.ecommerce_spend_chance = 0.05 * personality_config['spend_chance_mod']

        # Set a healthy starting balance
        self.balance = random.uniform(self.salary_amount * 0.3, self.salary_amount * 0.8)

    def _handle_recurring_events(self, date, events):
        """Handles fixed monthly credits and debits."""
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Govt Salary Deposit", self.salary_amount, date)
            if txn: events.append(txn)
            
        if self.has_loan_emi and date.day == 5:
            emi_amount = self.salary_amount * self.emi_percentage
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", emi_amount, date)
            if txn: events.append(txn)

        if self.has_insurance_payments and date.day == 10:
            insurance_total = self.salary_amount * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "LIC Premium", insurance_total, date)
            if txn: events.append(txn)

        if self.has_investment_activity and date.day == 15:
            invest_amt = self.salary_amount * self.investment_percentage
            txn = self.log_transaction("DEBIT", "PPF/FD Investment", invest_amt, date)
            if txn: events.append(txn)
            
        if date.day == 20:
            bill = self.salary_amount * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", bill, date)
            if txn: events.append(txn)

    def _handle_daily_spending(self, date, events):
        """Handles infrequent, need-based daily spending."""
        if random.random() < self.ecommerce_spend_chance:
            ecommerce_amt = random.uniform(500, 2500)
            txn = self.log_transaction("DEBIT", "E-commerce Purchase (Essentials)", ecommerce_amt, date)
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the agent's highly predictable and disciplined daily financial actions.
        """
        events = []
        self._handle_recurring_events(date, events)
        self._handle_daily_spending(date, events)
        return events