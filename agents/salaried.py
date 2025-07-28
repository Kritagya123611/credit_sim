# agents/salaried_professional.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent # Make sure this import path is correct

class SalariedProfessional(BaseAgent):
    """
    A specific agent profile for a Salaried Professional.
    It inherits from the BaseAgent class and defines its unique attributes and daily actions.
    """
    def __init__(self):
        # 1. Define all profile attributes in a dictionary
        # This matches the STANDARDIZED_FIELDS in the BaseAgent
        profile_attributes = {
            "archetype_name": "Salaried Professional",
            "risk_profile": "Low",
            "employment_status": "Salaried",
            "employment_verification": "EPFO_Verified",
            "income_type": "Salary",
            "avg_monthly_income_range": "40000-80000",
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "High",
            "has_investment_activity": True,
            "investment_types": ["SIP", "Mutual_Funds"],
            "has_loan_emi": True,
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

        # 2. Call the parent's __init__ method, passing the attributes
        super().__init__(**profile_attributes)

        # 3. Define behavioral parameters for the simulation
        self.salary_day = random.randint(1, 5)
        # Unpack the range string to get min and max for salary calculation
        min_sal, max_sal = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal, max_sal)
        
        # Define expenses as percentages of salary for realistic budgeting
        self.emi_percentage = 0.25
        self.investment_percentage = 0.15
        self.insurance_percentage = 0.05
        self.utility_bill_percentage = 0.05
        
        # Define daily spending probabilities
        self.ecommerce_spend_chance = 0.15  # 15% chance per day
        self.discretionary_spend_chance = 0.40 # 40% chance on weekdays

        # Set a realistic starting balance
        self.balance = random.uniform(self.salary_amount * 0.1, self.salary_amount * 0.4)

    def act(self, date: datetime):
        """Simulates the agent's daily financial actions based on their profile."""
        events = []

        # --- Recurring Monthly Events ---
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Salary Deposit", self.salary_amount, date)
            if txn: events.append(txn)

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

        # --- Probabilistic Daily Events ---
        if random.random() < self.ecommerce_spend_chance:
            ecommerce_amt = random.uniform(1000, 4000)
            txn = self.log_transaction("DEBIT", "E-commerce Purchase", ecommerce_amt, date)
            if txn: events.append(txn)

        if date.weekday() < 5 and random.random() < self.discretionary_spend_chance:
            spend = random.uniform(300, 1500)
            txn = self.log_transaction("DEBIT", "UPI/Card Spend", spend, date)
            if txn: events.append(txn)

        return events