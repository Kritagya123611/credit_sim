# agents/homemaker.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class Homemaker(BaseAgent):
    def __init__(self):
        profile_attributes = {
            "archetype_name": "Homemaker",
            "risk_profile": "Very_High",
            "employment_status": "Not_Applicable",
            "employment_verification": "Not_Applicable",
            "income_type": "Family_Support",
            "avg_monthly_income_range": "10000-30000",
            "income_pattern": "Fixed_Date",

            # Financial Habits
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": True,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Child_Education_Plan"],

            # Utility Payments
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Prepaid",

            # Digital Footprint
            "device_consistency_score": 0.7,
            "ip_consistency_score": 0.98,
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["UPI", "Mobile_Banking"],
            "login_pattern": "Infrequent",

            # E-commerce
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Medium",
        }

        super().__init__(**profile_attributes)

        min_inc, max_inc = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_inc, max_inc)

        self.loan_emi = self.monthly_income * 0.25
        self.utility_cost = random.uniform(1200, 2200)
        self.insurance_cost = 1500

        self.ecom_chance = 0.4
        self.balance = random.uniform(3000, 8000)

    def act(self, date: datetime):
        events = []

        # Monthly family support income
        if date.day == 5:
            txn = self.log_transaction("CREDIT", "Monthly Family Transfer", self.monthly_income, date)
            if txn: events.append(txn)

        # EMI
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", self.loan_emi, date)
            if txn: events.append(txn)

        # Insurance
        if self.has_insurance_payments and date.day == 20:
            txn = self.log_transaction("DEBIT", "Child Education Insurance", self.insurance_cost, date)
            if txn: events.append(txn)

        # Utility
        if date.day == 15:
            txn = self.log_transaction("DEBIT", "Utility Payment", self.utility_cost, date)
            if txn: events.append(txn)

        # E-commerce shopping
        if random.random() < self.ecom_chance:
            amount = random.uniform(1000, 5000)
            txn = self.log_transaction("DEBIT", "Online Purchase", amount, date)
            if txn: events.append(txn)

        return events
