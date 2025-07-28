# agents/police.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class PoliceOfficer(BaseAgent):
    def __init__(self):
        profile_attributes = {
            "archetype_name": "Police / Security Staff",
            "risk_profile": "Very_Low",
            "employment_status": "Salaried",
            "employment_verification": "GOVT_Verified",
            "income_type": "Uniformed_Services_Salary",
            "avg_monthly_income_range": "30000-50000",
            "income_pattern": "Fixed_Date",

            # Financial Habits
            "savings_retention_rate": "Medium",
            "has_investment_activity": True,
            "investment_types": ["LIC", "FD"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["LIC", "Government_Schemes"],

            # Utility Payments
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid",

            # Digital Footprint
            "device_consistency_score": 0.95,
            "ip_consistency_score": 0.80,
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Mobile_Banking", "UPI"],
            "login_pattern": "Shift_Work_Night_Activity",

            # E-commerce
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Medium",
        }

        super().__init__(**profile_attributes)

        self.salary_day = 1
        min_sal, max_sal = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal, max_sal)

        self.emi_percentage = 0.20
        self.investment_percentage = 0.15
        self.insurance_percentage = 0.08
        self.utility_bill_percentage = 0.05

        self.ecommerce_spend_chance = 0.12
        self.discretionary_spend_chance = 0.25

        self.balance = random.uniform(self.salary_amount * 0.15, self.salary_amount * 0.4)

    def act(self, date: datetime):
        events = []

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
            txn = self.log_transaction("DEBIT", "Investment - LIC/FD", invest_amt, date)
            if txn: events.append(txn)

        if date.day == 25:
            bill = self.salary_amount * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Utility Bill", bill, date)
            if txn: events.append(txn)

        if random.random() < self.ecommerce_spend_chance:
            ecommerce_amt = random.uniform(800, 2500)
            txn = self.log_transaction("DEBIT", "E-commerce Purchase", ecommerce_amt, date)
            if txn: events.append(txn)

        if random.random() < self.discretionary_spend_chance:
            spend = random.uniform(200, 1000)
            txn = self.log_transaction("DEBIT", "Daily Expense", spend, date)
            if txn: events.append(txn)

        return events
