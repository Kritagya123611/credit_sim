# agents/lawyer.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class Lawyer(BaseAgent):
    def __init__(self):
        profile_attributes = {
            "archetype_name": "Lawyer / Consultant",
            "risk_profile": "Medium",
            "employment_status": "Self-Employed_Professional",
            "employment_verification": "Professional_License_Verified",
            "income_type": "Professional_Fees",
            "avg_monthly_income_range": "50000-200000",
            "income_pattern": "Lumpy",

            # Financial Habits
            "savings_retention_rate": "Medium",
            "has_investment_activity": True,
            "investment_types": ["Equity", "Real_Estate"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life", "Prof_Indemnity"],

            # Utility Payments
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "High-Value_Postpaid",

            # Digital Footprint
            "device_consistency_score": 0.9,
            "ip_consistency_score": 0.85,
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Cards"],
            "login_pattern": "Structured_Daytime",

            # E-commerce
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "High",
        }

        super().__init__(**profile_attributes)

        # Define lumpy income logic
        min_inc, max_inc = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_inc, max_inc)
        self.lump_dates = random.sample(range(1, 29), k=3)
        self.lump_amounts = [self.monthly_income * random.uniform(0.2, 0.5) for _ in self.lump_dates]

        # Expense ratios
        self.emi_pct = 0.20
        self.insurance_pct = 0.05
        self.utility_pct = 0.03

        # Probabilistic expenses
        self.ecommerce_chance = 0.3
        self.daily_spend_chance = 0.5

        self.balance = random.uniform(5000, 30000)

    def act(self, date: datetime):
        events = []

        # Lumpy Income
        if date.day in self.lump_dates:
            idx = self.lump_dates.index(date.day)
            amount = self.lump_amounts[idx]
            txn = self.log_transaction("CREDIT", "Professional Fee Received", amount, date)
            if txn: events.append(txn)

        # EMI
        if self.has_loan_emi and date.day == 10:
            emi = self.monthly_income * self.emi_pct
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", emi, date)
            if txn: events.append(txn)

        # Insurance
        if self.has_insurance_payments and date.day == 15:
            ins_amt = self.monthly_income * self.insurance_pct
            txn = self.log_transaction("DEBIT", "Insurance Premiums", ins_amt, date)
            if txn: events.append(txn)

        # Utilities
        if date.day == 20:
            utility_amt = self.monthly_income * self.utility_pct
            txn = self.log_transaction("DEBIT", "Utility Bill", utility_amt, date)
            if txn: events.append(txn)

        # E-commerce
        if random.random() < self.ecommerce_chance:
            amt = random.uniform(3000, 10000)
            txn = self.log_transaction("DEBIT", "Luxury/Online Purchase", amt, date)
            if txn: events.append(txn)

        # Daily Card/Netbanking Spend
        if random.random() < self.daily_spend_chance:
            spend = random.uniform(500, 3000)
            txn = self.log_transaction("DEBIT", "Daily Spend", spend, date)
            if txn: events.append(txn)

        return events
