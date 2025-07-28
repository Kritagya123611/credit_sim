# agents/senior.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class SeniorCitizen(BaseAgent):
    def __init__(self):
        profile_attributes = {
            "archetype_name": "Retired Senior Citizen",
            "risk_profile": "Very_Low",
            "employment_status": "Not_Applicable",
            "employment_verification": "Pensioner_ID_Verified",
            "income_type": "Pension, Rent",
            "avg_monthly_income_range": "10000-30000",
            "income_pattern": "Fixed_Date",

            # Financial Habits
            "savings_retention_rate": "High",
            "has_investment_activity": True,
            "investment_types": ["FD", "Senior_Citizen_Schemes"],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": True,
            "insurance_types": ["Health"],

            # Utility Payments
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Basic_Postpaid",

            # Digital Footprint
            "device_consistency_score": 0.99,
            "ip_consistency_score": 0.99,
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Branch"],
            "login_pattern": "Infrequent",

            # E-commerce
            "ecommerce_activity_level": "None",
            "ecommerce_avg_ticket_size": "None",
        }

        super().__init__(**profile_attributes)

        self.pension_day = 1
        min_income, max_income = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_income, max_income)

        self.investment_percentage = 0.10
        self.insurance_percentage = 0.08
        self.utility_bill_percentage = 0.05

        self.discretionary_spend_chance = 0.10  # Low chance, infrequent spending

        self.balance = random.uniform(self.monthly_income * 0.3, self.monthly_income * 0.7)

    def act(self, date: datetime):
        events = []

        if date.day == self.pension_day:
            txn = self.log_transaction("CREDIT", "Pension/Rent Deposit", self.monthly_income, date)
            if txn: events.append(txn)

        if self.has_insurance_payments and date.day == 15:
            insurance_amt = self.monthly_income * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "Health Insurance Premium", insurance_amt, date)
            if txn: events.append(txn)

        if self.has_investment_activity and date.day == 20:
            invest_amt = self.monthly_income * self.investment_percentage
            txn = self.log_transaction("DEBIT", "Fixed Deposit Investment", invest_amt, date)
            if txn: events.append(txn)

        if date.day == 25:
            bill = self.monthly_income * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", bill, date)
            if txn: events.append(txn)

        if random.random() < self.discretionary_spend_chance:
            spend_amt = random.uniform(300, 1000)
            txn = self.log_transaction("DEBIT", "Pharmacy/Essentials", spend_amt, date)
            if txn: events.append(txn)

        return events
