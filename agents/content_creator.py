# agents/content_creator.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class ContentCreator(BaseAgent):
    def __init__(self):
        profile_attributes = {
            "archetype_name": "Influencer / Content Creator",
            "risk_profile": "High",
            "employment_status": "Self-Employed",
            "employment_verification": "ITR_Inconsistent",
            "income_type": "Sponsorships, Platform_Payouts",
            "avg_monthly_income_range": "20000-100000",
            "income_pattern": "Erratic_High_Variance",

            # Financial Habits
            "savings_retention_rate": "Low",
            "has_investment_activity": True,
            "investment_types": ["Crypto", "Stocks"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],

            # Utility Payments
            "utility_payment_status": "Occasionally_Late",
            "mobile_plan_type": "High-Value_Postpaid",

            # Digital Footprint
            "device_consistency_score": 0.6,
            "ip_consistency_score": 0.5,
            "sim_churn_rate": "High",
            "primary_digital_channels": ["All"],
            "login_pattern": "Geographically_Dynamic",

            # E-commerce
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
        }

        super().__init__(**profile_attributes)

        min_inc, max_inc = map(int, self.avg_monthly_income_range.split('-'))
        self.income_fluctuation = random.uniform(0.5, 1.5)
        self.monthly_income = random.uniform(min_inc, max_inc) * self.income_fluctuation

        self.emi_percentage = 0.3
        self.investment_chance = 0.4
        self.utility_bill = random.uniform(1000, 3000)
        self.ecommerce_chance = 0.7

        self.balance = random.uniform(1000, 8000)

    def act(self, date: datetime):
        events = []

        # --- Random Sponsorship or Platform Income ---
        if random.random() < 0.3:
            payout = random.uniform(5000, self.monthly_income)
            txn = self.log_transaction("CREDIT", "Brand/Platform Payout", payout, date)
            if txn: events.append(txn)

        # --- EMI ---
        if self.has_loan_emi and date.day == 10:
            emi_amt = self.monthly_income * self.emi_percentage
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", emi_amt, date)
            if txn: events.append(txn)

        # --- Investment (occasionally) ---
        if self.has_investment_activity and random.random() < self.investment_chance:
            invest_amt = random.uniform(2000, 10000)
            txn = self.log_transaction("DEBIT", "Crypto/Stock Investment", invest_amt, date)
            if txn: events.append(txn)

        # --- Utility Bill (occasionally late) ---
        if date.day in [18, 20, 25] and random.random() < 0.4:
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", self.utility_bill, date)
            if txn: events.append(txn)

        # --- High e-commerce spending ---
        if random.random() < self.ecommerce_chance:
            ecommerce_amt = random.uniform(2000, 10000)
            txn = self.log_transaction("DEBIT", "E-commerce Order", ecommerce_amt, date)
            if txn: events.append(txn)

        return events
