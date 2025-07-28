# agents/migrant.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class MigrantWorker(BaseAgent):
    def __init__(self):
        profile_attributes = {
            "archetype_name": "Migrant Worker",
            "risk_profile": "Very_High",
            "employment_status": "Informal_Labor",
            "employment_verification": "Not_Verified",
            "income_type": "Wages",
            "avg_monthly_income_range": "7000-15000",
            "income_pattern": "Weekly_or_Monthly",

            # Financial Habits
            "savings_retention_rate": "Near_Zero",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": [],

            # Utility Payments
            "utility_payment_status": "N/A",
            "mobile_plan_type": "Prepaid",

            # Digital Footprint
            "device_consistency_score": 0.5,
            "ip_consistency_score": 0.4,
            "sim_churn_rate": "High",
            "primary_digital_channels": ["UPI", "IMPS"],
            "login_pattern": "Remittance_Cycle",

            # E-commerce
            "ecommerce_activity_level": "None",
            "ecommerce_avg_ticket_size": "None",
        }

        super().__init__(**profile_attributes)

        # Income logic: weekly or monthly
        min_inc, max_inc = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_inc, max_inc)
        self.income_type = random.choice(["weekly", "monthly"])
        self.weekly_wage = self.monthly_income / 4 if self.income_type == "weekly" else 0
        self.monthly_pay_day = random.randint(1, 5)

        # Remittance behavior (send money home)
        self.remit_chance = 0.6
        self.remit_pct = 0.4

        # Set low balance
        self.balance = random.uniform(200, 1000)

    def act(self, date: datetime):
        events = []

        # --- Wage Credit ---
        if self.income_type == "weekly" and date.weekday() == 6:
            txn = self.log_transaction("CREDIT", "Weekly Wage", self.weekly_wage, date)
            if txn: events.append(txn)

        elif self.income_type == "monthly" and date.day == self.monthly_pay_day:
            txn = self.log_transaction("CREDIT", "Monthly Wage", self.monthly_income, date)
            if txn: events.append(txn)

        # --- Remittance ---
        if random.random() < self.remit_chance:
            rem_amt = self.balance * self.remit_pct
            txn = self.log_transaction("DEBIT", "Family Remittance", rem_amt, date)
            if txn: events.append(txn)

        # --- Prepaid Recharge (occasional) ---
        if random.random() < 0.2:
            recharge = random.uniform(100, 300)
            txn = self.log_transaction("DEBIT", "Mobile Recharge", recharge, date)
            if txn: events.append(txn)

        return events
