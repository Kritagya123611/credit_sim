# agents/delivery_agent.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class DeliveryAgent(BaseAgent):
    def __init__(self):
        profile_attributes = {
            "archetype_name": "Delivery Agent / Rider",
            "risk_profile": "Medium",
            "employment_status": "Gig_Work_Contractor",
            "employment_verification": "Not_Verified",
            "income_type": "Platform_Payout",
            "avg_monthly_income_range": "15000-25000",
            "income_pattern": "Daily",

            # Financial Habits
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": True,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],

            # Utility Payments
            "utility_payment_status": "Mostly_On_Time",
            "mobile_plan_type": "Prepaid",

            # Digital Footprint
            "device_consistency_score": 0.9,
            "ip_consistency_score": 0.4,
            "sim_churn_rate": "Medium",
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Geographically_Dynamic",

            # E-commerce
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Low",
        }

        super().__init__(**profile_attributes)

        # Convert monthly range to a daily payout
        min_income, max_income = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_income, max_income)
        self.daily_payout = self.monthly_income / 26  # Assuming ~26 working days

        self.emi_percentage = 0.20
        self.utility_bill_percentage = 0.10

        self.daily_spend_chance = 0.6
        self.ecommerce_chance = 0.1

        self.balance = random.uniform(500, 3000)

    def act(self, date: datetime):
        events = []

        # Daily Payout
        txn = self.log_transaction("CREDIT", "Platform Daily Payout", self.daily_payout, date)
        if txn: events.append(txn)

        # Loan EMI on 10th
        if self.has_loan_emi and date.day == 10:
            emi = self.monthly_income * self.emi_percentage
            txn = self.log_transaction("DEBIT", "Loan EMI", emi, date)
            if txn: events.append(txn)

        # Utility Bill on 20th
        if date.day == 20:
            bill = self.monthly_income * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Prepaid/Data Recharge", bill, date)
            if txn: events.append(txn)

        # Random discretionary UPI/wallet spend
        if random.random() < self.daily_spend_chance:
            spend = random.uniform(100, 600)
            txn = self.log_transaction("DEBIT", "UPI/Tea/Fuel", spend, date)
            if txn: events.append(txn)

        # Occasional e-commerce
        if random.random() < self.ecommerce_chance:
            ecommerce_amt = random.uniform(500, 1500)
            txn = self.log_transaction("DEBIT", "Online Order", ecommerce_amt, date)
            if txn: events.append(txn)

        return events
