# agents/student.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent # Make sure this import path is correct

class Student(BaseAgent):
    """
    A specific agent profile for a Student.
    Represents a "thin-file" user with no formal income, high digital activity,
    and consumption-driven spending patterns.
    """
    def __init__(self):
        # 1. Define all profile attributes for the Student
        profile_attributes = {
            "archetype_name": "Student",
            "risk_profile": "High",
            "employment_status": "Not_Applicable",
            "employment_verification": "Not_Applicable",
            "income_type": "Allowance",
            "avg_monthly_income_range": "3000-10000",
            "income_pattern": "Irregular",
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": ["None"],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": ["None"],
            "utility_payment_status": "N/A", # Typically don't pay household bills
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.70, 0.85), 2),
            "ip_consistency_score": round(random.uniform(0.40, 0.60), 2),
            "sim_churn_rate": "Medium",
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Late_Night_Activity",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "Low",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        min_allowance, max_allowance = map(int, self.avg_monthly_income_range.split('-'))
        self.allowance_amount = random.uniform(min_allowance, max_allowance)
        # Students receive allowance on random days, simulating parental transfers
        self.allowance_days = sorted(random.sample(range(2, 28), random.randint(1, 2)))

        # High probability of daily spending
        self.daily_spend_chance = 0.75
        self.recharge_chance = 0.08 # 8% chance per day to need a recharge

        # Set a very low starting balance
        self.balance = random.uniform(100, 500)

    def _handle_income(self, date, events):
        """Simulates receiving allowance from family on random days of the month."""
        if date.day in self.allowance_days:
            txn = self.log_transaction("CREDIT", "Allowance/Family Support", self.allowance_amount, date)
            if txn: events.append(txn)

    def _handle_spending(self, date, events):
        """Simulates high-frequency, low-value spending on lifestyle and essentials."""
        # --- Prepaid Mobile Recharge ---
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([99, 149, 199, 239])
            txn = self.log_transaction("DEBIT", "Prepaid Mobile Recharge", recharge_amount, date)
            if txn: events.append(txn)

        # --- Daily UPI Spending ---
        if random.random() < self.daily_spend_chance:
            spend_category = random.choice([
                "Food_Delivery", "Cab_Service", "OTT_Subscription",
                "Groceries", "Gaming_Purchase", "Peer_Transfer"
            ])
            spend_amount = random.uniform(100, 600)
            txn = self.log_transaction("DEBIT", f"UPI Spend - {spend_category}", spend_amount, date)
            if txn: events.append(txn)


    def act(self, date: datetime):
        """
        Simulates the student's daily financial actions, which are primarily
        consumption-based with no fixed obligations.
        """
        events = []
        self._handle_income(date, events)
        self._handle_spending(date, events)
        return events