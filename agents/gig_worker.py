# agents/gig_worker.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent # Make sure this import path is correct

class GigWorker(BaseAgent):
    """
    A specific agent profile for a Gig Worker or Freelancer.
    Characterized by irregular income, high transaction velocity, and lower financial stability.
    """
    def __init__(self):
        # 1. Define all profile attributes for the Gig Worker
        profile_attributes = {
            "archetype_name": "Gig Worker / Freelancer",
            "risk_profile": "Medium",
            "employment_status": "Self-Employed",
            "employment_verification": "Not_Verified",
            "income_type": "Gig_Work",
            "avg_monthly_income_range": "8000-35000",
            "income_pattern": "Irregular",
            "savings_retention_rate": "Low",
            "has_investment_activity": False,
            "investment_types": ["None"],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": ["None"],
            "utility_payment_status": "Mostly_On_Time",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.65, 0.80), 2),
            "ip_consistency_score": round(random.uniform(0.50, 0.70), 2),
            "sim_churn_rate": "Medium",
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Irregular",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Low",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.daily_income_chance = 0.35  # 35% chance to earn something on any given day
        self.avg_gig_payment = random.uniform(500, 2500)
        self.income_sources = ["Client Project", "Platform Payout", "Freelance Task"]
        
        # Bill payment parameters
        self.rent_day = random.randint(5, 10)
        self.rent_amount = random.uniform(4000, 8000)
        self.bill_payment_late_chance = 0.20 # 20% chance to pay a bill late
        
        # Spending probabilities
        self.daily_spend_chance = 0.80 # High chance of small daily spending
        self.prepaid_recharge_chance = 0.10 # 10% chance per day to recharge

        # Set a low starting balance, typical for this profile
        self.balance = random.uniform(500, 3000)

    def _handle_income(self, date, events):
        """Simulates the probabilistic daily income of a gig worker."""
        if random.random() < self.daily_income_chance:
            # Income amount varies around the average
            income_amount = self.avg_gig_payment * random.uniform(0.7, 1.3)
            income_source = random.choice(self.income_sources)
            txn = self.log_transaction("CREDIT", income_source, income_amount, date)
            if txn: events.append(txn)

    def _handle_bills(self, date, events):
        """Simulates manual and sometimes late bill payments."""
        # --- Rent Payment ---
        # They might pay rent a few days late if the payment day arrives
        rent_payment_day = self.rent_day + (random.randint(0, 3) if random.random() < self.bill_payment_late_chance else 0)
        if date.day == rent_payment_day:
            txn = self.log_transaction("DEBIT", "Rent Payment", self.rent_amount, date)
            if txn: events.append(txn)

        # --- Prepaid Mobile Recharge ---
        if random.random() < self.prepaid_recharge_chance:
            recharge_amount = random.choice([149, 199, 239, 299])
            txn = self.log_transaction("DEBIT", "Prepaid Mobile Recharge", recharge_amount, date)
            if txn: events.append(txn)

    def _handle_daily_spending(self, date, events):
        """Simulates high-frequency, low-value daily spending."""
        if random.random() < self.daily_spend_chance:
            spend_category = random.choice(["Food", "Transport", "Groceries", "Tea/Snacks"])
            spend_amount = random.uniform(50, 400)
            txn = self.log_transaction("DEBIT", f"UPI Spend - {spend_category}", spend_amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime):
        """
        Simulates the agent's daily financial actions, reflecting their volatile lifestyle.
        """
        events = []
        self._handle_income(date, events)
        self._handle_bills(date, events)
        self._handle_daily_spending(date, events)
        return events