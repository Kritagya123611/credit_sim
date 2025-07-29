# agents/gig_worker.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES

class GigWorker(BaseAgent):
    """
    A multi-dimensional profile for a Gig Worker or Freelancer.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Over_Spender'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])

        base_income_range = "8000-35000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Gig Worker / Freelancer",
            "risk_profile": "Medium",
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-Employed",
            "employment_verification": "Not_Verified",
            "income_type": "Gig_Work",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Irregular",
            "savings_retention_rate": "Low",
            "has_investment_activity": len(personality_config['investment_types']) > 0 and financial_personality != 'Over_Spender',
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] * 0.5 else False, # Lower loan chance
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],
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
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        avg_monthly_income = random.uniform(min_mod, max_mod)

        self.daily_income_chance = 0.35
        self.avg_gig_payment = avg_monthly_income / 8 # Assume ~8 paying gigs a month
        self.income_sources = ["Client Project", "Platform Payout", "Freelance Task"]
        
        # Modify expenses and discipline based on personality
        self.rent_day = random.randint(5, 10)
        self.rent_amount = avg_monthly_income * random.uniform(0.3, 0.5)
        self.bill_payment_late_chance = 0.20 / personality_config['spend_chance_mod'] # Savers are less likely to be late
        
        self.daily_spend_chance = 0.80 * personality_config['spend_chance_mod']
        self.prepaid_recharge_chance = 0.10 * personality_config['spend_chance_mod']

        self.balance = random.uniform(avg_monthly_income * 0.05, avg_monthly_income * 0.2)

    def _handle_income(self, date, events):
        """Simulates the probabilistic daily income of a gig worker."""
        if random.random() < self.daily_income_chance:
            income_amount = self.avg_gig_payment * random.uniform(0.7, 1.3)
            income_source = random.choice(self.income_sources)
            txn = self.log_transaction("CREDIT", income_source, income_amount, date)
            if txn: events.append(txn)

    def _handle_bills(self, date, events):
        """Simulates manual and sometimes late bill payments."""
        rent_payment_day = self.rent_day + (random.randint(0, 3) if random.random() < self.bill_payment_late_chance else 0)
        if date.day == rent_payment_day:
            txn = self.log_transaction("DEBIT", "Rent Payment", self.rent_amount, date)
            if txn: events.append(txn)

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

    def act(self, date: datetime, **context):
        """
        Simulates the agent's daily financial actions, reflecting their volatile lifestyle.
        """
        events = []
        self._handle_income(date, events)
        self._handle_bills(date, events)
        self._handle_daily_spending(date, events)
        return events