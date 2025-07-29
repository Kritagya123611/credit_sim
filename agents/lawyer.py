# agents/lawyer.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES

class Lawyer(BaseAgent):
    """
    A multi-dimensional profile for a Lawyer or Consultant.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Middle', financial_personality='Rational_Investor'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])

        base_income_range = "50000-200000"
        min_monthly, max_monthly = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_monthly * income_multiplier)}-{int(max_monthly * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Lawyer / Consultant",
            "risk_profile": "Medium",
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-Employed_Professional",
            "employment_verification": "Professional_License_Verified",
            "income_type": "Professional_Fees",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Lumpy",
            "savings_retention_rate": "Medium",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life", "Prof_Indemnity"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "High-Value_Postpaid",
            "device_consistency_score": round(random.uniform(0.88, 0.95), 2),
            "ip_consistency_score": round(random.uniform(0.82, 0.92), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Cards"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "High",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        avg_monthly = (min_mod + max_mod) / 2

        self.payout_months = sorted(random.sample(range(1, 13), k=random.randint(2, 4)))
        self.lump_sum_payment = avg_monthly * (12 / len(self.payout_months))
        self.has_large_cash_reserve = False

        self.junior_retainer_fee = avg_monthly * random.uniform(0.4, 0.6)
        self.loan_emi_amount = avg_monthly * 0.30
        self.prof_indemnity_premium = avg_monthly * 0.5

        self.spend_chance_mod = personality_config['spend_chance_mod']
        self.invest_chance_mod = personality_config['invest_chance_mod']
        
        self.balance = random.uniform(avg_monthly * 1.5, avg_monthly * 3.0)

    def _handle_lumpy_income(self, date, events):
        """Simulates receiving large, infrequent payments."""
        if date.month in self.payout_months and date.day == 15:
            payout = self.lump_sum_payment * random.uniform(0.8, 1.2)
            txn = self.log_transaction("CREDIT", "Client/Project Fee Received", payout, date)
            if txn:
                events.append(txn)
                self.has_large_cash_reserve = True

    def _handle_recurring_debits(self, date, events):
        """Handles regular monthly and annual professional/personal expenses."""
        if date.day == 5:
            txn = self.log_transaction("DEBIT", "Junior/Retainer Fee", self.junior_retainer_fee, date)
            if txn: events.append(txn)

        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", self.loan_emi_amount, date)
            if txn: events.append(txn)

        if self.has_insurance_payments and date.month == 7 and date.day == 20:
            txn = self.log_transaction("DEBIT", "Professional Indemnity Insurance", self.prof_indemnity_premium, date)
            if txn: events.append(txn)

    def _handle_spending_and_investment(self, date, events):
        """Simulates spending and large investments, often after a payout."""
        if self.has_large_cash_reserve:
            # --- THIS IS THE FIX ---
            # Only attempt to invest if the agent's profile allows for it.
            if self.has_investment_activity and random.random() < (0.5 * self.invest_chance_mod):
                investment_amount = self.balance * random.uniform(0.3, 0.6)
                investment_type = random.choice(self.investment_types)
                txn = self.log_transaction("DEBIT", f"Lump-Sum Investment - {investment_type}", investment_amount, date)
                if txn:
                    events.append(txn)
                    self.has_large_cash_reserve = False

        if random.random() < (0.4 * self.spend_chance_mod):
            spend_category = random.choice(["Fine Dining", "Travel Booking", "Professional Books"])
            amount = random.uniform(1000, 8000)
            txn = self.log_transaction("DEBIT", f"Card Spend - {spend_category}", amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the lawyer's "feast or famine" financial cycle.
        """
        events = []
        self._handle_lumpy_income(date, events)
        self._handle_recurring_debits(date, events)
        self._handle_spending_and_investment(date, events)
        return events