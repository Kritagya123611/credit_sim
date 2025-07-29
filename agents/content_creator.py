# agents/content_creator.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np

class ContentCreator(BaseAgent):
    """
    A multi-dimensional profile for a Content Creator.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Risk_Addict'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Content Creator / Influencer"

        # --- RISK SCORE CALCULATION ---
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "20000-100000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-Employed",
            "employment_verification": "ITR_Inconsistent",
            "income_type": "Sponsorships, Platform_Payouts",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Erratic_High_Variance",
            "savings_retention_rate": "Low",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "Occasionally_Late",
            "mobile_plan_type": "High-Value_Postpaid",
            "device_consistency_score": round(random.uniform(0.50, 0.70), 2),
            "ip_consistency_score": round(random.uniform(0.40, 0.60), 2),
            "sim_churn_rate": "High",
            "primary_digital_channels": ["All"],
            "login_pattern": "Geographically_Dynamic",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
        }
        
        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral parameters using the new dimensions
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_income = random.uniform(min_mod, max_mod)

        self.platform_payout_chance = 0.25 * personality_config['invest_chance_mod']
        self.sponsorship_chance = 0.03
        self.has_sponsorship_funds = False

        self.loan_emi_amount = self.avg_monthly_income * 0.20
        self.software_subscription = random.uniform(2000, 4000)
        self.utility_bill_day = random.randint(20, 28)
        self.late_payment_chance = 0.3
        
        self.spend_chance_mod = personality_config['spend_chance_mod']

        self.balance = random.uniform(5000, 20000)

    def _handle_income(self, date, events):
        """Simulates a volatile mix of small platform payouts and large sponsorships."""
        if random.random() < self.sponsorship_chance:
            sponsorship_amount = self.avg_monthly_income * random.uniform(2.0, 5.0)
            txn = self.log_transaction("CREDIT", "Brand Sponsorship", sponsorship_amount, date)
            if txn:
                events.append(txn)
                self.has_sponsorship_funds = True

        if random.random() < self.platform_payout_chance:
            payout_amount = self.avg_monthly_income * random.uniform(0.1, 0.4)
            source = random.choice(["YouTube AdSense", "Instagram Bonus", "Affiliate Payout"])
            txn = self.log_transaction("CREDIT", source, payout_amount, date)
            if txn: events.append(txn)

    def _handle_fixed_and_professional_expenses(self, date, events):
        """Handles recurring debits like loans, software, and late utility bills."""
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Equipment Loan EMI", self.loan_emi_amount, date)
            if txn: events.append(txn)
            
        if date.day == 5:
            txn = self.log_transaction("DEBIT", "SaaS Subscription (Adobe/Canva)", self.software_subscription, date)
            if txn: events.append(txn)

        payment_day = self.utility_bill_day + (random.randint(1, 5) if random.random() < self.late_payment_chance else 0)
        if date.day == payment_day:
            utility_bill_amount = random.uniform(2000, 5000)
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", utility_bill_amount, date)
            if txn: events.append(txn)
            
    def _handle_dynamic_spending(self, date, events):
        """Simulates large, dynamic spending often triggered by a sponsorship."""
        spend_chance_multiplier = 5.0 if self.has_sponsorship_funds else 1.0

        if self.has_sponsorship_funds and self.has_investment_activity and random.random() < 0.6:
            spend_category = random.choice(self.investment_types + ["New Camera/Laptop Gear", "Content Trip Booking"])
            amount = self.balance * random.uniform(0.3, 0.7)
            txn = self.log_transaction("DEBIT", spend_category, amount, date)
            if txn:
                events.append(txn)
                self.has_sponsorship_funds = False

        if random.random() < (0.2 * spend_chance_multiplier * self.spend_chance_mod):
             ecommerce_amt = random.uniform(2000, 15000)
             txn = self.log_transaction("DEBIT", "E-commerce (Fashion/Props)", ecommerce_amt, date)
             if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the creator's "high entropy" financial life, where large
        payouts trigger large expenses.
        """
        events = []
        self._handle_income(date, events)
        self._handle_fixed_and_professional_expenses(date, events)
        self._handle_dynamic_spending(date, events)
        
        return events