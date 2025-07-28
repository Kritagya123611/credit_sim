# agents/content_creator.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class ContentCreator(BaseAgent):
    """
    An enhanced profile for an Influencer / Content Creator.
    Simulates a "high entropy" financial life with volatile, project-based income
    that directly triggers high-value professional and lifestyle spending.
    """
    def __init__(self):
        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Influencer / Content Creator",
            "risk_profile": "High",
            "employment_status": "Self-Employed",
            "employment_verification": "ITR_Inconsistent",
            "income_type": "Sponsorships, Platform_Payouts",
            "avg_monthly_income_range": "20000-100000",
            "income_pattern": "Erratic_High_Variance",
            "savings_retention_rate": "Low",
            "has_investment_activity": True,
            "investment_types": ["Crypto", "Stocks"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": ["None"],
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

        # 3. Define behavioral and simulation parameters
        min_inc, max_inc = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_income = random.uniform(min_inc, max_inc)

        # Income simulation
        self.platform_payout_chance = 0.25 # 25% chance of a small payout
        self.sponsorship_chance = 0.03 # 3% chance of a big sponsorship deal
        self.has_sponsorship_funds = False # Flag to trigger big spending

        # Expense parameters
        self.loan_emi_amount = self.avg_monthly_income * 0.20
        self.software_subscription = random.uniform(2000, 4000)
        self.utility_bill_day = random.randint(20, 28) # Unpredictable payment day
        self.late_payment_chance = 0.3 # 30% chance to pay late

        # Set a volatile starting balance
        self.balance = random.uniform(5000, 20000)

    def _handle_income(self, date, events):
        """Simulates a volatile mix of small platform payouts and large sponsorships."""
        # --- Large, rare sponsorship deal ---
        if random.random() < self.sponsorship_chance:
            sponsorship_amount = self.avg_monthly_income * random.uniform(2.0, 5.0)
            txn = self.log_transaction("CREDIT", "Brand Sponsorship", sponsorship_amount, date)
            if txn:
                events.append(txn)
                self.has_sponsorship_funds = True # Flag that a big payout was received

        # --- Smaller, more frequent platform payouts ---
        if random.random() < self.platform_payout_chance:
            payout_amount = self.avg_monthly_income * random.uniform(0.1, 0.4)
            source = random.choice(["YouTube AdSense", "Instagram Bonus", "Affiliate Payout"])
            txn = self.log_transaction("CREDIT", source, payout_amount, date)
            if txn: events.append(txn)

    def _handle_fixed_and_professional_expenses(self, date, events):
        """Handles recurring debits like loans, software, and late utility bills."""
        # --- Loan EMI for equipment/vehicle ---
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Equipment Loan EMI", self.loan_emi_amount, date)
            if txn: events.append(txn)
            
        # --- Software Subscriptions ---
        if date.day == 5:
            txn = self.log_transaction("DEBIT", "SaaS Subscription (Adobe/Canva)", self.software_subscription, date)
            if txn: events.append(txn)

        # --- Utility Bill (paid on a variable day, sometimes late) ---
        payment_day = self.utility_bill_day + (random.randint(1, 5) if random.random() < self.late_payment_chance else 0)
        if date.day == payment_day:
            utility_bill_amount = random.uniform(2000, 5000)
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", utility_bill_amount, date)
            if txn: events.append(txn)
            
    def _handle_dynamic_spending(self, date, events):
        """Simulates large, dynamic spending often triggered by a sponsorship."""
        spend_chance_multiplier = 5.0 if self.has_sponsorship_funds else 1.0

        # --- High-value equipment purchase or investment after a big payout ---
        if self.has_sponsorship_funds and random.random() < 0.6: # 60% chance to spend big after sponsorship
            spend_category = random.choice(["Crypto/Stock Investment", "New Camera/Laptop Gear", "Content Trip Booking"])
            amount = self.balance * random.uniform(0.3, 0.7) # Spend 30-70% of current balance
            txn = self.log_transaction("DEBIT", spend_category, amount, date)
            if txn:
                events.append(txn)
                self.has_sponsorship_funds = False # Reset flag after the big spend

        # --- Regular high e-commerce spending ---
        if random.random() < (0.2 * spend_chance_multiplier): # Much higher chance after payout
             ecommerce_amt = random.uniform(2000, 15000)
             txn = self.log_transaction("DEBIT", "E-commerce (Fashion/Props)", ecommerce_amt, date)
             if txn: events.append(txn)

    def act(self, date: datetime):
        """
        Simulates the creator's "high entropy" financial life, where large
        payouts trigger large expenses.
        """
        events = []
        self._handle_income(date, events)
        self._handle_fixed_and_professional_expenses(date, events)
        self._handle_dynamic_spending(date, events)
        return events