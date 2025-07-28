# agents/homemaker.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class Homemaker(BaseAgent):
    """
    An enhanced profile for a Homemaker.
    Simulates their role as a household financial manager with structured, need-based
    spending and highlights the data ambiguity from a shared digital footprint.
    """
    def __init__(self):
        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Homemaker",
            "risk_profile": "Very_High",
            "employment_status": "Not_Applicable",
            "employment_verification": "Not_Applicable",
            "income_type": "Family_Support",
            "avg_monthly_income_range": "10000-30000",
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": ["None"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Child_Education_Plan"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.60, 0.80), 2),
            "ip_consistency_score": 0.98,
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["UPI", "Mobile_Banking"],
            "login_pattern": "Infrequent",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Medium",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        min_inc, max_inc = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_allowance = random.uniform(min_inc, max_inc)

        # Household expense parameters
        self.loan_emi_amount = self.monthly_allowance * 0.30
        self.insurance_premium = random.uniform(1500, 3000)
        self.utility_bill_amount = random.uniform(2000, 4000)
        self.weekly_grocery_day = 5 # Saturday
        self.school_fee_months = [1, 4, 7, 10] # Fees due at start of quarter

        # Low chance for other ad-hoc spending
        self.occasional_spend_chance = 0.08

        # Placeholder for main script to link devices
        self.shared_device_id = None 

        # Set a starting balance for cash flow
        self.balance = random.uniform(1000, 5000)

    def _handle_monthly_income_and_fixed_costs(self, date, events):
        """Handles the monthly allowance and fixed, recurring household payments."""
        # --- Monthly Family Support Transfer ---
        if date.day == 1:
            txn = self.log_transaction("CREDIT", "Family Support Transfer", self.monthly_allowance, date)
            if txn: events.append(txn)

        # --- Loan EMI (as co-payment) ---
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Home/Car Loan EMI (Co-payment)", self.loan_emi_amount, date)
            if txn: events.append(txn)
            
        # --- Utility Bill Payment ---
        if date.day == 15:
            txn = self.log_transaction("DEBIT", "Utility Bill Payment (Gas/DTH)", self.utility_bill_amount, date)
            if txn: events.append(txn)

        # --- Child's Insurance Plan ---
        if self.has_insurance_payments and date.day == 20:
            txn = self.log_transaction("DEBIT", "Child Education Plan Premium", self.insurance_premium, date)
            if txn: events.append(txn)

    def _handle_household_spending(self, date, events):
        """Simulates structured and occasional spending for the household."""
        # --- Weekly Grocery Shopping ---
        if date.weekday() == self.weekly_grocery_day:
            grocery_amount = random.uniform(1000, 2500)
            txn = self.log_transaction("DEBIT", "UPI - Weekly Groceries", grocery_amount, date)
            if txn: events.append(txn)

        # --- Quarterly School Fees ---
        if date.month in self.school_fee_months and date.day == 5:
            fee_amount = random.uniform(5000, 15000)
            txn = self.log_transaction("DEBIT", "Netbanking - School Fees", fee_amount, date)
            if txn: events.append(txn)

        # --- Occasional E-commerce ---
        if random.random() < self.occasional_spend_chance:
            amount = random.uniform(800, 3000)
            category = random.choice(["Kids Clothing", "Home Goods", "Online Pharmacy"])
            txn = self.log_transaction("DEBIT", f"E-commerce - {category}", amount, date)
            if txn: events.append(txn)
    
    def act(self, date: datetime):
        """
        Simulates the Homemaker's role as a household financial manager.
        """
        events = []
        self._handle_monthly_income_and_fixed_costs(date, events)
        self._handle_household_spending(date, events)
        return events