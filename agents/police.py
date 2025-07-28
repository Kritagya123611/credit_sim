# agents/police.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class PoliceOfficer(BaseAgent):
    """
    An enhanced profile for Police / Security Staff.
    Focuses on discipline, routine family remittances, and non-standard work hours.
    """
    def __init__(self):
        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Police / Security Staff",
            "risk_profile": "Very_Low",
            "employment_status": "Salaried",
            "employment_verification": "GOVT_Verified",
            "income_type": "Uniformed_Services_Salary",
            "avg_monthly_income_range": "30000-50000",
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "Medium",
            "has_investment_activity": True,
            "investment_types": ["LIC", "FD"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["LIC", "Government_Schemes"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid",
            "device_consistency_score": round(random.uniform(0.92, 0.98), 2),
            "ip_consistency_score": round(random.uniform(0.75, 0.85), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Mobile_Banking", "UPI"],
            "login_pattern": "Shift_Work_Night_Activity",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Medium",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.salary_day = 1
        min_sal, max_sal = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal, max_sal)

        # Behavioral percentages
        self.remittance_percentage = 0.30 # Sends 30% of salary home
        self.emi_percentage = 0.20
        self.investment_percentage = 0.10
        self.insurance_percentage = 0.08
        
        # Spending probabilities
        self.daily_spend_chance = 0.60 # Higher chance of small daily spends
        
        # Set a realistic starting balance
        self.balance = random.uniform(self.salary_amount * 0.15, self.salary_amount * 0.4)

    def _handle_fixed_monthly_events(self, date, events):
        """Handles salary, remittances, and other fixed-date debits."""
        # --- Salary Credit ---
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Salary Deposit", self.salary_amount, date)
            if txn: events.append(txn)

            # --- Family Remittance (immediately after salary) ---
            remittance_amount = self.salary_amount * self.remittance_percentage
            txn = self.log_transaction("DEBIT", "Family Remittance (P2P)", remittance_amount, date)
            if txn: events.append(txn)

        # --- Loan EMI ---
        if self.has_loan_emi and date.day == 10:
            emi_amount = self.salary_amount * self.emi_percentage
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", emi_amount, date)
            if txn: events.append(txn)

        # --- Insurance Payment ---
        if self.has_insurance_payments and date.day == 15:
            insurance_total = self.salary_amount * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "Insurance Premium (LIC)", insurance_total, date)
            if txn: events.append(txn)

        # --- Investment ---
        if self.has_investment_activity and date.day == 20:
            invest_amt = self.salary_amount * self.investment_percentage
            txn = self.log_transaction("DEBIT", "Investment - FD", invest_amt, date)
            if txn: events.append(txn)

    def _handle_daily_spending(self, date, events):
        """
        Simulates daily spending that can occur 24/7 due to shift work,
        including night-time activity.
        """
        if random.random() < self.daily_spend_chance:
            # Shift work means expenses can happen any day, any time.
            # We can add flavor to the description to reflect this.
            is_night_shift = random.random() < 0.4 # 40% chance the spend is during a night shift
            
            if is_night_shift:
                description = "Night Duty - Food/Tea"
                amount = random.uniform(100, 300)
            else:
                description = "Daily Expense - Groceries/Misc"
                amount = random.uniform(200, 800)
                
            txn = self.log_transaction("DEBIT", description, amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime):
        """
        Simulates the agent's disciplined financial life, including the key
        remittance behavior and 24/7 spending patterns.
        """
        events = []
        self._handle_fixed_monthly_events(date, events)
        self._handle_daily_spending(date, events)
        return events