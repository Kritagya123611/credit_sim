# agents/senior.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class SeniorCitizen(BaseAgent):
    """
    An enhanced profile for a Retired Senior Citizen.
    Simulates a "passive" or "sleeping" account with low-frequency, highly predictable
    transactions and occasional large financial planning events.
    """
    def __init__(self):
        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Retired Senior Citizen",
            "risk_profile": "Very_Low",
            "employment_status": "Not_Applicable",
            "employment_verification": "Pensioner_ID_Verified",
            "income_type": "Pension, Rent",
            "avg_monthly_income_range": "10000-30000",
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "High",
            "has_investment_activity": True,
            "investment_types": ["FD", "Senior_Citizen_Schemes"],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": True,
            "insurance_types": ["Health"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Basic_Postpaid",
            "device_consistency_score": 0.99,
            "ip_consistency_score": 0.99,
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Branch"],
            "login_pattern": "Infrequent",
            "ecommerce_activity_level": "None",
            "ecommerce_avg_ticket_size": "N/A",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.pension_day = 1
        min_income, max_income = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_income, max_income)
        
        # Expenses and investments as a percentage of income
        self.insurance_percentage = 0.15 # Health insurance is a major expense
        self.utility_bill_percentage = 0.08
        
        # Parameters for structured spending
        self.weekly_grocery_day = 4 # Friday
        self.monthly_pharmacy_day = 10

        # Parameters for large, infrequent financial events
        self.large_event_month = random.randint(1, 12)
        self.has_done_large_event_this_year = False

        # Set a healthy starting balance
        self.balance = random.uniform(self.monthly_income * 2.0, self.monthly_income * 5.0)

    def _handle_monthly_events(self, date, events):
        """Handles fixed monthly income and debits."""
        # --- Pension Credit ---
        if date.day == self.pension_day:
            txn = self.log_transaction("CREDIT", "Pension/Rent Deposit", self.monthly_income, date)
            if txn: events.append(txn)
            # Reset annual event flag in January
            if date.month == 1:
                self.has_done_large_event_this_year = False

        # --- Health Insurance Premium ---
        if self.has_insurance_payments and date.day == 5:
            insurance_amt = self.monthly_income * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "Health Insurance Premium", insurance_amt, date)
            if txn: events.append(txn)

        # --- Monthly Pharmacy Visit ---
        if date.day == self.monthly_pharmacy_day:
            pharma_spend = random.uniform(500, 2000)
            txn = self.log_transaction("DEBIT", "Pharmacy/Medicines", pharma_spend, date)
            if txn: events.append(txn)

        # --- Utility Bill Payment ---
        if date.day == 20:
            bill = self.monthly_income * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", bill, date)
            if txn: events.append(txn)

    def _handle_weekly_events(self, date, events):
        """Handles structured weekly spending, like for groceries."""
        if date.weekday() == self.weekly_grocery_day:
            grocery_spend = random.uniform(800, 1500)
            txn = self.log_transaction("DEBIT", "Weekly Groceries/Essentials", grocery_spend, date)
            if txn: events.append(txn)

    def _handle_annual_events(self, date, events):
        """Simulates a major, infrequent financial planning event."""
        if self.has_investment_activity and date.month == self.large_event_month and date.day == 25 and not self.has_done_large_event_this_year:
            # Create a new Fixed Deposit from accumulated savings
            fd_amount = self.balance * random.uniform(0.3, 0.5) # Invest 30-50% of current balance
            if fd_amount > 10000: # Only if they have substantial savings
                txn = self.log_transaction("DEBIT", "New Fixed Deposit Creation", fd_amount, date)
                if txn: events.append(txn)
                self.has_done_large_event_this_year = True

    def act(self, date: datetime):
        """
        Simulates the agent's low-frequency and highly predictable financial life.
        """
        events = []
        self._handle_monthly_events(date, events)
        self._handle_weekly_events(date, events)
        self._handle_annual_events(date, events)
        return events