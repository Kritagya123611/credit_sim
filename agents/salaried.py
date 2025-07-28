# agents/salaried_professional.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent # Make sure this import path is correct

class SalariedProfessional(BaseAgent):
    """
    An enhanced profile for a Salaried Professional, featuring more detailed
    and realistic daily financial actions.
    """
    def __init__(self):
        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Salaried Professional",
            "risk_profile": "Low",
            "employment_status": "Salaried",
            "employment_verification": "EPFO_Verified",
            "income_type": "Salary",
            "avg_monthly_income_range": "40000-80000",
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "High",
            "has_investment_activity": True,
            "investment_types": ["SIP", "Mutual_Funds"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid",
            "device_consistency_score": round(random.uniform(0.90, 0.98), 2),
            "ip_consistency_score": round(random.uniform(0.85, 0.95), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Mobile_Banking"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Medium",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.salary_day = random.randint(1, 5)
        min_sal, max_sal = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal, max_sal)
        
        # Expenses as percentages of salary
        self.emi_percentage = 0.25
        self.investment_percentage = 0.15
        self.insurance_percentage = 0.05
        self.utility_bill_percentage = 0.05
        
        # Probabilities for daily actions
        self.ecommerce_spend_chance = 0.15
        self.weekday_spend_chance = 0.50
        self.weekend_spend_chance = 0.70
        
        # Annual bonus parameters
        self.annual_bonus_month = 3 # March
        self.has_received_bonus_this_year = False

        # Set a realistic starting balance
        self.balance = random.uniform(self.salary_amount * 0.2, self.salary_amount * 0.5)

    def _handle_monthly_credits(self, date, events):
        """Handles salary and annual bonus credits."""
        # --- Salary Credit ---
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Salary Deposit", self.salary_amount, date)
            if txn: events.append(txn)
            # Reset bonus flag for the new year
            if date.month == 1:
                self.has_received_bonus_this_year = False

        # --- Annual Bonus Credit ---
        if date.month == self.annual_bonus_month and date.day == self.salary_day and not self.has_received_bonus_this_year:
            bonus_amount = self.salary_amount * random.uniform(1.5, 3.0)
            txn = self.log_transaction("CREDIT", "Annual Bonus", bonus_amount, date)
            if txn: events.append(txn)
            self.has_received_bonus_this_year = True

    def _handle_recurring_debits(self, date, events):
        """Handles fixed monthly payments like EMIs, insurance, and bills."""
        if self.has_loan_emi and date.day == 10:
            emi_amount = self.salary_amount * self.emi_percentage
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", emi_amount, date)
            if txn: events.append(txn)

        if self.has_insurance_payments and date.day == 15:
            insurance_total = self.salary_amount * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "Insurance Premium", insurance_total, date)
            if txn: events.append(txn)

        if self.has_investment_activity and date.day == 20:
            invest_amt = self.salary_amount * self.investment_percentage
            txn = self.log_transaction("DEBIT", "SIP Investment", invest_amt, date)
            if txn: events.append(txn)

        if date.day == 25:
            bill = self.salary_amount * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", bill, date)
            if txn: events.append(txn)

    def _handle_daily_spending(self, date, events):
        """Handles probabilistic daily spending which varies by day type."""
        # --- E-commerce and Large Purchases (more likely after bonus) ---
        ecommerce_chance = self.ecommerce_spend_chance * (2.5 if self.has_received_bonus_this_year else 1)
        if random.random() < ecommerce_chance:
            ecommerce_amt = random.uniform(1000, 5000)
            txn = self.log_transaction("DEBIT", "E-commerce Purchase", ecommerce_amt, date)
            if txn: events.append(txn)

        # --- Weekday vs. Weekend Spending ---
        is_weekend = date.weekday() >= 5
        if is_weekend:
            if random.random() < self.weekend_spend_chance:
                spend = random.uniform(500, 2500)
                txn = self.log_transaction("DEBIT", "Weekend Entertainment/Dining", spend, date)
                if txn: events.append(txn)
        else: # It's a weekday
            if random.random() < self.weekday_spend_chance:
                spend_type = random.choice(["Transport", "Groceries", "Lunch"])
                spend_amount = random.uniform(150, 800)
                txn = self.log_transaction("DEBIT", f"UPI Spend - {spend_type}", spend_amount, date)
                if txn: events.append(txn)

    def act(self, date: datetime):
        """
        Simulates the agent's daily financial actions by calling helper methods.
        This makes the logic clean and easy to follow.
        """
        events = []
        self._handle_monthly_credits(date, events)
        self._handle_recurring_debits(date, events)
        self._handle_daily_spending(date, events)
        return events