# agents/daily_wage_laborer.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent # Make sure this import path is correct

class DailyWageLaborer(BaseAgent):
    """
    A specific agent profile for a Daily Wage Laborer.
    Represents a financially vulnerable user with informal, unstable income
    and a "pass-through" usage pattern for their bank account.
    """
    def __init__(self):
        # 1. Define all profile attributes for the Daily Wage Laborer
        profile_attributes = {
            "archetype_name": "Daily Wage Laborer",
            "risk_profile": "Very_High",
            "employment_status": "Informal_Labor",
            "employment_verification": "Not_Verified",
            "income_type": "Cash_Deposit, Wages",
            "avg_monthly_income_range": "7000-15000",
            "income_pattern": "Daily",
            "savings_retention_rate": "Near_Zero",
            "has_investment_activity": False,
            "investment_types": ["None"],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": ["None"],
            "utility_payment_status": "N/A",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.30, 0.55), 2),
            "ip_consistency_score": round(random.uniform(0.20, 0.40), 2),
            "sim_churn_rate": "High",
            "primary_digital_channels": ["Cash", "UPI"],
            "login_pattern": "Irregular",
            "ecommerce_activity_level": "None",
            "ecommerce_avg_ticket_size": "N/A",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.daily_work_chance = 0.80  # 80% chance of finding work on a given day
        self.daily_wage_amount = random.uniform(300, 600)
        
        # They send most of their money home immediately
        self.remittance_percentage = random.uniform(0.6, 0.8) # 60-80%
        
        # Sachet recharges
        self.recharge_chance = 0.05 # 5% chance per day

        # Start with a near-zero balance
        self.balance = random.uniform(50, 200)

    def _handle_work_and_remittance(self, date, events):
        """Simulates getting paid for a day's work and immediately sending money home."""
        if random.random() < self.daily_work_chance:
            # --- The "Spike": Cash Deposit ---
            wage_txn = self.log_transaction("CREDIT", "Cash Wage Deposit", self.daily_wage_amount, date)
            if wage_txn:
                events.append(wage_txn)

                # --- The "Drain": Immediate Remittance and Withdrawal ---
                # This happens right after getting paid.
                
                # 1. Send money to family
                remittance_amount = self.daily_wage_amount * self.remittance_percentage
                remit_txn = self.log_transaction("DEBIT", "Family Remittance (P2P)", remittance_amount, date)
                if remit_txn:
                    events.append(remit_txn)
                
                # 2. Withdraw the rest for personal expenses
                # Check current balance to see what's left to withdraw
                if self.balance > 0:
                    cash_out_amount = self.balance * random.uniform(0.8, 1.0) # Withdraw 80-100% of what's left
                    if cash_out_amount > 10: # Avoid tiny withdrawals
                        cash_txn = self.log_transaction("DEBIT", "Cash Withdrawal for Expenses", cash_out_amount, date)
                        if cash_txn:
                            events.append(cash_txn)

    def _handle_recharge(self, date, events):
        """Simulates small, infrequent mobile recharges."""
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([10, 20, 49, 99]) # Sachet packs
            txn = self.log_transaction("DEBIT", "Sachet Mobile Recharge", recharge_amount, date)
            if txn: events.append(txn)


    def act(self, date: datetime):
        """
        Simulates the daily "spike and drain" financial cycle of a daily wage laborer.
        """
        events = []
        self._handle_work_and_remittance(date, events)
        self._handle_recharge(date, events)
        return events