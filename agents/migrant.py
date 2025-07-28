# agents/migrant.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class MigrantWorker(BaseAgent):
    """
    An enhanced profile for a Migrant Worker.
    Simulates the critical "spike and drain" remittance cycle, where income is
    immediately sent home, a pattern that can be confused with money muling.
    """
    def __init__(self):
        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Migrant Worker",
            "risk_profile": "Very_High",
            "employment_status": "Informal_Labor",
            "employment_verification": "Not_Verified",
            "income_type": "Wages",
            "avg_monthly_income_range": "7000-15000",
            "income_pattern": "Weekly_or_Monthly",
            "savings_retention_rate": "Near_Zero",
            "has_investment_activity": False,
            "investment_types": ["None"],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": ["None"],
            "utility_payment_status": "N/A",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.40, 0.60), 2),
            "ip_consistency_score": round(random.uniform(0.30, 0.50), 2),
            "sim_churn_rate": "High",
            "primary_digital_channels": ["UPI", "IMPS"],
            "login_pattern": "Remittance_Cycle",
            "ecommerce_activity_level": "None",
            "ecommerce_avg_ticket_size": "N/A",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        # --- Geographic Signature ---
        self.home_state = random.choice(["Uttar Pradesh", "Bihar", "Odisha", "Rajasthan"])
        self.work_city = random.choice(["Mumbai", "Delhi", "Bengaluru", "Surat"])

        # --- Income and Remittance Logic ---
        min_inc, max_inc = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_inc, max_inc)
        self.pay_cycle = random.choice(["weekly", "monthly"])
        
        self.weekly_wage = self.monthly_income / 4
        self.monthly_pay_day = random.randint(1, 5)
        self.weekly_pay_day = 6 # Sunday
        
        self.remittance_percentage = random.uniform(0.6, 0.85) # Sends 60-85% of income home

        # --- Other Behaviors ---
        self.recharge_chance = 0.07 # 7% chance for a small recharge

        # Set a very low starting balance
        self.balance = random.uniform(100, 500)

    def _handle_payday_and_remittance(self, date, events):
        """
        Simulates the entire payday cycle: receive wage, remit home, withdraw remainder.
        This function is the core of the Migrant Worker's financial life.
        """
        is_payday = False
        wage_amount = 0

        if self.pay_cycle == "weekly" and date.weekday() == self.weekly_pay_day:
            is_payday = True
            wage_amount = self.weekly_wage
        elif self.pay_cycle == "monthly" and date.day == self.monthly_pay_day:
            is_payday = True
            wage_amount = self.monthly_income

        if is_payday:
            # --- 1. The "Spike": Wage Credit ---
            wage_txn = self.log_transaction("CREDIT", f"Weekly/Monthly Wage ({self.work_city})", wage_amount, date)
            if wage_txn:
                events.append(wage_txn)

                # --- 2. The "Drain" Part 1: Immediate Remittance ---
                # This happens right after getting paid
                remittance_amount = wage_amount * self.remittance_percentage
                remit_txn = self.log_transaction("DEBIT", f"Family Remittance to {self.home_state}", remittance_amount, date)
                if remit_txn:
                    events.append(remit_txn)

                # --- 3. The "Drain" Part 2: Cash Withdrawal ---
                # Withdraws most of what's left for personal expenses
                if self.balance > 50: # Leave a tiny amount in the account
                    cash_out_amount = self.balance * random.uniform(0.8, 1.0)
                    cash_txn = self.log_transaction("DEBIT", f"Cash Withdrawal ({self.work_city})", cash_out_amount, date)
                    if cash_txn:
                        events.append(cash_txn)

    def _handle_recharge(self, date, events):
        """Simulates small, infrequent mobile recharges."""
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([49, 99, 149]) # Focus on talk time packs
            txn = self.log_transaction("DEBIT", "Prepaid Mobile Recharge", recharge_amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime):
        """
        Simulates the Migrant Worker's financial life, centered on the
        payday remittance cycle.
        """
        events = []
        self._handle_payday_and_remittance(date, events)
        self._handle_recharge(date, events)
        return events