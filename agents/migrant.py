# agents/migrant.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES

class MigrantWorker(BaseAgent):
    """
    A multi-dimensional profile for a Migrant Worker.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower', financial_personality='Saver'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])

        base_income_range = "7000-15000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"
        
        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Migrant Worker",
            "risk_profile": "Very_High",
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Informal_Labor",
            "employment_verification": "Not_Verified",
            "income_type": "Wages",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Weekly_or_Monthly",
            "savings_retention_rate": "Near_Zero",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": [],
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
        self.home_state = random.choice(["Uttar Pradesh", "Bihar", "Odisha", "Rajasthan"])
        self.work_city = random.choice(["Mumbai", "Delhi", "Bengaluru", "Surat"])

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_mod, max_mod)
        
        # Higher class might have a more stable monthly pay cycle
        self.pay_cycle = "monthly" if economic_class == 'Lower_Middle' else "weekly"
        
        self.weekly_wage = self.monthly_income / 4
        self.monthly_pay_day = random.randint(1, 5)
        self.weekly_pay_day = 6 # Sunday
        
        # "Saver" personality sends more money home
        self.remittance_percentage = random.uniform(0.6, 0.85) * (1.1 if financial_personality == 'Saver' else 1)
        self.recharge_chance = 0.07

        self.balance = random.uniform(100, 500)

    def _handle_payday_and_remittance(self, date, events):
        """
        Simulates the entire payday cycle: receive wage, remit home, withdraw remainder.
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
            wage_txn = self.log_transaction("CREDIT", f"{self.pay_cycle.capitalize()} Wage ({self.work_city})", wage_amount, date)
            if wage_txn:
                events.append(wage_txn)
                remittance_amount = wage_amount * self.remittance_percentage
                remit_txn = self.log_transaction("DEBIT", f"Family Remittance to {self.home_state}", remittance_amount, date)
                if remit_txn:
                    events.append(remit_txn)

                if self.balance > 50:
                    cash_out_amount = self.balance * random.uniform(0.8, 1.0)
                    cash_txn = self.log_transaction("DEBIT", f"Cash Withdrawal ({self.work_city})", cash_out_amount, date)
                    if cash_txn:
                        events.append(cash_txn)

    def _handle_recharge(self, date, events):
        """Simulates small, infrequent mobile recharges."""
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([49, 99, 149])
            txn = self.log_transaction("DEBIT", "Prepaid Mobile Recharge", recharge_amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the Migrant Worker's financial life, centered on the
        payday remittance cycle.
        """
        events = []
        self._handle_payday_and_remittance(date, events)
        self._handle_recharge(date, events)
        return events