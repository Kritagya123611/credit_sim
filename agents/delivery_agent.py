# agents/delivery_agent.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES

class DeliveryAgent(BaseAgent):
    """
    A multi-dimensional profile for a Delivery Agent.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower', financial_personality='Over_Spender'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])

        base_income_range = "15000-25000"
        min_monthly, max_monthly = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_monthly * income_multiplier)}-{int(max_monthly * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Delivery Agent / Rider",
            "risk_profile": "Medium",
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Gig_Work_Contractor",
            "employment_verification": "Not_Verified",
            "income_type": "Platform_Payout",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Daily",
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "Mostly_On_Time",
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.88, 0.95), 2),
            "ip_consistency_score": round(random.uniform(0.30, 0.50), 2),
            "sim_churn_rate": "Medium",
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Geographically_Dynamic",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Low",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.base_daily_payout = random.uniform(min_mod, max_mod) / 26 # Assuming 26 working days

        self.loan_emi_amount = (min_mod + max_mod) / 2 * 0.15
        self.cod_settlement_chance = 0.60
        self.cod_balance = 0.0

        # Modify spending chances based on personality
        self.fuel_spend_chance = 0.90 * personality_config['spend_chance_mod']
        self.recharge_chance = 0.10 * personality_config['spend_chance_mod']

        self.balance = random.uniform(500, 2000)

    def _handle_income_and_settlements(self, date, events):
        """Simulates daily payouts and the critical COD settlement cycle."""
        daily_payout = self.base_daily_payout * random.uniform(0.7, 1.4)
        txn = self.log_transaction("CREDIT", "Platform Payout", daily_payout, date)
        if txn: events.append(txn)

        if random.random() < self.cod_settlement_chance:
            num_cod_orders = random.randint(3, 10)
            for _ in range(num_cod_orders):
                cod_amount = random.uniform(100, 800)
                txn = self.log_transaction("CREDIT", "Cash on Delivery Deposit", cod_amount, date)
                if txn:
                    events.append(txn)
                    self.cod_balance += cod_amount

            if self.cod_balance > 0:
                txn = self.log_transaction("DEBIT", "COD Settlement to Platform", self.cod_balance, date)
                if txn:
                    events.append(txn)
                    self.cod_balance = 0.0

    def _handle_fixed_debits(self, date, events):
        """Handles loan payments."""
        if self.has_loan_emi and date.day == 10:
            if random.random() > 0.15: # 85% chance of paying on time
                txn = self.log_transaction("DEBIT", "Two-Wheeler Loan EMI", self.loan_emi_amount, date)
                if txn: events.append(txn)

    def _handle_operational_spending(self, date, events):
        """Simulates daily operational costs like fuel and mobile recharges."""
        if random.random() < self.fuel_spend_chance:
            fuel_amount = random.uniform(150, 400)
            txn = self.log_transaction("DEBIT", "UPI Spend - Fuel", fuel_amount, date)
            if txn: events.append(txn)

        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([99, 149, 199])
            txn = self.log_transaction("DEBIT", "Prepaid Data Recharge", recharge_amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the daily high-velocity financial life of a delivery agent.
        """
        events = []
        self._handle_income_and_settlements(date, events)
        self._handle_fixed_debits(date, events)
        self._handle_operational_spending(date, events)
        return events