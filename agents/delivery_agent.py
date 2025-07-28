# agents/delivery_agent.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class DeliveryAgent(BaseAgent):
    """
    An enhanced profile for a Delivery Agent.
    Simulates high-velocity transactions, variable daily income, and the critical
    Cash-on-Delivery (COD) settlement pattern.
    """
    def __init__(self):
        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Delivery Agent / Rider",
            "risk_profile": "Medium",
            "employment_status": "Gig_Work_Contractor",
            "employment_verification": "Not_Verified",
            "income_type": "Platform_Payout",
            "avg_monthly_income_range": "15000-25000",
            "income_pattern": "Daily",
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": ["None"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": ["None"],
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
        min_monthly, max_monthly = map(int, self.avg_monthly_income_range.split('-'))
        self.base_daily_payout = random.uniform(min_monthly, max_monthly) / 26 # Assuming 26 working days

        # Loan is likely for their vehicle
        self.loan_emi_amount = (min_monthly + max_monthly) / 2 * 0.15

        # COD simulation parameters
        self.cod_settlement_chance = 0.60 # 60% of days involve COD transactions
        self.cod_balance = 0.0 # Tracks collected cash to be settled

        # Operational expense probabilities
        self.fuel_spend_chance = 0.90 # High chance of daily fuel costs
        self.recharge_chance = 0.10

        # Set a low starting balance
        self.balance = random.uniform(500, 2000)

    def _handle_income_and_settlements(self, date, events):
        """Simulates daily payouts and the critical COD settlement cycle."""
        # --- Daily Payout from Platform ---
        # Income varies day to day
        daily_payout = self.base_daily_payout * random.uniform(0.7, 1.4)
        txn = self.log_transaction("CREDIT", "Platform Payout", daily_payout, date)
        if txn: events.append(txn)

        # --- COD Simulation ---
        if random.random() < self.cod_settlement_chance:
            # 1. Collect COD: Simulate this as multiple small cash deposits/credits
            num_cod_orders = random.randint(3, 10)
            for _ in range(num_cod_orders):
                cod_amount = random.uniform(100, 800)
                txn = self.log_transaction("CREDIT", "Cash on Delivery Deposit", cod_amount, date)
                if txn:
                    events.append(txn)
                    self.cod_balance += cod_amount # Track cash to be paid back

            # 2. Settle COD with Company: A single large debit
            if self.cod_balance > 0:
                txn = self.log_transaction("DEBIT", "COD Settlement to Zomato/Swiggy", self.cod_balance, date)
                if txn:
                    events.append(txn)
                    self.cod_balance = 0.0 # Reset tracked COD

    def _handle_fixed_debits(self, date, events):
        """Handles loan payments."""
        if self.has_loan_emi and date.day == 10:
            # Payment might be a few days late
            if random.random() > 0.15: # 85% chance of paying on time
                txn = self.log_transaction("DEBIT", "Two-Wheeler Loan EMI", self.loan_emi_amount, date)
                if txn: events.append(txn)

    def _handle_operational_spending(self, date, events):
        """Simulates daily operational costs like fuel and mobile recharges."""
        # --- Daily Fuel ---
        if random.random() < self.fuel_spend_chance:
            fuel_amount = random.uniform(150, 400)
            txn = self.log_transaction("DEBIT", "UPI Spend - Fuel", fuel_amount, date)
            if txn: events.append(txn)

        # --- Frequent Prepaid Recharges ---
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([99, 149, 199])
            txn = self.log_transaction("DEBIT", "Prepaid Data Recharge", recharge_amount, date)
            if txn: events.append(txn)


    def act(self, date: datetime):
        """
        Simulates the daily high-velocity financial life of a delivery agent.
        """
        events = []
        self._handle_income_and_settlements(date, events)
        self._handle_fixed_debits(date, events)
        self._handle_operational_spending(date, events)
        return events