# agents/doctor.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent # Make sure this import path is correct

class Doctor(BaseAgent):
    """
    A specific agent profile for a Doctor or private healthcare professional.
    Represents a low-risk, high-income individual with "lumpy" income and prudent financial habits.
    """
    def __init__(self):
        # 1. Define all profile attributes for the Doctor
        profile_attributes = {
            "archetype_name": "Doctor / Healthcare Worker",
            "risk_profile": "Low",
            "employment_status": "Self-Employed_Professional",
            "employment_verification": "Medical_License_Verified",
            "income_type": "Professional_Fees",
            "avg_monthly_income_range": "70000-300000",
            "income_pattern": "Lumpy",
            "savings_retention_rate": "High",
            "has_investment_activity": True,
            "investment_types": ["Equity", "Mutual_Funds"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life", "Prof_Indemnity"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "High-Value_Postpaid",
            "device_consistency_score": round(random.uniform(0.88, 0.96), 2),
            "ip_consistency_score": round(random.uniform(0.88, 0.96), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Cards"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        # Income parameters
        self.consultation_chance = 0.85 # 85% chance of consultations on a weekday
        self.avg_consultation_fee = random.uniform(500, 1500)
        self.num_consultations = random.randint(5, 15)
        self.large_payout_chance = 0.25 # 25% chance of a large payout each month

        # Expense and Investment parameters
        self.clinic_rent_amount = random.uniform(20000, 50000)
        self.sip_amount = random.uniform(20000, 50000)
        self.loan_emi_amount = random.uniform(30000, 70000)
        self.prof_indemnity_premium = random.uniform(25000, 50000)

        # Spending parameters
        self.high_end_spend_chance = 0.10 # 10% chance per day

        # Set a healthy starting balance
        self.balance = random.uniform(100000, 250000)

    def _handle_income(self, date, events):
        """Simulates a mix of regular consultation fees and large, lumpy payouts."""
        # --- Daily Consultation Fees (weekdays only) ---
        if date.weekday() < 5 and random.random() < self.consultation_chance:
            for _ in range(random.randint(1, self.num_consultations)):
                fee = self.avg_consultation_fee * random.uniform(0.9, 1.1)
                source = random.choice(["Patient UPI", "Patient Card Payment"])
                txn = self.log_transaction("CREDIT", source, fee, date)
                if txn: events.append(txn)
        
        # --- Large, Infrequent Payout (lumpy income spike) ---
        if date.day == 15 and random.random() < self.large_payout_chance:
            payout_amount = random.uniform(80000, 250000)
            source = random.choice(["Gateway Payout (Practo)", "Surgery Fee"])
            txn = self.log_transaction("CREDIT", source, payout_amount, date)
            if txn: events.append(txn)

    def _handle_professional_and_fixed_expenses(self, date, events):
        """Simulates paying for clinic rent, EMIs, investments, and insurance."""
        # --- Clinic Rent on the 5th ---
        if date.day == 5:
            txn = self.log_transaction("DEBIT", "Clinic Rent", self.clinic_rent_amount, date)
            if txn: events.append(txn)

        # --- Loan EMI on the 10th ---
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", self.loan_emi_amount, date)
            if txn: events.append(txn)
            
        # --- SIP Investment on the 15th ---
        if self.has_investment_activity and date.day == 15:
            txn = self.log_transaction("DEBIT", "Mutual Fund SIP", self.sip_amount, date)
            if txn: events.append(txn)

        # --- Annual Professional Indemnity Insurance (e.g., in June) ---
        if self.has_insurance_payments and date.month == 6 and date.day == 20:
             txn = self.log_transaction("DEBIT", "Professional Indemnity Insurance", self.prof_indemnity_premium, date)
             if txn: events.append(txn)

    def _handle_discretionary_spending(self, date, events):
        """Simulates high-end lifestyle spending."""
        if random.random() < self.high_end_spend_chance:
            spend_category = random.choice(["Fine Dining", "Luxury Goods", "Travel Booking", "Electronics"])
            spend_amount = random.uniform(5000, 25000)
            txn = self.log_transaction("DEBIT", f"Card Spend - {spend_category}", spend_amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime):
        """
        Simulates the doctor's complex financial life, balancing regular outgoings
        with lumpy professional income.
        """
        events = []
        self._handle_income(date, events)
        self._handle_professional_and_fixed_expenses(date, events)
        self._handle_discretionary_spending(date, events)
        return events