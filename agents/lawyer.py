# agents/lawyer.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent

class Lawyer(BaseAgent):
    """
    An enhanced profile for a Lawyer or Consultant.
    Simulates a "lumpy" income cycle with infrequent, high-value payments,
    professional expenses, and investment behavior linked to cash flow.
    """
    def __init__(self):
        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Lawyer / Consultant",
            "risk_profile": "Medium",
            "employment_status": "Self-Employed_Professional",
            "employment_verification": "Professional_License_Verified",
            "income_type": "Professional_Fees",
            "avg_monthly_income_range": "50000-200000",
            "income_pattern": "Lumpy",
            "savings_retention_rate": "Medium",
            "has_investment_activity": True,
            "investment_types": ["Equity", "Real_Estate"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life", "Prof_Indemnity"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "High-Value_Postpaid",
            "device_consistency_score": round(random.uniform(0.88, 0.95), 2),
            "ip_consistency_score": round(random.uniform(0.82, 0.92), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Cards"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "High",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        # --- Lumpy Income Simulation ---
        # They get paid 2-4 times a year in big chunks
        self.payout_months = sorted(random.sample(range(1, 13), k=random.randint(2, 4)))
        min_monthly, max_monthly = map(int, self.avg_monthly_income_range.split('-'))
        avg_monthly = (min_monthly + max_monthly) / 2
        self.lump_sum_payment = avg_monthly * (12 / len(self.payout_months)) # Annual income divided by payouts
        self.has_large_cash_reserve = False # Flag to track if recently paid

        # --- Professional & Personal Expenses ---
        self.junior_retainer_fee = random.uniform(20000, 40000)
        self.loan_emi_amount = avg_monthly * 0.30
        self.prof_indemnity_premium = random.uniform(20000, 40000)

        # Set a healthy starting balance to manage expenses between payouts
        self.balance = random.uniform(80000, 200000)

    def _handle_lumpy_income(self, date, events):
        """Simulates receiving large, infrequent payments."""
        if date.month in self.payout_months and date.day == 15: # Payout in the middle of a payout month
            payout = self.lump_sum_payment * random.uniform(0.8, 1.2)
            txn = self.log_transaction("CREDIT", "Client/Project Fee Received", payout, date)
            if txn:
                events.append(txn)
                self.has_large_cash_reserve = True # Now they have cash for big expenses

    def _handle_recurring_debits(self, date, events):
        """Handles regular monthly and annual professional/personal expenses."""
        # --- Monthly Payments ---
        if date.day == 5: # Pay junior/retainer
            txn = self.log_transaction("DEBIT", "Junior/Retainer Fee", self.junior_retainer_fee, date)
            if txn: events.append(txn)

        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", self.loan_emi_amount, date)
            if txn: events.append(txn)

        # --- Annual Payments ---
        if self.has_insurance_payments and date.month == 7 and date.day == 20: # Indemnity insurance in July
            txn = self.log_transaction("DEBIT", "Professional Indemnity Insurance", self.prof_indemnity_premium, date)
            if txn: events.append(txn)

    def _handle_spending_and_investment(self, date, events):
        """Simulates spending and large investments, often after a payout."""
        # Higher chance of big spending or investment if they have cash reserves
        if self.has_large_cash_reserve:
            if random.random() < 0.5: # 50% chance to make a big investment after getting paid
                investment_amount = self.balance * random.uniform(0.3, 0.6) # Invest 30-60% of current balance
                txn = self.log_transaction("DEBIT", "Lump-Sum Equity/Real Estate Investment", investment_amount, date)
                if txn:
                    events.append(txn)
                    self.has_large_cash_reserve = False # Reset flag after investing

        # Standard daily/weekly spending
        if random.random() < 0.4:
            spend_category = random.choice(["Fine Dining", "Travel Booking", "Professional Books"])
            amount = random.uniform(1000, 8000)
            txn = self.log_transaction("DEBIT", f"Card Spend - {spend_category}", amount, date)
            if txn: events.append(txn)

    def act(self, date: datetime):
        """
        Simulates the lawyer's "feast or famine" financial cycle.
        """
        events = []
        self._handle_lumpy_income(date, events)
        self._handle_recurring_debits(date, events)
        self._handle_spending_and_investment(date, events)
        return events