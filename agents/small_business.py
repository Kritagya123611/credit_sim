# agents/small_business_owner.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent # Make sure this import path is correct

class SmallBusinessOwner(BaseAgent):
    """
    A specific agent profile for a Small Business Owner (e.g., Kirana Shop).
    Characterized by high-volume, erratic sales and complex operational expenses.
    """
    def __init__(self):
        # 1. Define all profile attributes for the Small Business Owner
        profile_attributes = {
            "archetype_name": "Small Business Owner",
            "risk_profile": "Medium",
            "employment_status": "Self-Employed",
            "employment_verification": "Udyam_Registered",
            "income_type": "Business_Sales",
            "avg_monthly_income_range": "50000-200000",
            "income_pattern": "Erratic_High_Volume",
            "savings_retention_rate": "Low",
            "has_investment_activity": True,
            "investment_types": ["Business_Reinvestment"],
            "has_loan_emi": True,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": True,
            "insurance_types": ["Business_Insurance"],
            "utility_payment_status": "Mostly_On_Time",
            "mobile_plan_type": "Postpaid",
            "device_consistency_score": round(random.uniform(0.80, 0.90), 2),
            "ip_consistency_score": round(random.uniform(0.88, 0.95), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["UPI_for_Business", "Netbanking"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "High",
        }

        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.daily_sales_chance = 0.95  # High chance of having sales every day
        self.avg_sale_amount = random.uniform(200, 1500)
        self.num_daily_sales = random.randint(10, 50)
        
        # Business expenses
        self.num_employees = random.randint(2, 5)
        self.employee_salaries = [random.uniform(8000, 15000) for _ in range(self.num_employees)]
        self.vendor_payment_day = random.randint(15, 20)
        self.vendor_payment_amount = random.uniform(20000, 50000)
        self.business_loan_emi_amount = random.uniform(5000, 15000)
        self.owner_drawing_amount = random.uniform(15000, 30000)

        # Set a moderate starting balance for cash flow
        self.balance = random.uniform(20000, 50000)

    def _handle_sales_income(self, date, events):
        """Simulates erratic daily sales from various sources."""
        if random.random() < self.daily_sales_chance:
            # Sales are higher on weekends
            sales_multiplier = 2.0 if date.weekday() >= 5 else 1.0
            num_sales_today = int(self.num_daily_sales * sales_multiplier)
            
            for _ in range(num_sales_today):
                sale_amount = self.avg_sale_amount * random.uniform(0.5, 1.5)
                source = random.choice(["UPI QR Sale", "POS Card Sale", "Cash Deposit"])
                txn = self.log_transaction("CREDIT", source, sale_amount, date)
                if txn: events.append(txn)

    def _handle_operational_expenses(self, date, events):
        """Simulates payroll, vendor payments, loan EMIs, and owner's drawings."""
        # --- Monthly Payroll (creates one-to-many pattern) ---
        if date.day == 28:
            for i, salary in enumerate(self.employee_salaries):
                txn = self.log_transaction("DEBIT", f"Salary to Employee {i+1}", salary, date)
                if txn: events.append(txn)

        # --- Vendor Payments ---
        if date.day == self.vendor_payment_day:
            txn = self.log_transaction("DEBIT", "Vendor/Supplier Payment", self.vendor_payment_amount, date)
            if txn: events.append(txn)

        # --- Business Loan EMI ---
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Business Loan EMI", self.business_loan_emi_amount, date)
            if txn: events.append(txn)
            
        # --- Owner's Drawings for personal expenses ---
        if date.day == 5:
            txn = self.log_transaction("DEBIT", "Owner's Drawings", self.owner_drawing_amount, date)
            if txn: events.append(txn)

    def _handle_utility_bills(self, date, events):
        """Simulates paying for commercial utilities."""
        if date.day == 25:
            commercial_bill = random.uniform(3000, 8000)
            txn = self.log_transaction("DEBIT", "Commercial Electricity Bill", commercial_bill, date)
            if txn: events.append(txn)

    def act(self, date: datetime):
        """
        Simulates the daily financial operations of a small business.
        """
        events = []
        self._handle_sales_income(date, events)
        self._handle_operational_expenses(date, events)
        self._handle_utility_bills(date, events)
        return events