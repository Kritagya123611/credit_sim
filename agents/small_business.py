# agents/small_business_owner.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np

class SmallBusinessOwner(BaseAgent):
    """
    A multi-dimensional profile for a Small Business Owner.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Middle', financial_personality='Rational_Investor'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Small Business Owner"

        # --- RISK SCORE CALCULATION ---
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "50000-200000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-Employed",
            "employment_verification": "Udyam_Registered",
            "income_type": "Business_Sales",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Erratic_High_Volume",
            "savings_retention_rate": "Low",
            "has_investment_activity": True,
            "investment_types": ["Business_Reinvestment"],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
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
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        avg_monthly_turnover = random.uniform(min_mod, max_mod)

        self.daily_sales_chance = 0.95
        self.num_daily_sales = int(random.randint(10, 50) * income_multiplier)
        self.avg_sale_amount = random.uniform(200, 1500) * income_multiplier
        
        self.num_employees = int(random.randint(2, 5) * income_multiplier)
        self.employee_salaries = [random.uniform(8000, 15000) for _ in range(self.num_employees)]
        self.vendor_payment_day = random.randint(15, 20)
        self.vendor_payment_amount = avg_monthly_turnover * random.uniform(0.3, 0.5)
        self.business_loan_emi_amount = avg_monthly_turnover * 0.1
        self.owner_drawing_amount = avg_monthly_turnover * 0.2

        self.balance = random.uniform(avg_monthly_turnover * 0.2, avg_monthly_turnover * 0.4)

    def _handle_sales_income(self, date, events):
        """Simulates erratic daily sales from various sources."""
        if random.random() < self.daily_sales_chance:
            sales_multiplier = 2.0 if date.weekday() >= 5 else 1.0
            num_sales_today = int(self.num_daily_sales * sales_multiplier)
            
            for _ in range(num_sales_today):
                sale_amount = self.avg_sale_amount * random.uniform(0.5, 1.5)
                source = random.choice(["UPI QR Sale", "POS Card Sale", "Cash Deposit"])
                txn = self.log_transaction("CREDIT", source, sale_amount, date)
                if txn: events.append(txn)

    def _handle_operational_expenses(self, date, events):
        """Simulates payroll, vendor payments, loan EMIs, and owner's drawings."""
        if date.day == 28 and self.num_employees > 0:
            for i, salary in enumerate(self.employee_salaries):
                txn = self.log_transaction("DEBIT", f"Salary to Employee {i+1}", salary, date)
                if txn: events.append(txn)

        if date.day == self.vendor_payment_day:
            txn = self.log_transaction("DEBIT", "Vendor/Supplier Payment", self.vendor_payment_amount, date)
            if txn: events.append(txn)

        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Business Loan EMI", self.business_loan_emi_amount, date)
            if txn: events.append(txn)
            
        if date.day == 5:
            txn = self.log_transaction("DEBIT", "Owner's Drawings", self.owner_drawing_amount, date)
            if txn: events.append(txn)

    def _handle_utility_bills(self, date, events):
        """Simulates paying for commercial utilities."""
        if date.day == 25:
            commercial_bill = self.avg_sale_amount * 2.0
            txn = self.log_transaction("DEBIT", "Commercial Electricity Bill", commercial_bill, date)
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the daily financial operations of a small business.
        """
        events = []
        self._handle_sales_income(date, events)
        self._handle_operational_expenses(date, events)
        self._handle_utility_bills(date, events)
        return events