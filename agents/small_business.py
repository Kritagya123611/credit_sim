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
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Small Business Owner"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "50000-250000" # This is now interpreted as Monthly Revenue
        min_rev, max_rev = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_rev * income_multiplier)}-{int(max_rev * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-Employed",
            "employment_verification": "ITR_Verified",
            "income_type": "Business_Revenue",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Daily_High_Variance",
            "savings_retention_rate": "Medium",
            "has_investment_activity": True,
            "investment_types": ["Business_Reinvestment"],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": True,
            "insurance_types": ["Business_Insurance"],
            "utility_payment_status": "Mostly_On_Time",
            "mobile_plan_type": "Postpaid",
            "device_consistency_score": round(random.uniform(0.90, 0.98), 2),
            "ip_consistency_score": round(random.uniform(0.85, 0.95), 2),
            "sim_churn_rate": "Very_Low",
            "primary_digital_channels": ["Netbanking", "UPI_QR", "Cards"],
            "login_pattern": "Consistent_Daytime",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Medium",
        }
        
        super().__init__(**profile_attributes)

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_revenue = random.uniform(min_mod, max_mod)
        
        # --- FIX: Define realistic business cost structures as percentages of revenue ---
        self.cost_of_goods_sold_pct = random.uniform(0.40, 0.60)  # COGS is 40-60% of revenue
        self.operating_expenses_pct = random.uniform(0.20, 0.30) # Rent, salaries, etc.
        self.owner_salary_pct = 0.15 # Owner draws a 15% salary

        self.balance = random.uniform(self.avg_monthly_revenue * 0.5, self.avg_monthly_revenue)

    def _handle_income(self, date, events):
        """Simulates daily business revenue from sales."""
        # --- FIX: Anchor income to a stable baseline (avg_monthly_revenue), not dynamic multipliers ---
        daily_revenue_target = self.avg_monthly_revenue / 30
        # Add daily and weekend variance for realism
        variance_multiplier = 1.5 if date.weekday() >= 5 else 1.0
        actual_daily_revenue = daily_revenue_target * random.uniform(0.7, 1.3) * variance_multiplier
        
        income_source = random.choice(["POS Card Sale", "UPI QR Payment", "Cash Deposit"])
        txn = self.log_transaction("CREDIT", income_source, actual_daily_revenue, date)
        if txn:
            events.append(txn)
            # Pass the day's revenue to the expense handler
            return actual_daily_revenue
        return 0

    def _handle_business_expenses(self, date, daily_revenue, events):
        """
        --- NEW & IMPROVED: Simulates core business costs to realistically offset revenue ---
        """
        # 1. Cost of Goods Sold (COGS) - A variable cost debited daily based on sales
        if daily_revenue > 0:
            cogs_amount = daily_revenue * self.cost_of_goods_sold_pct
            txn = self.log_transaction("DEBIT", "Inventory/Supplier Payment", cogs_amount, date)
            if txn: events.append(txn)

        # 2. Fixed Operating Expenses (Rent, Salaries, Utilities) - Paid monthly
        if date.day == 28:
            opex_amount = self.avg_monthly_revenue * self.operating_expenses_pct
            txn = self.log_transaction("DEBIT", "Monthly Operating Expenses (Rent/Salaries)", opex_amount, date)
            if txn: events.append(txn)

        # 3. Owner's Salary - A fixed drawing from the business profits
        if date.day == 5:
            owner_salary = self.avg_monthly_revenue * self.owner_salary_pct
            txn = self.log_transaction("DEBIT", "Owner Salary Withdrawal", owner_salary, date)
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates a small business owner's daily financial cycle.
        """
        events = []
        # --- FIX: Restructured act() method for a realistic financial flow ---
        
        # 1. Generate revenue for the day
        todays_revenue = self._handle_income(date, events)
        
        # 2. Pay business costs related to revenue and fixed monthly bills
        self._handle_business_expenses(date, todays_revenue, events)
        
        # 3. Handle the owner's personal living expenses (inherited from BaseAgent)
        self._handle_daily_living_expenses(date, events)
        
        return events
