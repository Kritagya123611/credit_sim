# agents/tech_professional.py

import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES

class TechProfessional(BaseAgent):
    """
    A multi-dimensional profile for a Tech Professional.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Upper_Middle', financial_personality='Rational_Investor'):
        
        # --- Dynamically modify parameters based on new dimensions ---
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])

        base_income_range = "60000-200000"
        min_sal, max_sal = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_sal * income_multiplier)}-{int(max_sal * income_multiplier)}"

        # 1. Define all profile attributes
        profile_attributes = {
            "archetype_name": "Tech Professional",
            "risk_profile": "Low",
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Salaried",
            "employment_verification": "EPFO_Verified",
            "income_type": "Salary_IT",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date_with_Bonus",
            "savings_retention_rate": "Medium",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "High-Value_Postpaid",
            "device_consistency_score": round(random.uniform(0.92, 0.98), 2),
            "ip_consistency_score": round(random.uniform(0.60, 0.75), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["All"],
            "login_pattern": "Geographically_Dynamic",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
        }
        
        # 2. Call the parent's __init__ method
        super().__init__(**profile_attributes)

        # 3. Define behavioral and simulation parameters
        self.salary_day = random.randint(1, 5)
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_mod, max_mod)
        
        self.stock_investment_amount = self.salary_amount * random.uniform(0.15, 0.25) * personality_config['invest_chance_mod']
        self.crypto_investment_amount = self.salary_amount * random.uniform(0.05, 0.10) * (2 if financial_personality == 'Risk_Addict' else 1)
        self.loan_emi_amount = self.salary_amount * 0.20
        self.saas_subscription_amount = random.uniform(500, 2000)

        self.is_traveling = False
        self.travel_start_day = 0
        self.travel_duration = 0
        
        self.annual_bonus_month = random.choice([3, 4])
        self.has_received_bonus_this_year = False

        self.balance = random.uniform(self.salary_amount * 0.5, self.salary_amount)

    def _handle_income(self, date, events):
        """Handles monthly salary and large annual bonuses/stock sales."""
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Salary Credit (IT)", self.salary_amount, date)
            if txn: events.append(txn)
        
        if date.month == self.annual_bonus_month and date.day == self.salary_day and not self.has_received_bonus_this_year:
            bonus_amount = self.salary_amount * random.uniform(3.0, 6.0)
            txn = self.log_transaction("CREDIT", "Annual Bonus/RSU Sale", bonus_amount, date)
            if txn: events.append(txn)
            self.has_received_bonus_this_year = True
        
        if date.month == 1: self.has_received_bonus_this_year = False

    def _handle_fixed_debits(self, date, events):
        """Handles recurring payments for loans, utilities, and investments."""
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", self.loan_emi_amount, date)
            if txn: events.append(txn)
            
        if self.has_investment_activity:
            if "Stocks" in self.investment_types and date.day == 5:
                txn = self.log_transaction("DEBIT", "Stock Investment (Zerodha)", self.stock_investment_amount, date)
                if txn: events.append(txn)
            if "Crypto" in self.investment_types and date.day == 15:
                txn = self.log_transaction("DEBIT", "Crypto Investment (WazirX)", self.crypto_investment_amount, date)
                if txn: events.append(txn)

        if date.day == 20:
            txn = self.log_transaction("DEBIT", "SaaS Subscriptions (Cloud/VPN)", self.saas_subscription_amount, date)
            if txn: events.append(txn)

    def _handle_dynamic_spending(self, date, events):
        """Simulates their dynamic lifestyle, including travel."""
        if date.day == 1 and date.month in [1, 4, 7, 10]:
            if random.random() < 0.5:
                self.is_traveling = True
                self.travel_start_day = random.randint(5, 15)
                self.travel_duration = random.randint(7, 14)
                
                travel_cost = random.uniform(20000, 80000) * (1.5 if self.economic_class in ['Upper_Middle', 'High'] else 1)
                txn = self.log_transaction("DEBIT", "Travel Booking (Flights/Hotels)", travel_cost, date)
                if txn: events.append(txn)
        
        if self.is_traveling:
            if date.day >= self.travel_start_day and date.day < self.travel_start_day + self.travel_duration:
                if random.random() < 0.9:
                    spend = random.uniform(1000, 5000)
                    txn = self.log_transaction("DEBIT", "Travel/Forex Card Spend", spend, date)
                    if txn: events.append(txn)
            else:
                self.is_traveling = False
        else:
            if random.random() < 0.3:
                spend = random.uniform(1500, 6000)
                txn = self.log_transaction("DEBIT", "E-commerce/Dining", spend, date)
                if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """
        Simulates the tech professional's daily life, blending stable income
        with modern investment and a dynamic, travel-heavy lifestyle.
        """
        events = []
        self._handle_income(date, events)
        self._handle_fixed_debits(date, events)
        self._handle_dynamic_spending(date, events)
        return events