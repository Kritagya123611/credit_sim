import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np

class GovernmentEmployee(BaseAgent):
    """
    A multi-dimensional profile for a Government Employee.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Middle', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Government Employee"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "35000-70000"
        min_sal, max_sal = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_sal * income_multiplier)}-{int(max_sal * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Salaried",
            "employment_verification": "GOVT_Verified",
            "income_type": "Government_Salary",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "High",
            "has_investment_activity": True,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["LIC", "Government_Schemes"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid",
            "device_consistency_score": round(random.uniform(0.95, 0.99), 2),
            "ip_consistency_score": round(random.uniform(0.92, 0.98), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Medium",
        }
        
        super().__init__(**profile_attributes)

        self.salary_day = 1
        min_sal_mod, max_sal_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal_mod, max_sal_mod)
        
        self.emi_percentage = 0.30
        self.investment_percentage = 0.10 * personality_config['invest_chance_mod']
        self.insurance_percentage = 0.08
        self.utility_bill_percentage = 0.05
        self.remittance_percentage = 0.20 * (1.2 if financial_personality == 'Saver' else 1)

        self.ecommerce_spend_chance = 0.05 * personality_config['spend_chance_mod']
        
        self.family_member_recipient = None

        self.balance = random.uniform(self.salary_amount * 0.3, self.salary_amount * 0.8)

    def _handle_recurring_events(self, date, events, context):
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Govt Salary Deposit", self.salary_amount, date, channel="Bank Transfer")
            if txn: 
                events.append(txn)
                if self.family_member_recipient:
                    remittance_amount = self.salary_amount * self.remittance_percentage
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': self.family_member_recipient, 
                        'amount': remittance_amount, 
                        'desc': 'Family Support',
                        'channel': 'UPI'  # Specifying channel for clarity
                    })
            
        if self.has_loan_emi and date.day == 5:
            emi_amount = self.salary_amount * self.emi_percentage
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", emi_amount, date, channel="Auto_Debit")
            if txn: events.append(txn)

        if self.has_insurance_payments and date.day == 10:
            insurance_total = self.salary_amount * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "LIC Premium", insurance_total, date, channel="Auto_Debit")
            if txn: events.append(txn)

        if self.has_investment_activity and date.day == 15:
            invest_amt = self.salary_amount * self.investment_percentage
            txn = self.log_transaction("DEBIT", "PPF/FD Investment", invest_amt, date, channel="Netbanking")
            if txn: events.append(txn)
            
        if date.day == 20:
            bill = self.salary_amount * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", bill, date, channel="Netbanking")
            if txn: events.append(txn)

    def _handle_daily_spending(self, date, events):
        if random.random() < self.ecommerce_spend_chance:
            ecommerce_amt = random.uniform(500, 2500)
            txn = self.log_transaction("DEBIT", "E-commerce Purchase (Essentials)", ecommerce_amt, date, channel="Card")
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        events = []
        self._handle_recurring_events(date, events, context)
        self._handle_daily_spending(date, events)
        self._handle_daily_living_expenses(date, events)
        return events