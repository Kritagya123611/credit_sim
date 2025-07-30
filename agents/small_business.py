import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class SmallBusinessOwner(BaseAgent):
    """
    A multi-dimensional profile for a Small Business Owner.
    Behavior is modified by economic_class and financial_personality.
    Updated with realistic P2P transaction handling and balanced finances.
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

        base_income_range = "50000-200000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

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
        
        super().__init__(**profile_attributes)

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        avg_monthly_turnover = random.uniform(min_mod, max_mod)

        # ✅ BALANCED: Limit sales volume to realistic levels
        self.daily_sales_chance = random.uniform(0.7, 0.85)  # Not every day has sales
        self.num_daily_sales = min(random.randint(8, 25), int(15 * income_multiplier))  # Cap daily sales
        self.avg_sale_amount = min(random.uniform(150, 800), 500 * income_multiplier)  # Cap sale amount
        
        # ✅ BALANCED: More realistic employee structure
        self.num_employees = min(random.randint(2, 6), int(3 + income_multiplier))
        self.employee_salaries = [random.uniform(8000, 18000) for _ in range(self.num_employees)]
        
        # ✅ BALANCED: Vendor payments and expenses scale appropriately
        self.vendor_payment_day = random.randint(15, 20)
        self.vendor_payment_amount = avg_monthly_turnover * random.uniform(0.35, 0.55)  # Higher expense ratio
        self.business_loan_emi_amount = avg_monthly_turnover * random.uniform(0.08, 0.15)  # Realistic EMI
        self.owner_drawing_amount = avg_monthly_turnover * random.uniform(0.15, 0.25)  # Moderate drawings
        
        # ✅ BALANCED: Additional business expenses
        self.rent_amount = avg_monthly_turnover * random.uniform(0.05, 0.10)
        self.inventory_restocking_amount = avg_monthly_turnover * random.uniform(0.20, 0.30)
        self.marketing_expense = avg_monthly_turnover * random.uniform(0.02, 0.05)
        
        # ✅ BALANCED: Track business cycles and bad debt
        self.bad_debt_chance = 0.05  # 5% chance of bad debts
        self.seasonal_down_months = random.sample(range(1, 13), 2)  # 2 slow months per year
        self.emergency_expense_chance = 0.03  # Unexpected business expenses

        # Enhanced P2P attributes
        self.employees = []
        self.suppliers = []
        self.business_network = []
        self.customer_network = []
        
        self.business_p2p_chance = 0.12
        self.supplier_advance_chance = 0.06
        self.customer_refund_chance = 0.08
        self.business_networking_chance = 0.08

        # ✅ BALANCED: Lower starting balance
        self.balance = random.uniform(avg_monthly_turnover * 0.1, avg_monthly_turnover * 0.3)

    def _handle_sales_income(self, date, events):
        """✅ BALANCED: Handles realistic daily sales with variability."""
        # Seasonal adjustment
        seasonal_multiplier = 0.7 if date.month in self.seasonal_down_months else 1.0
        
        if random.random() < (self.daily_sales_chance * seasonal_multiplier):
            weekend_boost = 1.3 if date.weekday() >= 5 else 1.0
            num_sales_today = max(1, int(self.num_daily_sales * weekend_boost * seasonal_multiplier))
            
            for _ in range(num_sales_today):
                # ✅ BALANCED: More realistic sale amounts with variability
                sale_amount = self.avg_sale_amount * random.uniform(0.3, 2.0)  # Higher variance
                source = random.choice(["UPI QR Sale", "POS Card Sale", "Cash Deposit"])
                channel = {"UPI QR Sale": "UPI", "POS Card Sale": "Card", "Cash Deposit": "Cash Deposit"}[source]
                txn = self.log_transaction("CREDIT", source, sale_amount, date, channel=channel)
                if txn: events.append(txn)

    def _handle_additional_business_expenses(self, date, events):
        """✅ NEW: Handles additional realistic business expenses."""
        # Monthly rent
        if date.day == 1:
            txn = self.log_transaction("DEBIT", "Business Rent", self.rent_amount, date, channel="Netbanking")
            if txn: events.append(txn)
        
        # Inventory restocking (bi-weekly)
        if date.day in [7, 21]:
            restock_amount = self.inventory_restocking_amount / 2
            txn = self.log_transaction("DEBIT", "Inventory Restocking", restock_amount, date, channel="Netbanking")
            if txn: events.append(txn)
        
        # Marketing expenses (monthly)
        if date.day == 12:
            txn = self.log_transaction("DEBIT", "Marketing & Advertising", self.marketing_expense, date, channel="Card")
            if txn: events.append(txn)
        
        # Bad debt write-offs (random)
        if random.random() < self.bad_debt_chance:
            bad_debt = self.avg_sale_amount * random.randint(3, 8)
            txn = self.log_transaction("DEBIT", "Bad Debt Write-off", bad_debt, date, channel="Bank Transfer")
            if txn: events.append(txn)
        
        # Emergency business expenses
        if random.random() < self.emergency_expense_chance:
            emergency_amount = random.uniform(2000, 10000)
            emergency_type = random.choice(["Equipment Repair", "Legal Fees", "Regulatory Compliance", "Emergency Stock"])
            txn = self.log_transaction("DEBIT", f"Emergency: {emergency_type}", emergency_amount, date, channel="Card")
            if txn: events.append(txn)

    def _handle_employee_salary_payments(self, date, events, context):
        """✅ UPDATED: Handles employee salary payments with realistic channels."""
        if date.day == 28 and self.employees:
            for i, employee in enumerate(self.employees):
                if i < len(self.employee_salaries):
                    salary = self.employee_salaries[i]
                    
                    if salary > 50000:
                        channel = random.choice(['IMPS', 'NEFT'])
                    else:
                        channel = RealisticP2PStructure.select_realistic_channel()
                    
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': employee, 
                        'amount': round(salary, 2), 
                        'desc': 'UPI P2P Transfer',
                        'channel': channel
                    })

    def _handle_supplier_advance_payments(self, date, events, context):
        """✅ UPDATED: More conservative supplier advance payments."""
        if (self.suppliers and 
            random.random() < self.supplier_advance_chance and
            self.balance > 30000):
            
            supplier = random.choice(self.suppliers)
            advance_amount = random.uniform(3000, 15000)  # Reduced from 5k-25k
            
            if self.economic_class in ['High', 'Upper_Middle']:
                advance_amount *= random.uniform(1.2, 1.8)  # Reduced multiplier
            
            if advance_amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif advance_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': supplier, 
                'amount': round(advance_amount, 2), 
                'desc': 'UPI P2P Transfer',
                'channel': channel
            })

    def _handle_customer_refunds(self, date, events, context):
        """✅ UPDATED: Customer refunds with realistic frequency."""
        if (self.customer_network and 
            random.random() < self.customer_refund_chance and
            self.balance > 5000):
            
            customer = random.choice(self.customer_network)
            refund_amount = random.uniform(200, 2000)  # Reduced max refund
            
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': customer, 
                'amount': round(refund_amount, 2), 
                'desc': 'UPI P2P Transfer',
                'channel': channel
            })

    def _handle_business_networking_transfers(self, date, events, context):
        """✅ UPDATED: More conservative business networking transfers."""
        if (self.business_network and 
            random.random() < self.business_networking_chance and
            self.balance > 15000):
            
            business_partner = random.choice(self.business_network)
            networking_amount = random.uniform(1000, 8000)  # Reduced from 2k-15k
            
            if self.economic_class in ['High', 'Upper_Middle']:
                networking_amount *= random.uniform(1.2, 1.5)  # Reduced multiplier
            
            if self.financial_personality == 'Rational_Investor':
                networking_amount *= random.uniform(1.1, 1.3)  # Reduced multiplier
            
            if networking_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': business_partner, 
                'amount': round(networking_amount, 2), 
                'desc': 'UPI P2P Transfer',
                'channel': channel
            })

    def _handle_operational_expenses(self, date, events):
        """✅ BALANCED: Handles fixed operational business expenses."""
        if date.day == self.vendor_payment_day:
            txn = self.log_transaction("DEBIT", "Vendor/Supplier Payment", self.vendor_payment_amount, date, channel="Netbanking")
            if txn: events.append(txn)

        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Business Loan EMI", self.business_loan_emi_amount, date, channel="Auto_Debit")
            if txn: events.append(txn)
            
        if date.day == 5:
            txn = self.log_transaction("DEBIT", "Owner's Drawings", self.owner_drawing_amount, date, channel="Bank Transfer")
            if txn: events.append(txn)

    def _handle_utility_bills(self, date, events):
        """✅ BALANCED: More realistic commercial utility bills."""
        if date.day == 25:
            # ✅ BALANCED: More realistic utility calculation
            commercial_bill = random.uniform(3000, 8000) + (self.num_employees * 500)
            txn = self.log_transaction("DEBIT", "Commercial Electricity Bill", commercial_bill, date, channel="Netbanking")
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """✅ BALANCED: Now includes all expense categories for realistic cash flow."""
        events = []
        self._handle_sales_income(date, events)
        self._handle_additional_business_expenses(date, events)  # ✅ NEW: Additional expenses
        self._handle_employee_salary_payments(date, events, context)
        self._handle_supplier_advance_payments(date, events, context)
        self._handle_customer_refunds(date, events, context)
        self._handle_business_networking_transfers(date, events, context)
        self._handle_operational_expenses(date, events)
        self._handle_utility_bills(date, events)
        self._handle_daily_living_expenses(date, events)
        return events
