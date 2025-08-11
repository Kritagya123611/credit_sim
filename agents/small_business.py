import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class SmallBusinessOwner(BaseAgent):
    """
    Enhanced Small Business Owner agent for Phase 2: Business entity salary source tracking
    Includes business entities as income sources, realistic business relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Middle', financial_personality='Rational_Investor'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Small Business Owner"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
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
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.78, 0.92), 3),
            "ip_consistency_score": round(random.uniform(0.85, 0.95), 3), 
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["UPI_for_Business", "Netbanking"], 
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium", 
            "ecommerce_avg_ticket_size": "High",
            
            # ✅ NEW: Heterogeneous graph connections specific to SmallBusinessOwner
            "industry_sector": "Small_Business_Retail",
            "company_size": "Small",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Business entities as income sources (salary source tracking)
        self.business_entities = []  # Business companies as income sources
        self.primary_business_entity_id = None  # Main business entity
        self.business_entity_hierarchy = {}  # Track business types and revenue streams

        # ✅ NEW: Business relationship tracking
        self.business_tenure = random.randint(12, 180)  # 1-15 years in business
        self.income_consistency = random.uniform(0.7, 0.9)  # Variable business income
        self.last_revenue_date = None
        self.seasonal_patterns = {}

        # Financial calculations with more variation
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_turnover = random.uniform(min_mod, max_mod)

        # ✅ ENHANCED: More balanced business operations
        self.daily_sales_chance = random.uniform(0.65, 0.85)
        self.num_daily_sales = min(random.randint(6, 22), int(12 + income_multiplier * 8))
        self.avg_sale_amount = min(random.uniform(120, 750), 400 + income_multiplier * 300)
        
        # Employee structure with variation
        self.num_employees = min(random.randint(1, 8), int(2 + income_multiplier * 2))
        self.employee_salaries = [random.uniform(7500, 19000) for _ in range(self.num_employees)]
        
        # Business expenses with more realistic ratios
        self.vendor_payment_day = random.randint(12, 22)
        self.vendor_payment_amount = self.avg_monthly_turnover * random.uniform(0.30, 0.50)
        self.business_loan_emi_amount = self.avg_monthly_turnover * random.uniform(0.06, 0.14)
        self.owner_drawing_amount = self.avg_monthly_turnover * random.uniform(0.12, 0.28)
        
        # Additional business expenses with variation
        self.rent_amount = self.avg_monthly_turnover * random.uniform(0.04, 0.09)
        self.inventory_restocking_amount = self.avg_monthly_turnover * random.uniform(0.18, 0.32)
        self.marketing_expense = self.avg_monthly_turnover * random.uniform(0.015, 0.055)
        
        # Business risk factors with variation
        self.bad_debt_chance = random.uniform(0.03, 0.07)
        self.seasonal_down_months = random.sample(range(1, 13), random.randint(1, 3))
        self.emergency_expense_chance = random.uniform(0.02, 0.05)

        # ✅ Enhanced P2P networks for Small Business Owners
        self.employees = []
        self.suppliers = []
        self.business_network = []
        self.customer_network = []
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.business_entities = []  # Business companies as income sources
        self.business_partners = []  # Other small businesses as company nodes
        self.banks = []  # Banking relationships as company nodes
        self.trade_associations = []  # Business associations as special nodes
        self.government_agencies = []  # Regulatory bodies as institutional nodes
        
        # P2P transfer probabilities with more variation
        self.business_p2p_chance = random.uniform(0.10, 0.14)
        self.supplier_advance_chance = random.uniform(0.05, 0.08)
        self.customer_refund_chance = random.uniform(0.06, 0.10)
        self.business_networking_chance = random.uniform(0.06, 0.10)

        # Temporal tracking with enhanced features
        self.last_major_purchase_date = None
        self.cash_flow_cycles = []
        self.supplier_payment_patterns = []
        self.seasonal_revenue_patterns = []

        self.balance = random.uniform(self.avg_monthly_turnover * 0.05, self.avg_monthly_turnover * 0.4)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Small business owners typically have 2-4 devices (phone, business phone, tablet, laptop)"""
        device_options = [2, 3, 4, 5]
        weights = [0.2, 0.4, 0.3, 0.1]  # Most have 3-4 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_business_entities(self, business_entity_ids):
        """✅ NEW: Assign business entities as income sources for salary tracking"""
        self.business_entities = business_entity_ids
        
        if business_entity_ids:
            # Assign primary business entity as main income source
            self.primary_business_entity_id = random.choice(business_entity_ids)
            self.assign_employer(
                company_id=self.primary_business_entity_id,
                employment_start_date=datetime.now().date() - timedelta(days=self.business_tenure * 30)
            )
            
            # Set up business entity hierarchy
            for entity_id in business_entity_ids:
                self.business_entity_hierarchy[entity_id] = {
                    'business_type': random.choice(['Retail', 'Restaurant', 'Services', 'Manufacturing', 'Trading']),
                    'revenue_share': random.uniform(0.6, 1.4),  # Multiplier for revenue
                    'payment_reliability': random.uniform(0.7, 0.95)
                }

    def _handle_business_entity_revenue_payment(self, date, events):
        """✅ NEW: Handle daily revenue from business entities"""
        # Seasonal adjustment
        seasonal_multiplier = 0.6 if date.month in self.seasonal_down_months else 1.0
        
        if (random.random() < (self.daily_sales_chance * seasonal_multiplier * self.income_consistency)):
            entity_id = self.primary_business_entity_id or (
                random.choice(self.business_entities) if self.business_entities else None
            )
            
            # Calculate daily revenue with business patterns
            weekend_boost = 1.4 if date.weekday() >= 5 else 1.0
            num_sales_today = max(1, int(self.num_daily_sales * weekend_boost * seasonal_multiplier))
            
            total_revenue = 0
            for _ in range(num_sales_today):
                # Enhanced sale amount calculation
                base_sale = self.avg_sale_amount * random.uniform(0.2, 2.5)
                
                # Business entity revenue share
                if entity_id:
                    revenue_multiplier = self.business_entity_hierarchy.get(entity_id, {}).get('revenue_share', 1.0)
                    base_sale *= revenue_multiplier
                
                total_revenue += base_sale
            
            # ✅ NEW: Log as salary transaction from business entity
            if entity_id:
                txn = self.log_salary_transaction(
                    amount=total_revenue,
                    date=date,
                    company_id=entity_id
                )
                if txn:
                    txn['transaction_category'] = 'business_entity_revenue'
                    txn['company_type'] = 'business_entity'
                    txn['business_type'] = self.business_entity_hierarchy.get(entity_id, {}).get('business_type', 'Unknown')
                    events.append(txn)
            else:
                # Multiple sales transactions for realism
                for _ in range(min(num_sales_today, 5)):  # Limit to 5 transactions per day
                    sale_amount = total_revenue / min(num_sales_today, 5)
                    source = random.choice(["UPI QR Sale", "POS Card Sale", "Cash Deposit"])
                    channel = {"UPI QR Sale": "UPI", "POS Card Sale": "Card", "Cash Deposit": "Cash Deposit"}[source]
                    
                    txn = self.log_transaction("CREDIT", source, sale_amount, date, channel=channel)
                    if txn:
                        events.append(txn)
            
            self.last_revenue_date = date
            return total_revenue
        
        return 0

    def _handle_sales_income(self, date, events):
        """✅ UPDATED: Enhanced sales income with business entity tracking"""
        self._handle_business_entity_revenue_payment(date, events)

    def _handle_additional_business_expenses(self, date, events):
        """✅ UPDATED: Enhanced business expenses with merchant tracking"""
        # Monthly rent with variation
        if date.day == 1:
            rent_variation = random.uniform(0.95, 1.05)
            actual_rent = self.rent_amount * rent_variation
            
            rent_provider_id = f"business_property_{hash(self.agent_id) % 200}"
            
            txn = self.log_merchant_transaction(
                merchant_id=rent_provider_id,
                amount=actual_rent,
                description="Business Premises Rent",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)
        
        # Inventory restocking with enhanced tracking
        if date.day in [8, 22]:  # Bi-monthly restocking
            restock_variation = random.uniform(0.8, 1.3)
            restock_amount = (self.inventory_restocking_amount / 2) * restock_variation
            
            inventory_supplier_id = f"inventory_supplier_{hash(self.agent_id + str(date)) % 300}"
            
            txn = self.log_merchant_transaction(
                merchant_id=inventory_supplier_id,
                amount=restock_amount,
                description="Business Inventory Restocking",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)
        
        # Marketing expenses with variation
        if date.day == 15:
            marketing_variation = random.uniform(0.7, 1.4)
            actual_marketing = self.marketing_expense * marketing_variation
            
            marketing_provider_id = f"marketing_agency_{hash(self.agent_id) % 150}"
            
            txn = self.log_merchant_transaction(
                merchant_id=marketing_provider_id,
                amount=actual_marketing,
                description="Business Marketing & Advertising",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)
        
        # Bad debt write-offs with realistic tracking
        if random.random() < self.bad_debt_chance:
            bad_debt = self.avg_sale_amount * random.randint(2, 12)
            
            txn = self.log_transaction(
                "DEBIT", "Bad Debt Write-off", bad_debt, date, channel="Internal_Transfer"
            )
            if txn:
                events.append(txn)
        
        # Emergency business expenses
        if random.random() < self.emergency_expense_chance:
            emergency_categories = [
                ("Equipment_Repair", random.uniform(1500, 8000)),
                ("Legal_Compliance", random.uniform(2000, 12000)),
                ("Emergency_Stock", random.uniform(3000, 15000)),
                ("Maintenance", random.uniform(1000, 5000))
            ]
            
            category, emergency_amount = random.choice(emergency_categories)
            emergency_provider_id = f"emergency_{category}_{hash(self.agent_id + str(date)) % 200}"
            
            txn = self.log_merchant_transaction(
                merchant_id=emergency_provider_id,
                amount=emergency_amount,
                description=f"Emergency {category.replace('_', ' ')}",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def _handle_employee_salary_payments(self, date, events, context):
        """✅ UPDATED: Enhanced employee salary payments with realistic channels"""
        if date.day == 28 and self.employees:
            for i, employee in enumerate(self.employees):
                if i < len(self.employee_salaries):
                    # Add salary variation
                    base_salary = self.employee_salaries[i]
                    salary_variation = random.uniform(0.98, 1.02)
                    actual_salary = base_salary * salary_variation
                    
                    # ✅ NEW: Select realistic channel based on salary amount
                    if actual_salary > 100000:
                        channel = random.choice(['NEFT', 'RTGS'])
                    elif actual_salary > 50000:
                        channel = random.choice(['IMPS', 'NEFT'])
                    else:
                        channel = RealisticP2PStructure.select_realistic_channel()
                    
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': employee, 
                        'amount': round(actual_salary, 2), 
                        'desc': 'Business Employee Salary',
                        'channel': channel,
                        'transaction_category': 'employee_salary'
                    })

    def _handle_supplier_advance_payments(self, date, events, context):
        """✅ UPDATED: Enhanced supplier advance payments"""
        if (self.suppliers and 
            random.random() < self.supplier_advance_chance and
            self.balance > 25000):
            
            supplier = random.choice(self.suppliers)
            advance_amount = random.uniform(2500, 12000)
            
            # Economic class adjustments
            economic_multiplier = {
                'Lower_Middle': random.uniform(0.8, 1.1),
                'Middle': random.uniform(1.0, 1.3),
                'Upper_Middle': random.uniform(1.2, 1.7),
                'High': random.uniform(1.5, 2.2)
            }.get(self.economic_class, 1.0)
            
            final_amount = advance_amount * economic_multiplier
            
            # ✅ NEW: Select realistic channel based on amount
            if final_amount > 200000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif final_amount > 100000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': supplier, 
                'amount': round(final_amount, 2), 
                'desc': 'Business Supplier Advance',
                'channel': channel,
                'transaction_category': 'supplier_advance'
            })

    def _handle_customer_refunds(self, date, events, context):
        """✅ UPDATED: Enhanced customer refunds with realistic frequency"""
        if (self.customer_network and 
            random.random() < self.customer_refund_chance and
            self.balance > 3000):
            
            customer = random.choice(self.customer_network)
            refund_amount = random.uniform(150, 1800)
            
            # Adjust based on business type and class
            if self.economic_class in ['Upper_Middle', 'High']:
                refund_amount *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': customer, 
                'amount': round(refund_amount, 2), 
                'desc': 'Customer Refund',
                'channel': channel,
                'transaction_category': 'customer_refund'
            })

    def _handle_business_networking_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced business networking transfers"""
        if (self.business_network and 
            random.random() < self.business_networking_chance and
            self.balance > 8000):
            
            business_partner = random.choice(self.business_network)
            networking_amount = random.uniform(800, 6000)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                networking_amount *= random.uniform(1.2, 1.6)
            
            # Business tenure adjustments
            if self.business_tenure > 60:  # 5+ years in business
                networking_amount *= random.uniform(1.1, 1.4)
            
            # Financial personality adjustments
            if self.financial_personality == 'Rational_Investor':
                networking_amount *= random.uniform(1.05, 1.25)
            
            # ✅ NEW: Select realistic channel based on amount
            if networking_amount > 100000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': business_partner, 
                'amount': round(networking_amount, 2), 
                'desc': 'Business Networking Transfer',
                'channel': channel,
                'transaction_category': 'business_networking'
            })

    def _handle_operational_expenses(self, date, events):
        """✅ UPDATED: Enhanced operational business expenses with merchant tracking"""
        # Vendor payments with variation
        if date.day == self.vendor_payment_day:
            vendor_variation = random.uniform(0.9, 1.1)
            actual_vendor_payment = self.vendor_payment_amount * vendor_variation
            
            vendor_id = f"primary_vendor_{hash(self.agent_id) % 500}"
            
            txn = self.log_merchant_transaction(
                merchant_id=vendor_id,
                amount=actual_vendor_payment,
                description="Primary Vendor/Supplier Payment",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

        # Business loan EMI with variation
        emi_day = random.randint(8, 12)
        if self.has_loan_emi and date.day == emi_day:
            emi_variation = random.uniform(0.98, 1.02)
            actual_emi = self.business_loan_emi_amount * emi_variation
            
            loan_bank_id = f"business_loan_bank_{hash(self.agent_id) % 300}"
            
            txn = self.log_merchant_transaction(
                merchant_id=loan_bank_id,
                amount=actual_emi,
                description="Business Loan EMI Payment",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)
            
        # Owner's drawings with variation
        drawing_day = random.randint(3, 7)
        if date.day == drawing_day:
            drawing_variation = random.uniform(0.9, 1.2)
            actual_drawing = self.owner_drawing_amount * drawing_variation
            
            txn = self.log_transaction(
                "DEBIT", "Owner's Drawings", actual_drawing, date, channel="Bank_Transfer"
            )
            if txn:
                events.append(txn)

    def _handle_utility_bills(self, date, events):
        """✅ UPDATED: Enhanced commercial utility bills with merchant tracking"""
        if date.day == 25:
            # Enhanced utility calculation with employee factor
            base_utility = random.uniform(2500, 7000)
            employee_factor = self.num_employees * random.uniform(300, 800)
            economic_factor = {
                'Lower_Middle': 0.8,
                'Middle': 1.0,
                'Upper_Middle': 1.3,
                'High': 1.6
            }.get(self.economic_class, 1.0)
            
            total_utility = (base_utility + employee_factor) * economic_factor
            
            utility_provider_id = f"commercial_utility_{hash(self.agent_id) % 100}"
            
            txn = self.log_merchant_transaction(
                merchant_id=utility_provider_id,
                amount=total_utility,
                description="Commercial Utility Bills",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

    def get_small_business_owner_features(self):
        """✅ ENHANCED: Comprehensive small business owner features"""
        return {
            'business_entity_employer_count': len(self.business_entities),
            'primary_business_entity_tenure': self.get_employment_tenure_months(),
            'business_experience_years': self.business_tenure // 12,
            'business_partner_relationships': len(self.business_partners),
            'bank_relationships': len(self.banks),
            'supplier_relationships': len(self.suppliers),
            'customer_relationships': len(self.customer_network),
            'employee_count': self.num_employees,
            'business_network_size': len(self.business_network),
            'cash_flow_variability': len(self.seasonal_down_months) / 12.0,
            'operational_efficiency': 1.0 - self.bad_debt_chance,
            'business_maturity_score': len(self.banks) * 0.2 + len(self.suppliers) * 0.1,
            'income_consistency_score': self.income_consistency,
            'last_revenue_recency': (datetime.now().date() - self.last_revenue_date).days if self.last_revenue_date else 999,
            'seasonal_vulnerability_score': len(self.seasonal_down_months) / 12,
            'total_company_relationships': len(self.business_entities) + len(self.business_partners)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with business entity revenue tracking"""
        events = []
        
        # Handle all income sources (including business entity revenue tracking)
        self._handle_sales_income(date, events)
        
        # Handle business expenses
        self._handle_additional_business_expenses(date, events)
        
        # Handle P2P transfers
        self._handle_employee_salary_payments(date, events, context)
        self._handle_supplier_advance_payments(date, events, context)
        self._handle_customer_refunds(date, events, context)
        self._handle_business_networking_transfers(date, events, context)
        
        # Handle operational expenses
        self._handle_operational_expenses(date, events)
        self._handle_utility_bills(date, events)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
