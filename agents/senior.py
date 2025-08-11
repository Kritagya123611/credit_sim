import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class SeniorCitizen(BaseAgent):
    """
    Enhanced Senior Citizen agent for Phase 2: Pension provider salary source tracking
    Includes pension providers as employers, realistic retirement relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Senior Citizen"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "10000-30000"
        min_income, max_income = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_income * income_multiplier)}-{int(max_income * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name, 
            "risk_profile": risk_profile_category, 
            "risk_score": risk_score,
            "economic_class": economic_class, 
            "financial_personality": financial_personality,
            "employment_status": "Not_Applicable", 
            "employment_verification": "Pensioner_ID_Verified",
            "income_type": "Pension, Rent", 
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date", 
            "savings_retention_rate": "High",
            "has_investment_activity": True, 
            "investment_types": ["FD", "Senior_Citizen_Schemes"],
            "has_loan_emi": False, 
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": True, 
            "insurance_types": ["Health"],
            "utility_payment_status": "ALWAYS_ON_TIME", 
            "mobile_plan_type": "Basic_Postpaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.96, 0.99), 3),
            "ip_consistency_score": round(random.uniform(0.97, 0.99), 3),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Branch"], 
            "login_pattern": "Infrequent",
            "ecommerce_activity_level": "None", 
            "ecommerce_avg_ticket_size": "N/A",
            
            # ✅ NEW: Heterogeneous graph connections specific to SeniorCitizen
            "industry_sector": "Retirement_Pension",
            "company_size": "Not_Applicable",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Pension providers as employers (salary source tracking)
        self.pension_providers = []  # Government pension funds, EPF as company nodes
        self.primary_pension_provider_id = None  # Main pension source
        self.pension_provider_hierarchy = {}  # Track pension types and amounts

        # ✅ NEW: Employment relationship tracking
        self.retirement_tenure = random.randint(1, 25)  # 1-25 years retired
        self.pension_consistency = random.uniform(0.98, 1.0)  # Very high consistency
        self.last_pension_date = None

        # Financial calculations with more variation
        self.pension_day = random.randint(1, 3)  # Beginning of month
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_mod, max_mod)
        
        # Fixed expenses with variation
        self.insurance_percentage = random.uniform(0.13, 0.17)
        self.utility_bill_percentage = random.uniform(0.07, 0.09)
        
        # Regular patterns with variation
        self.weekly_grocery_day = random.randint(3, 5)  # Wed-Fri
        self.monthly_pharmacy_day = random.randint(8, 12)

        # Annual events with variation
        self.large_event_month = random.randint(1, 12)
        self.has_done_large_event_this_year = False
        
        # ✅ Enhanced P2P networks for Senior Citizens
        self.family_members = []  # Adult children and family
        self.grandchildren = []  # Grandchildren for special gifts
        self.children_network = []  # Adult children for regular support
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.pension_providers = []  # Government pension agencies as company nodes
        self.healthcare_providers = []  # Hospitals, clinics as merchant relationships
        self.investment_institutions = []  # Banks, FD providers as merchant nodes
        self.community_organizations = []  # Senior citizen groups, clubs
        self.charitable_organizations = []  # Charity and religious organizations
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.06, 0.10) * personality_config.get('spend_chance_mod', 1.0)
        self.grandchildren_gift_chance = random.uniform(0.10, 0.14)  # Higher chance for grandchildren gifts
        self.emergency_family_support_chance = random.uniform(0.04, 0.06)  # Emergency family support
        self.charitable_donation_chance = random.uniform(0.02, 0.04)  # Charitable donations
        
        # Temporal patterns with more variation
        self.festival_months = [3, 10, 11]  # Holi, Diwali, etc.
        self.birthday_months = random.sample(range(1, 13), k=random.randint(2, 5))  # Family birthdays
        
        # Temporal tracking with enhanced features
        self.last_investment_date = None
        self.healthcare_expense_cycles = []
        self.family_gift_patterns = []
        self.charitable_giving_cycles = []

        self.balance = random.uniform(self.monthly_income * 1.8, self.monthly_income * 6.0)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Senior citizens typically have 1-2 devices (basic phone, sometimes tablet)"""
        device_options = [1, 2]
        weights = [0.6, 0.4]  # Most have just 1 device
        return random.choices(device_options, weights=weights)[0]

    def assign_pension_providers(self, pension_company_ids):
        """✅ NEW: Assign pension providers as employers for salary tracking"""
        self.pension_providers = pension_company_ids
        
        if pension_company_ids:
            # Assign primary pension provider as main employer
            self.primary_pension_provider_id = random.choice(pension_company_ids)
            self.assign_employer(
                company_id=self.primary_pension_provider_id,
                employment_start_date=datetime.now().date() - timedelta(days=self.retirement_tenure * 365)
            )
            
            # Set up pension provider hierarchy
            for provider_id in pension_company_ids:
                self.pension_provider_hierarchy[provider_id] = {
                    'pension_type': random.choice(['Government_Pension', 'EPF', 'Private_Pension', 'Military_Pension']),
                    'pension_amount': random.uniform(0.6, 1.4),  # Multiplier for base amount
                    'payment_reliability': random.uniform(0.98, 1.0)
                }

    def _handle_pension_provider_payment(self, date, events):
        """✅ NEW: Handle monthly pension from pension providers"""
        if (date.day == self.pension_day and 
            random.random() < self.pension_consistency):
            
            provider_id = self.primary_pension_provider_id or (
                random.choice(self.pension_providers) if self.pension_providers else None
            )
            
            # Calculate pension with retirement patterns
            base_pension = self.monthly_income
            
            # Add pension-specific variations
            if provider_id:
                pension_multiplier = self.pension_provider_hierarchy.get(provider_id, {}).get('pension_amount', 1.0)
                base_pension *= pension_multiplier
            
            # Add DA and other pension benefits
            da_percentage = random.uniform(0.15, 0.25)  # 15-25% DA for pensioners
            medical_allowance = random.uniform(0.05, 0.08)  # 5-8% medical allowance
            
            final_pension = base_pension * (1 + da_percentage + medical_allowance)
            
            # ✅ NEW: Log as salary transaction from pension provider
            if provider_id:
                txn = self.log_salary_transaction(
                    amount=final_pension,
                    date=date,
                    company_id=provider_id
                )
                if txn:
                    txn['transaction_category'] = 'pension_provider_payment'
                    txn['company_type'] = 'pension_provider'
                    txn['pension_type'] = self.pension_provider_hierarchy.get(provider_id, {}).get('pension_type', 'Unknown')
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", "Pension/Rent Deposit", final_pension, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)
                    
            self.last_pension_date = date
            return final_pension
        
        return 0

    def _handle_monthly_events(self, date, events):
        """✅ UPDATED: Enhanced monthly events with pension provider tracking"""
        # Pension provider payment
        pension_amount = self._handle_pension_provider_payment(date, events)
        
        # Reset annual flags in January
        if date.month == 1:
            self.has_done_large_event_this_year = False

        # Health insurance premium with variation
        insurance_day = random.randint(4, 7)
        if self.has_insurance_payments and date.day == insurance_day:
            # Add variation to insurance amount
            insurance_variation = random.uniform(0.95, 1.05)
            insurance_amt = (self.monthly_income * self.insurance_percentage) * insurance_variation
            
            insurance_provider_id = f"senior_health_insurance_{hash(self.agent_id) % 300}"
            
            txn = self.log_merchant_transaction(
                merchant_id=insurance_provider_id,
                amount=insurance_amt,
                description="Senior Citizen Health Insurance",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)

        # Monthly pharmacy expenses with enhanced tracking
        if date.day == self.monthly_pharmacy_day:
            # Add variation to pharmacy spending
            pharma_variation = random.uniform(0.8, 1.2)
            pharma_spend = (self.monthly_income * 0.05) * pharma_variation
            
            # ✅ NEW: Enhanced healthcare provider tracking
            pharmacy_id = f"senior_pharmacy_{hash(self.agent_id) % 200}"
            self.add_healthcare_provider(pharmacy_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=pharmacy_id,
                amount=pharma_spend,
                description="Senior Medicines/Healthcare",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

        # Utility bills with enhanced tracking
        utility_day = random.randint(18, 22)
        if date.day == utility_day:
            # Add variation to utility amount
            utility_variation = random.uniform(0.9, 1.1)
            bill_amount = (self.monthly_income * self.utility_bill_percentage) * utility_variation
            
            utility_provider_id = f"senior_utility_{hash(self.agent_id) % 150}"
            
            txn = self.log_merchant_transaction(
                merchant_id=utility_provider_id,
                amount=bill_amount,
                description="Senior Household Utilities",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

    def add_healthcare_provider(self, provider_id, first_visit_date=None):
        """✅ NEW: Track healthcare provider relationships"""
        if provider_id not in self.healthcare_providers:
            self.healthcare_providers.append(provider_id)
            self.add_frequent_merchant(provider_id, first_visit_date)

    def add_investment_institution(self, institution_id, first_investment_date=None):
        """✅ NEW: Track investment institution relationships"""
        if institution_id not in self.investment_institutions:
            self.investment_institutions.append(institution_id)
            self.add_frequent_merchant(institution_id, first_investment_date)

    def _handle_weekly_events(self, date, events):
        """✅ UPDATED: Enhanced weekly spending with merchant tracking"""
        if date.weekday() == self.weekly_grocery_day:
            # Add variation to grocery spending
            grocery_variation = random.uniform(0.85, 1.15)
            grocery_spend = (self.monthly_income * 0.08) * grocery_variation
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                grocery_spend *= random.uniform(1.2, 1.6)
            elif self.economic_class == 'Lower':
                grocery_spend *= random.uniform(0.8, 0.9)
            
            grocery_merchant_id = f"senior_grocery_{hash(self.agent_id) % 150}"
            
            # ✅ NEW: Enhanced grocery merchant tracking
            self.add_frequent_merchant(grocery_merchant_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=grocery_merchant_id,
                amount=grocery_spend,
                description="Senior Weekly Groceries",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def _handle_annual_events(self, date, events):
        """✅ UPDATED: Enhanced annual investment events with institution tracking"""
        if (self.has_investment_activity and 
            date.month == self.large_event_month and 
            date.day == 25 and 
            not self.has_done_large_event_this_year):
            
            fd_amount = self.balance * random.uniform(0.25, 0.55)
            min_threshold = 15000 * (1 if self.economic_class in ['Lower', 'Lower_Middle'] else 2)
            
            if fd_amount > min_threshold:
                # ✅ NEW: Enhanced investment institution tracking
                institution_id = f"senior_fd_bank_{hash(self.agent_id) % 100}"
                self.add_investment_institution(institution_id, date)
                
                txn = self.log_merchant_transaction(
                    merchant_id=institution_id,
                    amount=fd_amount,
                    description="Senior Citizen FD Investment",
                    date=date,
                    channel="Netbanking"
                )
                if txn:
                    events.append(txn)
                    self.has_done_large_event_this_year = True
                    self.last_investment_date = date

    def _handle_regular_family_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced regular family support transfers"""
        if (self.family_members and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 8000):
            
            recipient = random.choice(self.family_members)
            
            # Senior citizens send moderate to higher amounts to family
            base_amount = random.uniform(1500, 6000)
            
            # Economic class adjustments
            economic_multiplier = {
                'Lower': random.uniform(0.5, 0.8),
                'Lower_Middle': random.uniform(0.7, 1.0),
                'Middle': random.uniform(1.0, 1.3),
                'Upper_Middle': random.uniform(1.3, 2.0),
                'High': random.uniform(1.8, 3.0)
            }.get(self.economic_class, 1.0)
            
            final_amount = base_amount * economic_multiplier
            
            # ✅ NEW: Select realistic channel based on amount
            if final_amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])  # Seniors might use traditional channels
            elif final_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(final_amount, 2), 
                'desc': 'Senior Family Support',
                'channel': channel,
                'transaction_category': 'family_support'
            })

    def _handle_grandchildren_gifts(self, date, events, context):
        """✅ UPDATED: Enhanced grandchildren gift transfers"""
        if (self.grandchildren and 
            random.random() < self.grandchildren_gift_chance and
            self.balance > 4000):
            
            grandchild = random.choice(self.grandchildren)
            
            # Grandchildren gifts are typically generous but smaller than family support
            gift_amount = random.uniform(800, 3500)
            
            # Higher amounts during birthdays and festivals
            if date.month in self.birthday_months or date.month in self.festival_months:
                gift_amount *= random.uniform(1.6, 2.8)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                gift_amount *= random.uniform(1.4, 2.5)
            elif self.economic_class == 'Lower':
                gift_amount *= random.uniform(0.6, 0.8)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': grandchild, 
                'amount': round(gift_amount, 2), 
                'desc': 'Grandchildren Gift',
                'channel': channel,
                'transaction_category': 'grandchildren_gift'
            })

    def _handle_festival_and_special_occasions(self, date, events, context):
        """✅ UPDATED: Enhanced festival and special occasion transfers"""
        # Increased P2P activity during festival months
        if (date.month in self.festival_months and 
            date.day <= 5 and
            self.family_members and 
            random.random() < (self.p2p_transfer_chance * 3) and
            self.balance > 12000):
            
            recipient = random.choice(self.family_members)
            festival_amount = self.monthly_income * random.uniform(0.18, 0.35)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                festival_amount *= random.uniform(1.5, 2.5)
            elif self.economic_class == 'Lower':
                festival_amount *= random.uniform(0.7, 0.9)
            
            # ✅ NEW: Select realistic channel based on amount
            if festival_amount > 75000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(festival_amount, 2), 
                'desc': 'Festival Support Transfer',
                'channel': channel,
                'transaction_category': 'festival_support'
            })

    def _handle_emergency_family_support(self, date, events, context):
        """✅ UPDATED: Enhanced emergency family support transfers"""
        if (self.children_network and 
            random.random() < self.emergency_family_support_chance and
            self.balance > 20000):  # Need significant balance for emergency support
            
            child = random.choice(self.children_network)
            
            # Emergency support from seniors can be substantial
            emergency_amount = self.balance * random.uniform(0.25, 0.45)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                emergency_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Select appropriate channel for emergency amounts
            if emergency_amount > 200000:
                channel = random.choice(['NEFT', 'RTGS'])  # Very large emergency amounts
            elif emergency_amount > 100000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': child, 
                'amount': round(emergency_amount, 2), 
                'desc': 'Emergency Family Support',
                'channel': channel,
                'transaction_category': 'emergency_support'
            })

    def _handle_healthcare_expenses(self, date, events, context):
        """✅ UPDATED: Enhanced healthcare-related expenses"""
        if (random.random() < 0.18 and  # 18% chance of additional healthcare expense
            self.balance > 4000):
            
            # Healthcare expenses beyond regular pharmacy
            expense_categories = [
                ("Medical_Consultation", random.uniform(800, 3000)),
                ("Diagnostic_Tests", random.uniform(1500, 5000)),
                ("Physiotherapy", random.uniform(600, 2000)),
                ("Specialist_Treatment", random.uniform(2000, 8000))
            ]
            
            category, expense_amount = random.choice(expense_categories)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                expense_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Enhanced healthcare provider tracking
            healthcare_id = f"senior_healthcare_{category}_{hash(self.agent_id + str(date)) % 300}"
            self.add_healthcare_provider(healthcare_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=healthcare_id,
                amount=expense_amount,
                description=f"Senior {category.replace('_', ' ')}",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def _handle_charitable_donations(self, date, events, context):
        """✅ NEW: Handle charitable and religious donations"""
        if (random.random() < self.charitable_donation_chance and
            self.balance > 10000 and
            date.day in [1, 15]):  # Twice monthly charitable giving
            
            donation_amount = self.monthly_income * random.uniform(0.02, 0.08)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                donation_amount *= random.uniform(1.5, 2.5)
            
            # ✅ NEW: Track charitable organization
            charity_id = f"charitable_org_{hash(self.agent_id) % 100}"
            
            txn = self.log_merchant_transaction(
                merchant_id=charity_id,
                amount=donation_amount,
                description="Senior Charitable Donation",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

    def get_senior_citizen_features(self):
        """✅ ENHANCED: Comprehensive senior citizen features"""
        return {
            'pension_provider_employer_count': len(self.pension_providers),
            'primary_pension_provider_tenure': self.get_employment_tenure_months(),
            'retirement_years': self.retirement_tenure,
            'healthcare_provider_relationships': len(self.healthcare_providers),
            'investment_institution_relationships': len(self.investment_institutions),
            'community_organization_memberships': len(self.community_organizations),
            'family_support_obligations': len(self.family_members),
            'grandchildren_count': len(self.grandchildren),
            'pension_consistency_score': self.pension_consistency,  # Very consistent pension income
            'healthcare_dependency_score': 0.85,  # High healthcare usage
            'family_generosity_score': len(self.family_members) * 0.25,
            'investment_maturity_score': 1.0 if self.has_investment_activity else 0.0,
            'last_pension_recency': (datetime.now().date() - self.last_pension_date).days if self.last_pension_date else 999,
            'charitable_giving_frequency': self.charitable_donation_chance * 30,  # Monthly frequency
            'total_company_relationships': len(self.pension_providers)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with pension provider salary tracking"""
        events = []
        
        # Handle all income sources (including pension provider salary tracking)
        self._handle_monthly_events(date, events)
        
        # Handle regular expenses and activities
        self._handle_weekly_events(date, events)
        self._handle_annual_events(date, events)
        
        # Handle P2P transfers
        self._handle_regular_family_transfers(date, events, context)
        self._handle_grandchildren_gifts(date, events, context)
        self._handle_festival_and_special_occasions(date, events, context)
        self._handle_emergency_family_support(date, events, context)
        
        # Handle healthcare and charitable activities
        self._handle_healthcare_expenses(date, events, context)
        self._handle_charitable_donations(date, events, context)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events, daily_spend_chance=0.1)
        
        return events
