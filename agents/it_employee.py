import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class TechProfessional(BaseAgent):
    """
    Enhanced Tech Professional agent for Phase 2: Tech company salary source tracking
    Includes tech company employers, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Upper_Middle', financial_personality='Rational_Investor'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Tech Professional"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "60000-200000"
        min_sal, max_sal = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_sal * income_multiplier)}-{int(max_sal * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
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
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.89, 0.98), 3),
            "ip_consistency_score": round(random.uniform(0.55, 0.80), 3),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["All"],
            "login_pattern": "Geographically_Dynamic",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
            
            # ✅ NEW: Heterogeneous graph connections specific to TechProfessional
            "industry_sector": "Technology_IT",
            "company_size": "Large",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Tech companies as employers (salary source tracking)
        self.tech_companies = []  # Google, Microsoft, Amazon as company nodes
        self.primary_tech_company_id = None  # Main employer company
        self.tech_company_hierarchy = {}  # Track company roles and levels

        # ✅ NEW: Employment relationship tracking
        self.tech_industry_tenure = random.randint(12, 180)  # 1-15 years in tech
        self.salary_consistency = random.uniform(0.95, 1.0)  # High salary consistency
        self.last_salary_date = None
        self.last_bonus_date = None

        # Financial calculations with more variation
        self.salary_day = random.randint(1, 5)
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_mod, max_mod)
        
        # ✅ ENHANCED: More varied investment allocations
        self.stock_investment_amount = self.salary_amount * random.uniform(0.12, 0.28) * personality_config.get('invest_chance_mod', 1.0)
        crypto_multiplier = 2.5 if financial_personality == 'Risk_Addict' else 1.0
        self.crypto_investment_amount = self.salary_amount * random.uniform(0.03, 0.12) * crypto_multiplier
        
        # Fixed expenses with variation
        self.loan_emi_amount = self.salary_amount * random.uniform(0.15, 0.25)
        self.saas_subscription_amount = random.uniform(400, 2500)

        # Travel patterns with enhanced tracking
        self.is_traveling = False
        self.travel_start_day = 0
        self.travel_duration = 0
        
        # Bonus tracking with variation
        self.annual_bonus_month = random.choice([3, 4])
        self.has_received_bonus_this_year = False
        self.has_shared_bonus_this_year = False

        # ✅ Enhanced P2P networks for Tech Professionals
        self.contacts = []  # General tech network
        self.professional_network = []  # Tech colleagues, freelancers, etc.
        self.family_dependents = []  # Family members they support
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.tech_companies = []  # Current and previous employers as company nodes
        self.investment_platforms = []  # Zerodha, WazirX as merchant relationships
        self.saas_providers = []  # Cloud, VPN, software subscriptions
        self.travel_merchants = []  # Booking platforms, airlines
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.15, 0.21) * personality_config.get('spend_chance_mod', 1.0)
        self.professional_transfer_chance = random.uniform(0.10, 0.14)
        self.family_support_chance = random.uniform(0.08, 0.12)
        
        # Special occasions with variation
        self.bonus_sharing_chance = random.uniform(0.20, 0.30)

        # Temporal tracking with enhanced features
        self.last_bonus_date = None
        self.investment_cycles = []  # Track investment patterns
        self.travel_patterns = []  # Track travel behavior

        self.balance = random.uniform(self.salary_amount * 0.4, self.salary_amount * 1.2)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Tech professionals typically have 3-5 devices (phone, laptop, tablet, work devices)"""
        device_options = [3, 4, 5, 6]
        weights = [0.3, 0.4, 0.2, 0.1]  # Most have 3-4 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_tech_companies(self, tech_company_ids):
        """✅ NEW: Assign tech companies as employers for salary tracking"""
        self.tech_companies = tech_company_ids
        
        if tech_company_ids:
            # Assign primary tech company as main employer
            self.primary_tech_company_id = random.choice(tech_company_ids)
            self.assign_employer(
                company_id=self.primary_tech_company_id,
                employment_start_date=datetime.now().date() - timedelta(days=self.tech_industry_tenure * 30)
            )
            
            # Set up company hierarchy and roles
            for company_id in tech_company_ids:
                self.tech_company_hierarchy[company_id] = {
                    'level': random.choice(['SDE1', 'SDE2', 'SDE3', 'Senior', 'Principal', 'Staff']),
                    'team': random.choice(['Backend', 'Frontend', 'ML', 'DevOps', 'Product']),
                    'stock_options': random.choice([True, False])
                }

    def _handle_tech_company_salary_payment(self, date, events):
        """✅ NEW: Handle monthly salary from tech company"""
        if (date.day == self.salary_day and 
            random.random() < self.salary_consistency):
            
            company_id = self.primary_tech_company_id or (
                random.choice(self.tech_companies) if self.tech_companies else None
            )
            
            # Calculate salary with tech industry patterns
            base_salary = self.salary_amount
            
            # Add performance-based increments
            years_experience = self.tech_industry_tenure // 12
            increment_multiplier = (1.15 ** min(years_experience, 8))  # 15% yearly increment cap at 8 years
            
            # Add stock compensation and tech benefits
            stock_comp_percentage = random.uniform(0.10, 0.25)  # 10-25% stock compensation
            tech_benefits_percentage = random.uniform(0.05, 0.12)  # 5-12% benefits
            
            final_salary = base_salary * increment_multiplier * (1 + stock_comp_percentage + tech_benefits_percentage)
            
            # ✅ NEW: Log as salary transaction from tech company
            if company_id:
                txn = self.log_salary_transaction(
                    amount=final_salary,
                    date=date,
                    company_id=company_id
                )
                if txn:
                    txn['transaction_category'] = 'tech_company_salary'
                    txn['company_type'] = 'technology_company'
                    txn['employee_level'] = self.tech_company_hierarchy.get(company_id, {}).get('level', 'Unknown')
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", "IT Company Salary", final_salary, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)
            
            self.last_salary_date = date
            return final_salary
        
        return 0

    def _handle_income(self, date, events):
        """✅ UPDATED: Enhanced income handling with company salary tracking"""
        # Tech company salary payments
        salary_amount = self._handle_tech_company_salary_payment(date, events)
        
        # Annual bonus (RSU vesting, performance bonus)
        if (date.month == self.annual_bonus_month and 
            date.day == self.salary_day and 
            not self.has_received_bonus_this_year):
            
            bonus_amount = self.salary_amount * random.uniform(2.5, 8.0)  # Tech bonuses are substantial
            
            # ✅ NEW: Enhanced bonus tracking from tech company
            if self.tech_companies:
                company_id = random.choice(self.tech_companies)
                txn = self.log_salary_transaction(
                    amount=bonus_amount,
                    date=date,
                    company_id=company_id
                )
                if txn:
                    txn['transaction_category'] = 'tech_annual_bonus_rsu'
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", "Annual Bonus/RSU Vesting", bonus_amount, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)
            
            self.has_received_bonus_this_year = True
            self.last_bonus_date = date
        
        # Reset annual flags
        if date.month == 1:
            self.has_received_bonus_this_year = False
            self.has_shared_bonus_this_year = False

    def _handle_fixed_debits(self, date, events):
        """✅ UPDATED: Enhanced recurring payments with merchant tracking"""
        # Loan EMI with variation
        emi_day = random.randint(8, 12)
        if self.has_loan_emi and date.day == emi_day:
            # Add slight variation to EMI amount
            emi_variation = random.uniform(0.98, 1.02)
            actual_emi = self.loan_emi_amount * emi_variation
            
            loan_provider_id = f"tech_professional_loan_{hash(self.agent_id) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=loan_provider_id,
                amount=actual_emi,
                description="Tech Professional Home Loan EMI",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)
            
        # Investment handling with enhanced tracking
        if self.has_investment_activity:
            # Stock investments
            stock_investment_day = random.randint(3, 7)
            if "Stocks" in self.investment_types and date.day == stock_investment_day:
                # ✅ NEW: Enhanced investment platform tracking
                platform_id = f"stock_platform_zerodha_{hash(self.agent_id) % 500}"
                self.add_investment_platform(platform_id, date)
                
                # Add variation to investment amount
                investment_variation = random.uniform(0.8, 1.3)
                actual_investment = self.stock_investment_amount * investment_variation
                
                txn = self.log_merchant_transaction(
                    merchant_id=platform_id,
                    amount=actual_investment,
                    description="Tech Stock Portfolio Investment",
                    date=date,
                    channel="Netbanking"
                )
                if txn:
                    events.append(txn)
                    
            # Crypto investments
            crypto_investment_day = random.randint(13, 17)
            if "Crypto" in self.investment_types and date.day == crypto_investment_day:
                # ✅ NEW: Enhanced crypto platform tracking
                crypto_platform_id = f"crypto_platform_wazirx_{hash(self.agent_id) % 300}"
                self.add_investment_platform(crypto_platform_id, date)
                
                # Add variation to crypto investment
                crypto_variation = random.uniform(0.7, 1.5)
                actual_crypto = self.crypto_investment_amount * crypto_variation
                
                txn = self.log_merchant_transaction(
                    merchant_id=crypto_platform_id,
                    amount=actual_crypto,
                    description="Cryptocurrency Investment",
                    date=date,
                    channel="Netbanking"
                )
                if txn:
                    events.append(txn)

        # SaaS subscriptions with enhanced tracking
        saas_day = random.randint(18, 22)
        if date.day == saas_day:
            # ✅ NEW: Enhanced SaaS provider tracking
            saas_provider_id = f"tech_saas_provider_{hash(self.agent_id) % 200}"
            self.add_saas_provider(saas_provider_id, date)
            
            # Add variation to subscription cost
            saas_variation = random.uniform(0.9, 1.2)
            actual_saas = self.saas_subscription_amount * saas_variation
            
            txn = self.log_merchant_transaction(
                merchant_id=saas_provider_id,
                amount=actual_saas,
                description="Professional SaaS Subscriptions",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def add_investment_platform(self, platform_id, first_investment_date=None):
        """✅ NEW: Track investment platform relationships"""
        if platform_id not in self.investment_platforms:
            self.investment_platforms.append(platform_id)
            self.add_frequent_merchant(platform_id, first_investment_date)

    def add_saas_provider(self, provider_id, subscription_start_date=None):
        """✅ NEW: Track SaaS subscription relationships"""
        if provider_id not in self.saas_providers:
            self.saas_providers.append(provider_id)
            self.add_frequent_merchant(provider_id, subscription_start_date)

    def _handle_dynamic_spending(self, date, events):
        """✅ UPDATED: Enhanced travel and lifestyle spending with merchant tracking"""
        # Quarterly travel planning
        if date.day == 1 and date.month in [1, 4, 7, 10]:
            if random.random() < 0.6:  # 60% chance of quarterly travel
                self.is_traveling = True
                self.travel_start_day = random.randint(5, 20)
                self.travel_duration = random.randint(7, 21)  # Extended travel for tech workers
                
                # Enhanced travel cost calculation
                base_cost = random.uniform(25000, 100000)
                economic_multiplier = {
                    'Upper_Middle': random.uniform(1.2, 1.8),
                    'High': random.uniform(1.8, 3.0)
                }.get(self.economic_class, 1.0)
                
                travel_cost = base_cost * economic_multiplier
                
                # ✅ NEW: Enhanced travel merchant tracking
                travel_merchant_id = f"travel_booking_{hash(self.agent_id + str(date)) % 1000}"
                self.add_frequent_merchant(travel_merchant_id, date)
                
                txn = self.log_merchant_transaction(
                    merchant_id=travel_merchant_id,
                    amount=travel_cost,
                    description="International Travel Booking",
                    date=date,
                    channel="Card"
                )
                if txn:
                    events.append(txn)
        
        # Travel expenses during travel period
        if self.is_traveling:
            if (date.day >= self.travel_start_day and 
                date.day < self.travel_start_day + self.travel_duration):
                
                if random.random() < 0.8:  # 80% chance of daily travel expenses
                    spend_amount = random.uniform(1500, 8000)
                    travel_expense_merchant_id = f"travel_expense_{hash(self.agent_id) % 500}"
                    
                    txn = self.log_merchant_transaction(
                        merchant_id=travel_expense_merchant_id,
                        amount=spend_amount,
                        description="International Travel Expenses",
                        date=date,
                        channel="Card"
                    )
                    if txn:
                        events.append(txn)
            else:
                self.is_traveling = False
        else:
            # Regular lifestyle spending
            if random.random() < 0.35:  # 35% chance of lifestyle spending
                spend_categories = [
                    ("Tech_Gadgets", random.uniform(5000, 25000)),
                    ("Fine_Dining", random.uniform(2000, 8000)),
                    ("Premium_E_commerce", random.uniform(3000, 15000)),
                    ("Entertainment", random.uniform(1000, 5000))
                ]
                
                category, spend_amount = random.choice(spend_categories)
                
                # Economic class adjustments
                if self.economic_class == 'High':
                    spend_amount *= random.uniform(1.5, 2.5)
                
                lifestyle_merchant_id = f"lifestyle_{category}_{hash(self.agent_id + str(date)) % 500}"
                
                txn = self.log_merchant_transaction(
                    merchant_id=lifestyle_merchant_id,
                    amount=spend_amount,
                    description=f"Premium {category.replace('_', ' ')}",
                    date=date,
                    channel="Card"
                )
                if txn:
                    events.append(txn)

    def _handle_social_p2p_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced social transfers with realistic channels"""
        if (self.contacts and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 5000):
            
            recipient = random.choice(self.contacts)
            
            # Tech professionals typically send higher amounts
            base_amount = random.uniform(1500, 8000)
            
            # Economic class adjustments
            economic_multiplier = {
                'Lower_Middle': random.uniform(0.8, 1.2),
                'Middle': random.uniform(1.0, 1.4),
                'Upper_Middle': random.uniform(1.3, 2.0),
                'High': random.uniform(1.8, 3.0)
            }.get(self.economic_class, 1.0)
            
            final_amount = base_amount * economic_multiplier
            
            # Personality adjustments
            if self.financial_personality == 'Over_Spender':
                final_amount *= random.uniform(1.2, 1.6)
            elif self.financial_personality == 'Saver':
                final_amount *= random.uniform(0.8, 1.0)
            
            # ✅ NEW: Select realistic channel based on amount
            if final_amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif final_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(final_amount, 2), 
                'desc': 'Tech Professional Social Transfer',
                'channel': channel,
                'transaction_category': 'social_transfer'
            })

    def _handle_professional_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced professional transfers"""
        if (self.professional_network and 
            random.random() < self.professional_transfer_chance and
            self.balance > 10000):
            
            recipient = random.choice(self.professional_network)
            
            # Professional transfer amounts are typically higher for tech workers
            amount = random.uniform(3000, 15000)
            
            # Higher amounts for senior tech professionals
            if self.tech_industry_tenure > 60:  # 5+ years experience
                amount *= random.uniform(1.3, 2.0)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                amount *= random.uniform(1.4, 2.5)
            
            # ✅ NEW: Select realistic channel based on amount
            if amount > 200000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif amount > 100000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(amount, 2), 
                'desc': 'Tech Professional Network Transfer',
                'channel': channel,
                'transaction_category': 'professional_transfer'
            })

    def _handle_family_support(self, date, events, context):
        """✅ UPDATED: Enhanced family support transfers"""
        if (self.family_dependents and 
            random.random() < self.family_support_chance and
            self.balance > 15000):
            
            recipient = random.choice(self.family_dependents)
            
            # Family support amounts based on tech salary
            support_amount = self.salary_amount * random.uniform(0.06, 0.18)
            
            # Adjust based on economic class and experience
            if self.economic_class in ['High', 'Upper_Middle']:
                support_amount *= random.uniform(1.2, 2.0)
                
            if self.tech_industry_tenure > 60:  # Senior professionals support more
                support_amount *= random.uniform(1.1, 1.5)
            
            # ✅ NEW: Select realistic channel based on amount
            if support_amount > 100000:
                channel = random.choice(['IMPS', 'NEFT'])
            elif support_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(support_amount, 2), 
                'desc': 'Tech Professional Family Support',
                'channel': channel,
                'transaction_category': 'family_support'
            })

    def _handle_bonus_sharing(self, date, events, context):
        """✅ UPDATED: Enhanced bonus sharing with realistic channels"""
        if (self.has_received_bonus_this_year and 
            not self.has_shared_bonus_this_year and
            date.month == self.annual_bonus_month and
            date.day >= self.salary_day + 5 and  # Few days after bonus
            random.random() < self.bonus_sharing_chance):
            
            # Share bonus with family or close contacts
            recipients = []
            if self.family_dependents:
                recipients.extend(self.family_dependents[:3])  # Max 3 family members
            if self.contacts and len(recipients) < 3:
                recipients.extend(random.sample(self.contacts, min(2, len(self.contacts))))
            
            if recipients:
                for recipient in recipients[:3]:  # Limit to 3 recipients
                    # Tech bonus sharing is typically very generous
                    bonus_share = self.salary_amount * random.uniform(0.4, 1.2)
                    
                    # Economic class adjustments
                    if self.economic_class == 'High':
                        bonus_share *= random.uniform(1.5, 2.5)
                    
                    # ✅ NEW: Select appropriate channel for large bonus shares
                    if bonus_share > 200000:
                        channel = random.choice(['NEFT', 'RTGS'])
                    elif bonus_share > 100000:
                        channel = random.choice(['IMPS', 'NEFT'])
                    else:
                        channel = RealisticP2PStructure.select_realistic_channel()
                    
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': recipient, 
                        'amount': round(bonus_share, 2), 
                        'desc': 'Tech Bonus Sharing',
                        'channel': channel,
                        'transaction_category': 'bonus_sharing'
                    })
                
                self.has_shared_bonus_this_year = True

    def get_tech_professional_features(self):
        """✅ ENHANCED: Comprehensive tech professional features"""
        return {
            'tech_company_employer_count': len(self.tech_companies),
            'primary_tech_company_tenure': self.get_employment_tenure_months(),
            'tech_industry_experience_years': self.tech_industry_tenure // 12,
            'investment_platform_relationships': len(self.investment_platforms),
            'saas_provider_relationships': len(self.saas_providers),
            'professional_network_size': len(self.professional_network),
            'family_support_obligations': len(self.family_dependents),
            'bonus_pattern_consistency': 1.0 if self.annual_bonus_month else 0.0,
            'investment_diversity_score': len(self.investment_types),
            'travel_frequency_score': 0.6,  # Quarterly travel pattern
            'salary_consistency_score': self.salary_consistency,
            'last_salary_recency': (datetime.now().date() - self.last_salary_date).days if self.last_salary_date else 999,
            'last_bonus_recency': (datetime.now().date() - self.last_bonus_date).days if self.last_bonus_date else 999,
            'total_company_relationships': len(self.tech_companies)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with tech company salary tracking"""
        events = []
        
        # Handle all income sources (including tech company salary tracking)
        self._handle_income(date, events)
        
        # Handle fixed expenses and investments
        self._handle_fixed_debits(date, events)
        
        # Handle lifestyle and travel spending
        self._handle_dynamic_spending(date, events)
        
        # Handle P2P transfers
        self._handle_social_p2p_transfers(date, events, context)
        self._handle_professional_transfers(date, events, context)
        self._handle_family_support(date, events, context)
        self._handle_bonus_sharing(date, events, context)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
