import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class SalariedProfessional(BaseAgent):
    """
    Enhanced Salaried Professional agent for Phase 2: Corporate salary source tracking
    Includes employer companies as salary sources, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Middle', financial_personality='Rational_Investor'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Salaried Professional"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "40000-80000"
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
            "income_type": "Salary",
            "avg_monthly_income_range": modified_income_range, 
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "High" if financial_personality == "Saver" else "Medium",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME", 
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life"], 
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid", 
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.87, 0.98), 3),
            "ip_consistency_score": round(random.uniform(0.82, 0.95), 3), 
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Mobile_Banking"], 
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium", 
            "ecommerce_avg_ticket_size": "Medium",
            
            # ✅ NEW: Heterogeneous graph connections specific to SalariedProfessional
            "industry_sector": "Corporate_Private",
            "company_size": "Medium",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Corporate employers as salary sources (salary source tracking)
        self.corporate_employers = []  # TCS, Infosys, Wipro as company nodes
        self.primary_employer_id = None  # Main employer company
        self.employer_hierarchy = {}  # Track employee levels and departments

        # ✅ NEW: Employment relationship tracking
        self.corporate_tenure = random.randint(12, 180)  # 1-15 years in corporate
        self.salary_consistency = random.uniform(0.95, 1.0)  # High salary consistency
        self.last_salary_date = None
        self.last_bonus_date = None

        # Financial calculations with more variation
        self.salary_day = random.randint(1, 5)
        min_sal_mod, max_sal_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal_mod, max_sal_mod)
        
        # ✅ ENHANCED: More varied percentages based on personality and experience
        self.emi_percentage = random.uniform(0.22, 0.28)
        self.investment_percentage = random.uniform(0.12, 0.18) * personality_config.get('invest_chance_mod', 1.0)
        self.insurance_percentage = random.uniform(0.04, 0.06)
        self.utility_bill_percentage = random.uniform(0.04, 0.06)
        
        # Spending patterns with more variation
        self.ecommerce_spend_chance = random.uniform(0.12, 0.18) * personality_config.get('spend_chance_mod', 1.0)
        self.weekday_spend_chance = random.uniform(0.45, 0.55) * personality_config.get('spend_chance_mod', 1.0)
        self.weekend_spend_chance = random.uniform(0.65, 0.75) * personality_config.get('spend_chance_mod', 1.0)
        
        # Bonus patterns with variation
        self.annual_bonus_month = random.choice([3, 4])  # March or April
        self.has_received_bonus_this_year = False
        
        # ✅ Enhanced P2P networks for Salaried Professionals
        self.dependents = []  # Family dependents for regular allowances
        self.professional_network = []  # Professional colleagues and contacts
        self.social_contacts = []  # Friends and social contacts
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.corporate_employers = []  # Current and previous employers as company nodes
        self.investment_platforms = []  # SIP, mutual fund platforms
        self.insurance_providers = []  # Health, life insurance companies
        self.utility_providers = []  # Electricity, gas, internet providers
        
        # P2P transfer probabilities with more variation
        self.family_support_chance = random.uniform(0.75, 0.85)  # High chance of family support
        self.professional_transfer_chance = random.uniform(0.10, 0.14)  # Professional transfers
        self.social_transfer_chance = random.uniform(0.12, 0.18)  # Social transfers
        self.bonus_sharing_chance = random.uniform(0.20, 0.30)  # Sharing bonus with family
        
        # Temporal tracking with enhanced features
        self.has_shared_bonus_this_year = False
        self.last_investment_date = None
        self.professional_relationship_cycles = []
        
        self.balance = random.uniform(self.salary_amount * 0.15, self.salary_amount * 0.6)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Salaried professionals typically have 2-3 devices (phone, laptop, sometimes tablet)"""
        device_options = [2, 3, 4]
        weights = [0.4, 0.5, 0.1]  # Most have 2-3 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_corporate_employers(self, employer_company_ids):
        """✅ NEW: Assign corporate employers as salary sources for tracking"""
        self.corporate_employers = employer_company_ids
        
        if employer_company_ids:
            # Assign primary employer as main salary source
            self.primary_employer_id = random.choice(employer_company_ids)
            self.assign_employer(
                company_id=self.primary_employer_id,
                employment_start_date=datetime.now().date() - timedelta(days=self.corporate_tenure * 30)
            )
            
            # Set up employer hierarchy and roles
            for employer_id in employer_company_ids:
                self.employer_hierarchy[employer_id] = {
                    'level': random.choice(['Associate', 'Senior_Associate', 'Manager', 'Senior_Manager', 'AVP']),
                    'department': random.choice(['IT', 'Finance', 'HR', 'Operations', 'Sales', 'Marketing']),
                    'employment_type': random.choice(['Full_Time', 'Contract', 'Consultant'])
                }

    def _handle_corporate_salary_payment(self, date, events):
        """✅ NEW: Handle monthly salary from corporate employer"""
        if (date.day == self.salary_day and 
            random.random() < self.salary_consistency):
            
            employer_id = self.primary_employer_id or (
                random.choice(self.corporate_employers) if self.corporate_employers else None
            )
            
            # Calculate salary with corporate patterns
            base_salary = self.salary_amount
            
            # Add experience-based increments
            years_experience = self.corporate_tenure // 12
            increment_multiplier = (1.12 ** min(years_experience, 10))  # 12% yearly increment cap at 10 years
            
            # Add corporate benefits and allowances
            allowance_percentage = random.uniform(0.08, 0.15)  # 8-15% allowances
            variable_pay_percentage = random.uniform(0.05, 0.12)  # 5-12% variable pay
            
            final_salary = base_salary * increment_multiplier * (1 + allowance_percentage + variable_pay_percentage)
            
            # ✅ NEW: Log as salary transaction from corporate employer
            if employer_id:
                txn = self.log_salary_transaction(
                    amount=final_salary,
                    date=date,
                    company_id=employer_id
                )
                if txn:
                    txn['transaction_category'] = 'corporate_salary'
                    txn['company_type'] = 'corporate_employer'
                    txn['employee_level'] = self.employer_hierarchy.get(employer_id, {}).get('level', 'Unknown')
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", "Corporate Salary Deposit", final_salary, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)
            
            self.last_salary_date = date
            return final_salary
        
        return 0

    def _handle_monthly_credits(self, date, events):
        """✅ UPDATED: Enhanced salary and bonus handling with company tracking"""
        # Corporate salary payment
        salary_amount = self._handle_corporate_salary_payment(date, events)
        
        # Reset annual flags in January
        if date.month == 1: 
            self.has_received_bonus_this_year = False
            self.has_shared_bonus_this_year = False
                
        # Annual bonus
        if (date.month == self.annual_bonus_month and 
            date.day == self.salary_day and 
            not self.has_received_bonus_this_year):
            
            bonus_amount = self.salary_amount * random.uniform(1.8, 4.0)  # Corporate bonuses
            
            # ✅ NEW: Enhanced bonus tracking from corporate employer
            if self.corporate_employers:
                employer_id = random.choice(self.corporate_employers)
                txn = self.log_salary_transaction(
                    amount=bonus_amount,
                    date=date,
                    company_id=employer_id
                )
                if txn:
                    txn['transaction_category'] = 'corporate_annual_bonus'
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", "Annual Performance Bonus", bonus_amount, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)
            
            self.has_received_bonus_this_year = True
            self.last_bonus_date = date

    def add_investment_platform(self, platform_id, first_investment_date=None):
        """✅ NEW: Track investment platform relationships"""
        if platform_id not in self.investment_platforms:
            self.investment_platforms.append(platform_id)
            self.add_frequent_merchant(platform_id, first_investment_date)

    def _handle_family_support_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced family support transfers with realistic channels"""
        if (date.day == self.salary_day and 
            self.dependents and 
            random.random() < self.family_support_chance):
            
            recipient = self.dependents[0]  # Primary dependent
            allowance = getattr(recipient, 'monthly_allowance', self.salary_amount * random.uniform(0.18, 0.25))
            
            # ✅ NEW: Select realistic channel based on amount
            if allowance > 100000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif allowance > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(allowance, 2), 
                'desc': 'Corporate Professional Family Allowance',
                'channel': channel,
                'transaction_category': 'family_allowance'
            })

    def _handle_professional_network_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced professional network transfers"""
        if (self.professional_network and 
            random.random() < self.professional_transfer_chance and
            self.balance > 8000):
            
            colleague = random.choice(self.professional_network)
            
            # Professional transfers (shared expenses, office collections, etc.)
            transfer_amount = random.uniform(1500, 5000)
            
            # Economic class adjustments
            economic_multiplier = {
                'Lower_Middle': random.uniform(0.8, 1.1),
                'Middle': random.uniform(1.0, 1.3),
                'Upper_Middle': random.uniform(1.2, 1.8),
                'High': random.uniform(1.5, 2.2)
            }.get(self.economic_class, 1.0)
            
            # Experience-based adjustments
            if self.corporate_tenure > 60:  # 5+ years experience
                transfer_amount *= random.uniform(1.1, 1.4)
            
            final_amount = transfer_amount * economic_multiplier
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': colleague, 
                'amount': round(final_amount, 2), 
                'desc': 'Corporate Professional Transfer',
                'channel': channel,
                'transaction_category': 'professional_transfer'
            })

    def _handle_social_network_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced social network transfers"""
        if (self.social_contacts and 
            random.random() < self.social_transfer_chance and
            self.balance > 4000):
            
            friend = random.choice(self.social_contacts)
            
            # Social transfers (shared outings, gifts, mutual support)
            social_amount = random.uniform(600, 3500)
            
            # Higher amounts on weekends (more social activities)
            if date.weekday() >= 5:  # Weekend
                social_amount *= random.uniform(1.3, 1.9)
            
            # Adjust based on financial personality
            if self.financial_personality == 'Over_Spender':
                social_amount *= random.uniform(1.2, 1.6)
            elif self.financial_personality == 'Saver':
                social_amount *= random.uniform(0.8, 1.0)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                social_amount *= random.uniform(1.2, 1.7)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': friend, 
                'amount': round(social_amount, 2), 
                'desc': 'Social Network Transfer',
                'channel': channel,
                'transaction_category': 'social_transfer'
            })

    def _handle_bonus_sharing(self, date, events, context):
        """✅ UPDATED: Enhanced bonus sharing with family"""
        if (self.has_received_bonus_this_year and 
            not self.has_shared_bonus_this_year and
            date.month == self.annual_bonus_month and
            date.day >= self.salary_day + 3 and  # Few days after bonus
            random.random() < self.bonus_sharing_chance):
            
            # Share bonus with dependents or family
            recipients = []
            if self.dependents:
                recipients.extend(self.dependents[:2])  # Max 2 dependents
            
            if recipients:
                for recipient in recipients:
                    # Bonus sharing is typically generous for salaried professionals
                    bonus_share = self.salary_amount * random.uniform(0.4, 1.0)
                    
                    # Economic class adjustments
                    if self.economic_class in ['Upper_Middle', 'High']:
                        bonus_share *= random.uniform(1.3, 2.0)
                    
                    # ✅ NEW: Select appropriate channel for larger bonus shares
                    if bonus_share > 100000:
                        channel = random.choice(['NEFT', 'RTGS'])
                    elif bonus_share > 50000:
                        channel = random.choice(['IMPS', 'NEFT'])
                    else:
                        channel = RealisticP2PStructure.select_realistic_channel()
                    
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': recipient, 
                        'amount': round(bonus_share, 2), 
                        'desc': 'Corporate Bonus Sharing',
                        'channel': channel,
                        'transaction_category': 'bonus_sharing'
                    })
                
                self.has_shared_bonus_this_year = True

    def _handle_recurring_debits(self, date, events):
        """✅ UPDATED: Enhanced recurring payments with merchant tracking"""
        # Loan EMI with variation
        emi_day = random.randint(8, 12)
        if self.has_loan_emi and date.day == emi_day:
            # Add slight variation to EMI amount
            emi_variation = random.uniform(0.98, 1.02)
            emi_amount = (self.salary_amount * self.emi_percentage) * emi_variation
            
            loan_provider_id = f"corporate_loan_bank_{hash(self.agent_id) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=loan_provider_id,
                amount=emi_amount,
                description="Corporate Professional Home Loan EMI",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)
            
        # Insurance premium with enhanced tracking
        insurance_day = random.randint(13, 17)
        if self.has_insurance_payments and date.day == insurance_day:
            # Add variation to insurance amount
            insurance_variation = random.uniform(0.95, 1.05)
            insurance_total = (self.salary_amount * self.insurance_percentage) * insurance_variation
            
            insurance_provider_id = f"corporate_insurance_{hash(self.agent_id) % 300}"
            
            txn = self.log_merchant_transaction(
                merchant_id=insurance_provider_id,
                amount=insurance_total,
                description="Corporate Health/Life Insurance",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)
            
        # Investment SIP with enhanced tracking
        investment_day = random.randint(18, 22)
        if self.has_investment_activity and date.day == investment_day:
            # Add variation to investment amount
            investment_variation = random.uniform(0.9, 1.1)
            invest_amt = (self.salary_amount * self.investment_percentage) * investment_variation
            
            # ✅ NEW: Enhanced investment platform tracking
            investment_platform_id = f"corporate_sip_platform_{hash(self.agent_id) % 200}"
            self.add_investment_platform(investment_platform_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=investment_platform_id,
                amount=invest_amt,
                description="Corporate Professional SIP Investment",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)
                self.last_investment_date = date
            
        # Utility bills with enhanced tracking
        utility_day = random.randint(23, 27)
        if date.day == utility_day:
            # Add variation to utility amount
            utility_variation = random.uniform(0.9, 1.1)
            bill_amount = (self.salary_amount * self.utility_bill_percentage) * utility_variation
            
            utility_provider_id = f"corporate_utility_{hash(self.agent_id) % 200}"
            
            txn = self.log_merchant_transaction(
                merchant_id=utility_provider_id,
                amount=bill_amount,
                description="Professional Household Utility Bills",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

    def _handle_daily_spending(self, date, events):
        """✅ UPDATED: Enhanced daily spending patterns with merchant tracking"""
        # E-commerce spending (enhanced after bonus)
        ecommerce_chance = self.ecommerce_spend_chance * (2.2 if self.has_received_bonus_this_year else 1)
        if random.random() < ecommerce_chance:
            ecommerce_categories = [
                ("Electronics", random.uniform(2000, 8000)),
                ("Fashion", random.uniform(1000, 4000)),
                ("Home_Goods", random.uniform(1500, 6000)),
                ("Books_Education", random.uniform(500, 2000))
            ]
            
            category, ecommerce_amt = random.choice(ecommerce_categories)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                ecommerce_amt *= random.uniform(1.3, 2.0)
            
            ecommerce_merchant_id = f"ecommerce_{category}_{hash(self.agent_id + str(date)) % 500}"
            
            txn = self.log_merchant_transaction(
                merchant_id=ecommerce_merchant_id,
                amount=ecommerce_amt,
                description=f"E-commerce {category.replace('_', ' ')}",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)
            
        # Weekend spending
        is_weekend = date.weekday() >= 5
        if is_weekend:
            if random.random() < self.weekend_spend_chance:
                weekend_categories = [
                    ("Entertainment", random.uniform(800, 3000)),
                    ("Fine_Dining", random.uniform(1200, 4000)),
                    ("Shopping", random.uniform(1000, 3500))
                ]
                
                category, spend_amount = random.choice(weekend_categories)
                
                # Economic class adjustments
                if self.economic_class in ['Upper_Middle', 'High']:
                    spend_amount *= random.uniform(1.4, 2.2)
                
                entertainment_merchant_id = f"weekend_{category}_{hash(self.agent_id + str(date)) % 300}"
                
                txn = self.log_merchant_transaction(
                    merchant_id=entertainment_merchant_id,
                    amount=spend_amount,
                    description=f"Weekend {category.replace('_', ' ')}",
                    date=date,
                    channel="Card"
                )
                if txn:
                    events.append(txn)
        else:
            # Weekday spending
            if random.random() < self.weekday_spend_chance:
                weekday_categories = [
                    ("Transport", random.uniform(200, 600)),
                    ("Groceries", random.uniform(400, 1200)),
                    ("Lunch", random.uniform(150, 500)),
                    ("Coffee_Snacks", random.uniform(100, 300))
                ]
                
                spend_type, spend_amount = random.choice(weekday_categories)
                
                # Economic class adjustments
                if self.economic_class in ['Upper_Middle', 'High']:
                    spend_amount *= random.uniform(1.2, 1.6)
                
                weekday_merchant_id = f"weekday_{spend_type}_{hash(self.agent_id + str(date)) % 400}"
                
                txn = self.log_merchant_transaction(
                    merchant_id=weekday_merchant_id,
                    amount=spend_amount,
                    description=f"Weekday {spend_type.replace('_', ' ')}",
                    date=date,
                    channel="UPI"
                )
                if txn:
                    events.append(txn)

    def get_salaried_professional_features(self):
        """✅ ENHANCED: Comprehensive salaried professional features"""
        return {
            'corporate_employer_count': len(self.corporate_employers),
            'primary_employer_tenure': self.get_employment_tenure_months(),
            'corporate_experience_years': self.corporate_tenure // 12,
            'investment_platform_relationships': len(self.investment_platforms),
            'professional_network_size': len(self.professional_network),
            'social_network_size': len(self.social_contacts),
            'dependents_count': len(self.dependents),
            'salary_consistency_score': self.salary_consistency,  # Very consistent salary
            'bonus_pattern_regularity': 1.0 if self.annual_bonus_month else 0.0,
            'investment_discipline_score': 1.0 if self.has_investment_activity else 0.0,
            'last_salary_recency': (datetime.now().date() - self.last_salary_date).days if self.last_salary_date else 999,
            'last_bonus_recency': (datetime.now().date() - self.last_bonus_date).days if self.last_bonus_date else 999,
            'professional_payment_reliability': 1.0,  # Corporate professionals are very reliable
            'total_company_relationships': len(self.corporate_employers)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with corporate salary tracking"""
        events = []
        
        # Handle all income sources (including corporate salary tracking)
        self._handle_monthly_credits(date, events)
        
        # Handle P2P transfers
        self._handle_family_support_transfers(date, events, context)
        self._handle_professional_network_transfers(date, events, context)
        self._handle_social_network_transfers(date, events, context)
        self._handle_bonus_sharing(date, events, context)
        
        # Handle recurring expenses
        self._handle_recurring_debits(date, events)
        
        # Handle daily spending
        self._handle_daily_spending(date, events)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
