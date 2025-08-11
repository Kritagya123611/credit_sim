import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class ContentCreator(BaseAgent):
    """
    Enhanced Content Creator agent for Phase 2: Company salary source tracking
    Includes platform companies as employers, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Risk_Addict'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Content Creator"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "20000-100000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-Employed",
            "employment_verification": "ITR_Inconsistent",
            "income_type": "Sponsorships, Platform_Payouts",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Erratic_High_Variance",
            "savings_retention_rate": "Low",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "Occasionally_Late",
            "mobile_plan_type": "High-Value_Postpaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.35, 0.85), 3),
            "ip_consistency_score": round(random.uniform(0.25, 0.75), 3),
            "sim_churn_rate": "High",
            "primary_digital_channels": ["All"],
            "login_pattern": "Geographically_Dynamic",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
            
            # ✅ NEW: Heterogeneous graph connections specific to ContentCreator
            "industry_sector": "Media_Entertainment",
            "company_size": "Freelancer",  # Self-employed
        }
        
        super().__init__(**profile_attributes)

        # Financial calculations
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_income = random.uniform(min_mod, max_mod)

        # ✅ NEW: Platform companies as employers (salary source tracking)
        self.platform_companies = []  # YouTube, Instagram, TikTok as company nodes
        self.primary_platform_id = None  # Main income source platform
        self.platform_payout_schedule = {}  # Track payout schedules per platform

        # Income sources with enhanced tracking
        self.platform_payout_chance = 0.25 * personality_config.get('invest_chance_mod', 1.0)
        self.sponsorship_chance = 0.03
        self.has_sponsorship_funds = False

        # ✅ NEW: Employment relationship tracking
        self.monthly_platform_payout_day = random.randint(25, 30)  # AdSense payout day
        self.last_platform_payout_date = None
        self.platform_income_consistency = random.uniform(0.6, 0.9)  # Payment regularity

        # Fixed expenses with more variation
        self.loan_emi_amount = self.avg_monthly_income * random.uniform(0.15, 0.25)
        self.software_subscription = random.uniform(1500, 4500)  # More varied
        self.utility_bill_day = random.randint(18, 28)  # More spread
        self.late_payment_chance = random.uniform(0.2, 0.4)  # Personality-based
        self.spend_chance_mod = personality_config.get('spend_chance_mod', 1.0)

        # ✅ Enhanced P2P networks for Content Creators
        self.collaborators = []  # Co-creators for joint projects
        self.creator_network = []  # Other content creators  
        self.brand_contacts = []  # Brand representatives
        self.freelancer_network = []  # Editors, designers, etc.
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.platform_companies = []  # YouTube, Instagram as company nodes
        self.equipment_merchants = []  # Camera stores, electronics
        self.service_merchants = []  # Editing services, graphic design
        self.brand_companies = []  # Sponsor companies as employer nodes
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.18, 0.26) * personality_config.get('spend_chance_mod', 1.0)
        self.collaboration_payment_chance = random.uniform(0.15, 0.21)
        self.freelancer_payment_chance = random.uniform(0.12, 0.18)
        self.creator_support_chance = random.uniform(0.06, 0.10)
        self.brand_advance_chance = random.uniform(0.03, 0.07)

        # Temporal tracking with enhanced cycles
        self.last_collaboration_payment_date = None
        self.monthly_freelancer_payment_day = random.randint(3, 12)  # More spread
        self.content_creation_cycles = []  # Track content production cycles
        self.seasonal_income_patterns = {}  # Track seasonal variations

        self.balance = random.uniform(3000, 25000)  # More realistic range

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Content creators typically have more devices"""
        device_options = [2, 3, 4, 5]
        weights = [0.2, 0.4, 0.3, 0.1]  # Most have 3-4 devices (phone, laptop, camera, tablet)
        return random.choices(device_options, weights=weights)[0]

    def assign_platform_companies(self, platform_company_ids):
        """✅ NEW: Assign platform companies as employers for salary tracking"""
        self.platform_companies = platform_company_ids
        
        if platform_company_ids:
            # Assign primary platform as main employer
            self.primary_platform_id = random.choice(platform_company_ids)
            self.assign_employer(
                company_id=self.primary_platform_id,
                employment_start_date=datetime.now().date() - timedelta(days=random.randint(30, 1095))
            )
            
            # Set up payout schedules for each platform
            for platform_id in platform_company_ids:
                self.platform_payout_schedule[platform_id] = {
                    'payout_day': random.randint(25, 30),
                    'consistency': random.uniform(0.7, 0.95),
                    'avg_amount_multiplier': random.uniform(0.3, 1.2)
                }

    def assign_brand_companies(self, brand_company_ids):
        """✅ NEW: Assign brand companies for sponsorship salary tracking"""
        self.brand_companies = brand_company_ids
        for brand_id in brand_company_ids:
            self.relationship_start_dates[f'brand_{brand_id}'] = datetime.now().date()

    def _handle_platform_salary_payouts(self, date, events):
        """✅ NEW: Handle platform payouts as salary from employer companies"""
        # Monthly platform payouts as salary
        if (self.platform_companies and 
            date.day == self.monthly_platform_payout_day and
            random.random() < self.platform_income_consistency):
            
            platform_id = self.primary_platform_id or random.choice(self.platform_companies)
            
            # Calculate payout amount with variance
            base_amount = self.avg_monthly_income * random.uniform(0.6, 1.4)
            
            # Add seasonal variations
            month_multiplier = {
                12: 1.5,  # December - high advertising revenue
                11: 1.3,  # November
                1: 0.8,   # January - post-holiday dip
                2: 0.9,   # February
            }.get(date.month, 1.0)
            
            final_amount = base_amount * month_multiplier
            
            # ✅ NEW: Log as salary transaction from company
            txn = self.log_salary_transaction(
                amount=final_amount,
                date=date,
                company_id=platform_id
            )
            
            if txn:
                txn['transaction_category'] = 'platform_payout'
                txn['platform_type'] = 'social_media'
                events.append(txn)
                self.last_platform_payout_date = date

    def _handle_brand_sponsorship_salary(self, date, events):
        """✅ NEW: Handle sponsorship payments as salary from brand companies"""
        if (self.brand_companies and 
            random.random() < self.sponsorship_chance):
            
            brand_company_id = random.choice(self.brand_companies)
            sponsorship_amount = self.avg_monthly_income * random.uniform(2.0, 6.0)
            
            # ✅ NEW: Log as salary transaction from brand company
            txn = self.log_salary_transaction(
                amount=sponsorship_amount,
                date=date,
                company_id=brand_company_id
            )
            
            if txn:
                txn['transaction_category'] = 'sponsorship_payment'
                txn['company_type'] = 'brand_sponsor'
                events.append(txn)
                self.has_sponsorship_funds = True

    def _handle_income(self, date, events):
        """✅ UPDATED: Enhanced income handling with company salary tracking"""
        # Platform salary payouts
        self._handle_platform_salary_payouts(date, events)
        
        # Brand sponsorship salary
        self._handle_brand_sponsorship_salary(date, events)
        
        # Secondary platform payouts (not main employer)
        if (self.platform_companies and 
            len(self.platform_companies) > 1 and
            random.random() < (self.platform_payout_chance * 0.5)):
            
            # Secondary platform payout
            secondary_platforms = [p for p in self.platform_companies if p != self.primary_platform_id]
            if secondary_platforms:
                platform_id = random.choice(secondary_platforms)
                payout_amount = self.avg_monthly_income * random.uniform(0.1, 0.3)
                
                txn = self.log_salary_transaction(
                    amount=payout_amount,
                    date=date,
                    company_id=platform_id
                )
                
                if txn:
                    txn['transaction_category'] = 'secondary_platform_payout'
                    events.append(txn)

    def _handle_fixed_and_professional_expenses(self, date, events):
        """✅ UPDATED: Enhanced expense handling with merchant tracking"""
        # Loan EMI with variation
        emi_day = random.randint(8, 12)  # More realistic EMI dates
        if self.has_loan_emi and date.day == emi_day:
            # Add some variation to EMI amount
            emi_variation = random.uniform(0.95, 1.05)
            actual_emi = self.loan_emi_amount * emi_variation
            
            txn = self.log_transaction(
                "DEBIT", "Equipment Loan EMI", actual_emi, date, channel="Auto_Debit"
            )
            if txn:
                events.append(txn)
            
        # Software subscriptions with merchant tracking
        subscription_day = random.randint(3, 7)  # More varied subscription days
        if date.day == subscription_day:
            # ✅ NEW: Enhanced merchant tracking
            software_merchant_id = f"saas_provider_{hash(self.agent_id) % 10000}"
            self.add_frequent_merchant(software_merchant_id, date)
            
            # Add variation to subscription cost
            subscription_variation = random.uniform(0.9, 1.1)
            actual_cost = self.software_subscription * subscription_variation
            
            txn = self.log_merchant_transaction(
                merchant_id=software_merchant_id,
                amount=actual_cost,
                description="Professional Software Subscription",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

        # Utility bills with enhanced variation
        payment_delay = random.randint(0, 7) if random.random() < self.late_payment_chance else 0
        payment_day = self.utility_bill_day + payment_delay
        
        if date.day == payment_day:
            utility_bill_amount = random.uniform(1800, 5200)  # More realistic range
            utility_merchant_id = f"utility_{hash(self.agent_id) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=utility_merchant_id,
                amount=utility_bill_amount,
                description="Monthly Utility Bills",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def _handle_dynamic_spending(self, date, events):
        """✅ UPDATED: Enhanced spending with better merchant relationships"""
        spend_multiplier = 3.0 if self.has_sponsorship_funds else 1.0

        # Major equipment investments after sponsorships
        if (self.has_sponsorship_funds and 
            self.has_investment_activity and 
            random.random() < 0.65):
            
            spend_category = random.choice([
                "Professional Camera Equipment", 
                "Laptop/Computing Upgrade",
                "Studio Equipment",
                "Content Production Trip"
            ])
            
            amount = self.balance * random.uniform(0.25, 0.8)
            
            # ✅ NEW: Enhanced equipment merchant tracking
            if "Camera" in spend_category or "Equipment" in spend_category:
                equipment_merchant_id = f"pro_equipment_{hash(self.agent_id + spend_category) % 10000}"
                self.add_equipment_merchant(equipment_merchant_id, date)
                
                txn = self.log_merchant_transaction(
                    merchant_id=equipment_merchant_id,
                    amount=amount,
                    description=spend_category,
                    date=date,
                    channel="Netbanking"
                )
            else:
                txn = self.log_transaction("DEBIT", spend_category, amount, date, channel="Netbanking")
            
            if txn:
                events.append(txn)
                self.has_sponsorship_funds = False

        # Regular e-commerce with enhanced merchant diversity
        ecommerce_chance = 0.2 * spend_multiplier * self.spend_chance_mod
        if random.random() < ecommerce_chance:
            ecommerce_amt = random.uniform(1500, 18000)  # Wider range
            
            # ✅ NEW: Diverse merchant categories
            merchant_categories = ["fashion", "electronics", "props", "books", "software"]
            merchant_category = random.choice(merchant_categories)
            merchant_id = f"ecommerce_{merchant_category}_{hash(self.agent_id + str(date)) % 10000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=merchant_id,
                amount=ecommerce_amt,
                description=f"E-commerce Purchase ({merchant_category.title()})",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def _handle_collaboration_payments(self, date, events, context):
        """✅ UPDATED: Enhanced collaboration payments with temporal tracking"""
        if (self.collaborators and 
            random.random() < self.collaboration_payment_chance and
            self.balance > 8000):  # Higher threshold for collaboration payments
            
            collaborator = random.choice(self.collaborators)
            
            # More sophisticated payment calculation
            base_amount = self.avg_monthly_income * random.uniform(0.08, 0.35)
            
            # Economic class adjustments
            economic_multiplier = {
                'Lower': random.uniform(0.6, 0.9),
                'Lower_Middle': random.uniform(0.8, 1.2),
                'Middle': random.uniform(1.0, 1.5),
                'Upper_Middle': random.uniform(1.3, 2.0),
                'High': random.uniform(1.8, 2.5)
            }.get(self.economic_class, 1.0)
            
            final_amount = base_amount * economic_multiplier
            
            # Sponsorship funds boost
            if self.has_sponsorship_funds:
                final_amount *= random.uniform(1.4, 2.2)
            
            # ✅ NEW: Realistic channel selection
            if final_amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif final_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            # ✅ NEW: Enhanced temporal tracking
            self.last_collaboration_payment_date = date
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': collaborator, 
                'amount': round(final_amount, 2), 
                'desc': 'Collaboration Payment',
                'channel': channel,
                'transaction_category': 'collaboration_payment'
            })

    def _handle_freelancer_payments(self, date, events, context):
        """✅ UPDATED: Monthly freelancer payments with enhanced realism"""
        if (self.freelancer_network and 
            date.day == self.monthly_freelancer_payment_day and
            random.random() < self.freelancer_payment_chance and
            self.balance > 5000):
            
            freelancer = random.choice(self.freelancer_network)
            
            # Base payment calculation
            payment_amount = self.avg_monthly_income * random.uniform(0.12, 0.4)
            
            # Adjust for sponsorship periods
            if self.has_sponsorship_funds:
                payment_amount *= random.uniform(1.1, 1.7)
            
            # Economic class influence
            if self.economic_class in ['Upper_Middle', 'High']:
                payment_amount *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Realistic channel selection for freelancer payments
            if payment_amount > 75000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': freelancer, 
                'amount': round(payment_amount, 2), 
                'desc': 'Freelancer Payment',
                'channel': channel,
                'transaction_category': 'freelancer_payment'
            })

    def get_content_creation_cycle_features(self):
        """✅ ENHANCED: Comprehensive temporal and relationship features"""
        return {
            'platform_employer_count': len(self.platform_companies),
            'brand_sponsor_count': len(self.brand_companies),
            'primary_platform_tenure': self.get_employment_tenure_months(),
            'collaboration_frequency': len(self.collaborators) * self.collaboration_payment_chance,
            'freelancer_dependency_score': len(self.freelancer_network) * self.freelancer_payment_chance,
            'equipment_merchant_relationships': len(self.equipment_merchants),
            'platform_income_consistency': self.platform_income_consistency,
            'last_payout_recency': (datetime.now().date() - self.last_platform_payout_date).days if self.last_platform_payout_date else 999,
            'total_company_relationships': len(self.platform_companies) + len(self.brand_companies)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with company salary tracking"""
        events = []
        
        # Handle all income sources (including company salary tracking)
        self._handle_income(date, events)
        
        # Handle expenses with merchant tracking
        self._handle_fixed_and_professional_expenses(date, events)
        self._handle_dynamic_spending(date, events)
        
        # Handle P2P transfers
        self._handle_collaboration_payments(date, events, context)
        self._handle_freelancer_payments(date, events, context)
        self._handle_creator_network_transfers(date, events, context)
        self._handle_brand_advance_payments(date, events, context)
        self._handle_creator_support_activities(date, events, context)
        
        # Handle daily expenses
        self._handle_daily_living_expenses(date, events)
        
        return events

    # ✅ Keep existing P2P methods from original code...
    def _handle_creator_network_transfers(self, date, events, context):
        """Creator community transfers with personality-based adjustments"""
        if (self.creator_network and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 2000):
            
            recipient = random.choice(self.creator_network)
            base_amount = random.uniform(1000, 6000)
            
            # Personality and income adjustments
            if self.has_sponsorship_funds:
                base_amount *= random.uniform(1.3, 1.8)
            
            if self.financial_personality == 'Risk_Addict':
                base_amount *= random.uniform(1.1, 1.5)
            elif self.financial_personality == 'Saver':
                base_amount *= random.uniform(0.7, 1.0)
            
            channel = RealisticP2PStructure.select_realistic_channel()
            if base_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(base_amount, 2), 
                'desc': 'Creator Support',
                'channel': channel,
                'transaction_category': 'creator_support'
            })

    def _handle_brand_advance_payments(self, date, events, context):
        """Brand advance payments with appropriate channels"""
        if (self.brand_contacts and 
            random.random() < self.brand_advance_chance and
            self.balance > 15000 and
            self.has_sponsorship_funds):
            
            brand_contact = random.choice(self.brand_contacts)
            advance_amount = self.avg_monthly_income * random.uniform(0.15, 0.4)
            
            if self.economic_class in ['High', 'Upper_Middle']:
                advance_amount *= random.uniform(1.4, 2.2)
            
            if advance_amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif advance_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': brand_contact, 
                'amount': round(advance_amount, 2), 
                'desc': 'Brand Advance',
                'channel': channel,
                'transaction_category': 'brand_advance'
            })

    def _handle_creator_support_activities(self, date, events, context):
        """Supporting other creators with enhanced behavior"""
        if (self.creator_network and 
            random.random() < self.creator_support_chance and
            self.balance > 10000):
            
            recipient = random.choice(self.creator_network)
            support_amount = random.uniform(2000, 10000)
            
            if self.has_sponsorship_funds:
                support_amount *= random.uniform(1.2, 1.8)
            
            if self.financial_personality == 'Risk_Addict':
                support_amount *= random.uniform(1.1, 1.4)
            
            channel = RealisticP2PStructure.select_realistic_channel()
            if support_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(support_amount, 2), 
                'desc': 'Community Support',
                'channel': channel,
                'transaction_category': 'community_support'
            })

    def add_equipment_merchant(self, merchant_id, first_purchase_date=None):
        """Track equipment merchant relationships"""
        if merchant_id not in self.equipment_merchants:
            self.equipment_merchants.append(merchant_id)
            self.add_frequent_merchant(merchant_id, first_purchase_date)
