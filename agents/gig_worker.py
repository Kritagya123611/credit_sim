import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class GigWorker(BaseAgent):
    """
    Enhanced Gig Worker agent for Phase 2: Company salary source tracking
    Includes gig platform companies as employers, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Over_Spender'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Gig Worker"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "8000-35000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-employed",
            "employment_verification": "Not_Verified",
            "income_type": "Gig_Work",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Irregular",
            "savings_retention_rate": "Low",
            "has_investment_activity": len(personality_config['investment_types']) > 0 and financial_personality != 'Over_Spender',
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] * 0.5 else False,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "Mostly_On_Time",
            "mobile_plan_type": "Prepaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.60, 0.85), 3),
            "ip_consistency_score": round(random.uniform(0.45, 0.75), 3),
            "sim_churn_rate": "Medium",
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Irregular",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Low",
            
            # ✅ NEW: Heterogeneous graph connections specific to GigWorker
            "industry_sector": "Gig_Economy",
            "company_size": "Freelancer",
        }
        
        super().__init__(**profile_attributes)

        # Financial calculations with more variation
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_income = random.uniform(min_mod, max_mod)

        # ✅ NEW: Gig platforms as employers (salary source tracking)
        self.gig_platforms = []  # Uber, Upwork, Freelancer as company nodes
        self.primary_platform_id = None  # Main income source platform
        self.platform_payout_schedule = {}  # Track payout schedules per platform

        # ✅ NEW: Employment relationship tracking
        self.gig_activity_consistency = random.uniform(0.6, 0.8)  # Work regularity
        self.weekly_payout_day = random.randint(1, 7)  # Weekly payout day
        self.last_platform_payout_date = None

        # Income patterns with enhanced tracking
        self.daily_income_chance = random.uniform(0.30, 0.40)  # More variation
        self.avg_gig_payment = self.avg_monthly_income / random.uniform(6, 10)  # More realistic
        self.income_sources = ["Client Project", "Platform Payout", "Freelance Task"]

        # Fixed expenses with more variation
        self.rent_day = random.randint(5, 10)
        self.rent_amount = self.avg_monthly_income * random.uniform(0.25, 0.55)  # More realistic range
        self.bill_payment_late_chance = random.uniform(0.15, 0.25) / personality_config.get('spend_chance_mod', 1.0)
        
        # Spending patterns with personality influence
        self.daily_spend_chance = random.uniform(0.75, 0.85) * personality_config.get('spend_chance_mod', 1.0)
        self.prepaid_recharge_chance = random.uniform(0.08, 0.12) * personality_config.get('spend_chance_mod', 1.0)
        
        # ✅ Enhanced P2P networks for Gig Workers
        self.contacts = []  # General contacts
        self.peer_network = []  # Other gig workers and freelancers
        self.client_network = []  # Potential clients for payments
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.gig_platforms = []  # Uber, Upwork, Freelancer as company nodes
        self.service_merchants = []  # Work-related service providers
        self.equipment_vendors = []  # Tool/equipment purchase relationships
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.12, 0.18) * personality_config.get('spend_chance_mod', 1.0)
        self.peer_support_chance = random.uniform(0.08, 0.12)  # Supporting other gig workers
        self.client_advance_chance = random.uniform(0.03, 0.07)  # Receiving advances from clients
        self.emergency_help_chance = random.uniform(0.06, 0.10)  # Emergency mutual aid

        # Temporal tracking with enhanced cycles
        self.last_client_payment_date = None
        self.gig_work_cycles = []  # Track active vs inactive periods
        self.weekly_earnings_pattern = {}  # Track weekly earning patterns

        self.balance = random.uniform(self.avg_monthly_income * 0.03, self.avg_monthly_income * 0.25)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Gig workers typically have 1-3 devices (phone, laptop, sometimes tablet)"""
        device_options = [1, 2, 3]
        weights = [0.4, 0.4, 0.2]  # Most have 1-2 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_gig_platforms(self, platform_company_ids):
        """✅ NEW: Assign gig platform companies as employers for salary tracking"""
        self.gig_platforms = platform_company_ids
        
        if platform_company_ids:
            # Assign primary platform as main employer
            self.primary_platform_id = random.choice(platform_company_ids)
            self.assign_employer(
                company_id=self.primary_platform_id,
                employment_start_date=datetime.now().date() - timedelta(days=random.randint(60, 1095))
            )
            
            # Set up payout schedules for each platform
            for platform_id in platform_company_ids:
                self.platform_payout_schedule[platform_id] = {
                    'payout_probability': random.uniform(0.6, 0.8),
                    'weekly_payout_day': random.randint(1, 7),
                    'avg_amount_multiplier': random.uniform(0.4, 1.6)
                }

    def _handle_platform_salary_payouts(self, date, events):
        """✅ NEW: Handle platform payouts as salary from employer companies"""
        # Weekly platform payouts as salary
        if (self.gig_platforms and 
            date.weekday() == self.weekly_payout_day and
            random.random() < self.gig_activity_consistency):
            
            platform_id = self.primary_platform_id or random.choice(self.gig_platforms)
            
            # Calculate weekly payout with variance
            base_amount = self.avg_gig_payment * random.uniform(3, 7)  # 3-7 gigs per week
            
            # Add day-of-week variations (weekends higher for some gigs)
            if date.weekday() >= 5:  # Weekend
                base_amount *= random.uniform(1.1, 1.4)
            
            # Add seasonal variations
            month_multiplier = {
                12: 1.3,  # December - holiday gigs
                11: 1.2,  # November - pre-holiday
                1: 0.8,   # January - post-holiday low
                2: 0.9,   # February - winter low
            }.get(date.month, 1.0)
            
            final_amount = base_amount * month_multiplier
            
            # ✅ NEW: Log as salary transaction from company
            txn = self.log_salary_transaction(
                amount=final_amount,
                date=date,
                company_id=platform_id
            )
            
            if txn:
                txn['transaction_category'] = 'gig_platform_payout'
                txn['platform_type'] = 'gig_economy'
                events.append(txn)
                self.last_platform_payout_date = date

    def _handle_income(self, date, events):
        """✅ UPDATED: Enhanced income handling with company salary tracking"""
        # Platform salary payouts
        self._handle_platform_salary_payouts(date, events)

        # Daily gig income from individual tasks
        if random.random() < self.daily_income_chance:
            income_amount = self.avg_gig_payment * random.uniform(0.6, 1.4)
            income_source = random.choice(self.income_sources)
            
            # ✅ NEW: Some income through secondary platforms
            if (self.gig_platforms and 
                "Platform Payout" in income_source and
                random.random() < 0.3):  # 30% secondary platform
                
                secondary_platforms = [p for p in self.gig_platforms if p != self.primary_platform_id]
                if secondary_platforms:
                    platform_id = random.choice(secondary_platforms)
                    txn = self.log_salary_transaction(
                        amount=income_amount,
                        date=date,
                        company_id=platform_id
                    )
                    if txn:
                        txn['transaction_category'] = 'secondary_gig_payout'
                        events.append(txn)
                else:
                    txn = self.log_transaction(
                        "CREDIT", income_source, income_amount, date, channel="Bank_Transfer"
                    )
                    if txn:
                        events.append(txn)
            else:
                # Direct client payment
                txn = self.log_transaction(
                    "CREDIT", income_source, income_amount, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)

    def _handle_bills(self, date, events):
        """✅ UPDATED: Enhanced bill handling with merchant tracking"""
        # Rent payment with occasional delays and variation
        rent_payment_day = self.rent_day + (random.randint(0, 4) if random.random() < self.bill_payment_late_chance else 0)
        if date.day == rent_payment_day:
            # Add some variation to rent amount
            rent_variation = random.uniform(0.95, 1.05)
            actual_rent = self.rent_amount * rent_variation
            
            rent_merchant_id = f"rent_landlord_{hash(self.agent_id) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=rent_merchant_id,
                amount=actual_rent,
                description="Monthly Rent",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

        # Prepaid mobile recharge with enhanced variation
        if random.random() < self.prepaid_recharge_chance:
            recharge_options = [129, 149, 199, 239, 299, 399]
            recharge_amount = random.choice(recharge_options)
            recharge_merchant_id = f"telecom_provider_{hash(self.agent_id) % 100}"
            
            txn = self.log_merchant_transaction(
                merchant_id=recharge_merchant_id,
                amount=recharge_amount,
                description="Prepaid Mobile Recharge",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def _handle_daily_spending(self, date, events):
        """✅ UPDATED: Enhanced daily spending with merchant diversity"""
        if random.random() < self.daily_spend_chance:
            spend_categories = [
                ("Food", random.uniform(40, 350)),
                ("Transport", random.uniform(30, 200)),
                ("Groceries", random.uniform(80, 500)),
                ("Tea_Snacks", random.uniform(20, 150))
            ]
            
            spend_category, spend_amount = random.choice(spend_categories)
            
            # Economic class adjustments
            if self.economic_class == 'Lower':
                spend_amount *= random.uniform(0.6, 0.9)
            elif self.economic_class in ['Middle', 'Upper_Middle']:
                spend_amount *= random.uniform(1.1, 1.4)
            
            # ✅ NEW: Enhanced merchant tracking
            merchant_id = f"daily_{spend_category.lower()}_{hash(self.agent_id + str(date)) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=merchant_id,
                amount=spend_amount,
                description=f"Daily {spend_category.replace('_', '/')}",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def add_service_merchant(self, merchant_id, first_service_date=None):
        """✅ NEW: Track service merchant relationships"""
        if merchant_id not in self.service_merchants:
            self.service_merchants.append(merchant_id)
            self.add_frequent_merchant(merchant_id, first_service_date)

    def _handle_peer_network_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced peer transfers with realistic channels"""
        if (self.contacts and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 300):
            
            recipient = random.choice(self.contacts)
            amount = random.uniform(150, 800)
            
            # Economic class and personality adjustments
            economic_multiplier = {
                'Lower': random.uniform(0.7, 1.0),
                'Lower_Middle': random.uniform(0.9, 1.2),
                'Middle': random.uniform(1.0, 1.3)
            }.get(self.economic_class, 1.0)
            
            if self.financial_personality == 'Over_Spender':
                amount *= random.uniform(1.1, 1.4)
            elif self.financial_personality == 'Saver':
                amount *= random.uniform(0.7, 1.0)
            
            final_amount = amount * economic_multiplier
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(final_amount, 2), 
                'desc': 'Peer Network Transfer',
                'channel': channel,
                'transaction_category': 'peer_transfer'
            })

    def _handle_peer_support_activities(self, date, events, context):
        """✅ UPDATED: Mutual support within gig worker community"""
        if (self.peer_network and 
            random.random() < self.peer_support_chance and
            self.balance > 500):
            
            recipient = random.choice(self.peer_network)
            
            # Support amounts based on current balance and economic class
            base_amount = random.uniform(200, 1000)
            
            # Can't give more than 25% of balance
            max_support = self.balance * 0.25
            final_amount = min(base_amount, max_support)
            
            if final_amount >= 150:
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Gig Worker Support',
                    'channel': channel,
                    'transaction_category': 'peer_support'
                })

    def _handle_emergency_support(self, date, events, context):
        """✅ UPDATED: Emergency support within gig worker network"""
        if (self.contacts and 
            random.random() < self.emergency_help_chance and
            self.balance > 800):
            
            recipient = random.choice(self.contacts)
            
            # Emergency amounts with personality adjustments
            emergency_amount = random.uniform(400, 1500)
            
            if self.financial_personality == 'Over_Spender':
                emergency_amount *= random.uniform(1.1, 1.4)
            elif self.financial_personality == 'Saver':
                emergency_amount *= random.uniform(0.8, 1.0)
            
            # Can't give more than 35% of balance in emergency
            max_emergency = self.balance * 0.35
            final_amount = min(emergency_amount, max_emergency)
            
            if final_amount >= 250:
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Emergency Financial Support',
                    'channel': channel,
                    'transaction_category': 'emergency_support'
                })

    def _handle_client_advance_requests(self, date, events, context):
        """✅ UPDATED: Enhanced client advance handling"""
        if (self.client_network and 
            random.random() < self.client_advance_chance and
            self.balance < 1500):
            
            client = random.choice(self.client_network)
            
            # Advance amounts based on typical project values
            advance_amount = random.uniform(800, 4000)
            
            # Adjust based on economic class
            if self.economic_class in ['Lower', 'Lower_Middle']:
                advance_amount *= random.uniform(0.6, 1.0)
            elif self.economic_class in ['Middle', 'Upper_Middle']:
                advance_amount *= random.uniform(1.2, 1.6)
            
            # ✅ NEW: Select appropriate channel
            if advance_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            # ✅ NEW: Track client advance for temporal analysis
            self.last_client_payment_date = date
            
            context.get('p2p_transfers', []).append({
                'sender': client,
                'recipient': self, 
                'amount': round(advance_amount, 2), 
                'desc': 'Client Project Advance',
                'channel': channel,
                'transaction_category': 'client_advance'
            })

    def _handle_equipment_purchases(self, date, events, context):
        """✅ UPDATED: Work-related equipment and tool purchases"""
        equipment_purchase_day = random.randint(20, 25)
        if (date.day == equipment_purchase_day and
            random.random() < 0.06 and  # 6% monthly chance
            self.balance > 2000):
            
            # Equipment purchases for gig work
            equipment_amount = random.uniform(400, 2000)
            
            # Higher amounts for better economic class
            if self.economic_class in ['Middle', 'Upper_Middle']:
                equipment_amount *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Enhanced equipment vendor tracking
            equipment_vendor_id = f"gig_equipment_{hash(self.agent_id + str(date)) % 1000}"
            self.add_frequent_merchant(equipment_vendor_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=equipment_vendor_id,
                amount=equipment_amount,
                description="Gig Work Equipment",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def get_gig_worker_features(self):
        """✅ ENHANCED: Comprehensive gig worker features"""
        return {
            'platform_employer_count': len(self.gig_platforms),
            'primary_platform_tenure': self.get_employment_tenure_months(),
            'gig_activity_consistency': self.gig_activity_consistency,
            'service_merchant_relationships': len(self.service_merchants),
            'equipment_vendor_relationships': len(self.equipment_vendors),
            'peer_network_size': len(self.peer_network),
            'client_network_size': len(self.client_network),
            'income_irregularity_score': 1.0 - self.daily_income_chance,
            'peer_support_activity': self.peer_support_chance + self.emergency_help_chance,
            'last_platform_payout_recency': (datetime.now().date() - self.last_platform_payout_date).days if self.last_platform_payout_date else 999,
            'financial_instability_score': self.bill_payment_late_chance,
            'total_company_relationships': len(self.gig_platforms)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with company salary tracking"""
        events = []
        
        # Handle all income sources (including company salary tracking)
        self._handle_income(date, events)
        
        # Handle expenses
        self._handle_bills(date, events)
        self._handle_daily_spending(date, events)
        
        # Handle P2P transfers
        self._handle_peer_network_transfers(date, events, context)
        self._handle_peer_support_activities(date, events, context)
        self._handle_emergency_support(date, events, context)
        self._handle_client_advance_requests(date, events, context)
        
        # Handle equipment purchases
        self._handle_equipment_purchases(date, events, context)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
