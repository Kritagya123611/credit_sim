import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class DeliveryAgent(BaseAgent):
    """
    Enhanced Delivery Agent for Phase 2: Company salary source tracking
    Includes delivery platform companies as employers, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Lower', financial_personality='Over_Spender'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Delivery Agent"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "15000-25000"
        min_monthly, max_monthly = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_monthly * income_multiplier)}-{int(max_monthly * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Gig_Work_Contractor",
            "employment_verification": "Not_Verified",
            "income_type": "Platform_Payout",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Daily",
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "Mostly_On_Time",
            "mobile_plan_type": "Prepaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.80, 0.95), 3),
            "ip_consistency_score": round(random.uniform(0.25, 0.60), 3),
            "sim_churn_rate": "Medium",
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Geographically_Dynamic",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Low",
            
            # ✅ NEW: Heterogeneous graph connections specific to DeliveryAgent
            "industry_sector": "Logistics_Delivery",
            "company_size": "Gig_Platform",
        }
        
        super().__init__(**profile_attributes)

        # Financial calculations with more variation
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.base_daily_payout = random.uniform(min_mod, max_mod) / random.uniform(24, 28)  # More realistic working days

        # ✅ NEW: Platform companies as employers (salary source tracking)
        self.delivery_platforms = []  # Swiggy, Zomato, Uber Eats as company nodes
        self.primary_platform_id = None  # Main income source platform
        self.platform_payout_schedule = {}  # Track payout schedules per platform

        # ✅ NEW: Employment relationship tracking
        self.daily_payout_consistency = random.uniform(0.7, 0.9)  # Payment reliability
        self.weekly_settlement_day = random.randint(1, 7)  # Weekly settlement day
        self.last_platform_payout_date = None

        # Fixed expenses with more variation
        self.loan_emi_amount = (min_mod + max_mod) / 2 * random.uniform(0.12, 0.18)  # More varied EMI
        self.cod_settlement_chance = random.uniform(0.55, 0.65)  # More variation
        self.cod_balance = 0.0

        # Operational expenses with personality influence
        self.fuel_spend_chance = random.uniform(0.85, 0.95) * personality_config.get('spend_chance_mod', 1.0)
        self.recharge_chance = random.uniform(0.08, 0.12) * personality_config.get('spend_chance_mod', 1.0)

        # ✅ Enhanced P2P networks for Delivery Agents
        self.fellow_agents = []  # Other delivery riders
        self.family_back_home = []  # Family for remittances
        self.peer_network = []  # Other gig workers
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.delivery_platforms = []  # Swiggy, Zomato, etc. as company nodes
        self.fuel_stations = []  # Regular fuel merchant relationships
        self.vehicle_service_merchants = []  # Bike service centers
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.18, 0.22) * personality_config.get('spend_chance_mod', 1.0)
        self.agent_support_chance = random.uniform(0.12, 0.18)
        self.family_remittance_chance = random.uniform(0.10, 0.14)
        self.emergency_help_chance = random.uniform(0.06, 0.10)
        
        # Operational sharing patterns with variation
        self.fuel_sharing_chance = random.uniform(0.08, 0.12)
        self.vehicle_maintenance_sharing = random.uniform(0.03, 0.07)
        
        # Temporal tracking with enhanced cycles
        self.last_family_remittance_date = None
        self.delivery_cycles = []  # Track peak vs off-peak patterns
        self.weekly_earnings_pattern = {}  # Track weekly earning patterns

        self.balance = random.uniform(300, 2500)  # More realistic range for delivery agents

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Delivery agents typically have 1-3 devices (phone, backup phone, sometimes GPS)"""
        device_options = [1, 2, 3]
        weights = [0.3, 0.5, 0.2]  # Most have 2 devices (primary + backup phone)
        return random.choices(device_options, weights=weights)[0]

    def assign_delivery_platforms(self, platform_company_ids):
        """✅ NEW: Assign delivery platform companies as employers for salary tracking"""
        self.delivery_platforms = platform_company_ids
        
        if platform_company_ids:
            # Assign primary platform as main employer
            self.primary_platform_id = random.choice(platform_company_ids)
            self.assign_employer(
                company_id=self.primary_platform_id,
                employment_start_date=datetime.now().date() - timedelta(days=random.randint(30, 730))
            )
            
            # Set up payout schedules for each platform
            for platform_id in platform_company_ids:
                self.platform_payout_schedule[platform_id] = {
                    'daily_payout_probability': random.uniform(0.7, 0.95),
                    'weekly_settlement_day': random.randint(1, 7),
                    'avg_amount_multiplier': random.uniform(0.5, 1.5)
                }

    def _handle_platform_salary_payouts(self, date, events):
        """✅ NEW: Handle daily platform payouts as salary from employer companies"""
        # Daily platform payouts as salary
        if (self.delivery_platforms and 
            random.random() < self.daily_payout_consistency):
            
            platform_id = self.primary_platform_id or random.choice(self.delivery_platforms)
            
            # Calculate daily payout with variance based on day of week
            base_amount = self.base_daily_payout
            
            # Weekend multiplier (higher demand)
            weekend_multiplier = 1.3 if date.weekday() >= 5 else 1.0
            
            # Add seasonal variations
            month_multiplier = {
                12: 1.2,  # December - holiday season
                1: 0.9,   # January - post-holiday low
                6: 1.1,   # June - summer high
                7: 1.1,   # July - summer high
            }.get(date.month, 1.0)
            
            final_amount = base_amount * weekend_multiplier * month_multiplier * random.uniform(0.7, 1.4)
            
            # ✅ NEW: Log as salary transaction from company
            txn = self.log_salary_transaction(
                amount=final_amount,
                date=date,
                company_id=platform_id
            )
            
            if txn:
                txn['transaction_category'] = 'daily_platform_payout'
                txn['platform_type'] = 'delivery_platform'
                events.append(txn)
                self.last_platform_payout_date = date

        # ✅ NEW: Weekly settlements for secondary platforms
        if (len(self.delivery_platforms) > 1 and 
            date.weekday() == self.weekly_settlement_day and
            random.random() < 0.6):
            
            secondary_platforms = [p for p in self.delivery_platforms if p != self.primary_platform_id]
            if secondary_platforms:
                platform_id = random.choice(secondary_platforms)
                weekly_amount = self.base_daily_payout * random.uniform(2, 5)  # 2-5 days worth
                
                txn = self.log_salary_transaction(
                    amount=weekly_amount,
                    date=date,
                    company_id=platform_id
                )
                
                if txn:
                    txn['transaction_category'] = 'weekly_platform_settlement'
                    events.append(txn)

    def _handle_income_and_settlements(self, date, events):
        """✅ UPDATED: Enhanced income handling with company salary tracking"""
        # Platform salary payouts
        self._handle_platform_salary_payouts(date, events)

        # COD settlement cycle with enhanced tracking
        if random.random() < self.cod_settlement_chance:
            num_cod_orders = random.randint(2, 12)  # More realistic range
            daily_cod_total = 0
            
            for _ in range(num_cod_orders):
                cod_amount = random.uniform(80, 900)  # More realistic COD range
                txn = self.log_transaction(
                    "CREDIT", "Cash on Delivery Collection", cod_amount, date, channel="Cash_Deposit"
                )
                if txn:
                    events.append(txn)
                    daily_cod_total += cod_amount

            self.cod_balance += daily_cod_total

            # Settlement to platform (end of day)
            if self.cod_balance > 500:  # Minimum threshold for settlement
                platform_id = self.primary_platform_id if self.primary_platform_id else (
                    random.choice(self.delivery_platforms) if self.delivery_platforms else None
                )
                
                if platform_id:
                    txn = self.log_merchant_transaction(
                        merchant_id=platform_id,
                        amount=self.cod_balance,
                        description="COD Settlement to Platform",
                        date=date,
                        channel="UPI"
                    )
                else:
                    txn = self.log_transaction(
                        "DEBIT", "COD Settlement to Platform", self.cod_balance, date, channel="UPI"
                    )
                
                if txn:
                    txn['transaction_category'] = 'cod_settlement'
                    events.append(txn)
                    self.cod_balance = 0.0

    def _handle_fixed_debits(self, date, events):
        """✅ UPDATED: Enhanced loan payment handling"""
        # Vehicle loan EMI with variation
        emi_day = random.randint(8, 12)  # More realistic EMI dates
        if self.has_loan_emi and date.day == emi_day:
            # 85% on-time payment with some amount variation
            if random.random() > 0.15:
                # Add slight variation to EMI amount (processing fees, etc.)
                emi_variation = random.uniform(0.98, 1.02)
                actual_emi = self.loan_emi_amount * emi_variation
                
                txn = self.log_transaction(
                    "DEBIT", "Two-Wheeler Loan EMI", actual_emi, date, channel="Auto_Debit"
                )
                if txn:
                    events.append(txn)

    def _handle_operational_spending(self, date, events):
        """✅ UPDATED: Enhanced operational costs with merchant tracking"""
        # Fuel expenses with enhanced merchant tracking
        if random.random() < self.fuel_spend_chance:
            fuel_amount = random.uniform(120, 450)  # More realistic fuel range
            
            # ✅ NEW: Enhanced fuel station tracking
            fuel_station_id = f"fuel_station_{hash(self.agent_id) % 1000}"
            self.add_fuel_station(fuel_station_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=fuel_station_id,
                amount=fuel_amount,
                description="Fuel Purchase",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

        # Mobile recharge with variation
        if random.random() < self.recharge_chance:
            # More realistic recharge amounts and variation
            recharge_options = [79, 99, 129, 149, 199, 249]
            recharge_amount = random.choice(recharge_options)
            recharge_merchant_id = f"recharge_provider_{hash(self.agent_id) % 100}"
            
            txn = self.log_merchant_transaction(
                merchant_id=recharge_merchant_id,
                amount=recharge_amount,
                description="Mobile Recharge",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

        # ✅ NEW: Vehicle maintenance expenses (periodic)
        maintenance_chance = 0.02  # 2% daily chance
        if random.random() < maintenance_chance:
            maintenance_amount = random.uniform(300, 1500)
            service_merchant_id = f"vehicle_service_{hash(self.agent_id + str(date)) % 500}"
            self.add_service_merchant(service_merchant_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=service_merchant_id,
                amount=maintenance_amount,
                description="Vehicle Maintenance",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def add_fuel_station(self, merchant_id, first_visit_date=None):
        """✅ NEW: Track fuel station merchant relationships"""
        if merchant_id not in self.fuel_stations:
            self.fuel_stations.append(merchant_id)
            self.add_frequent_merchant(merchant_id, first_visit_date)

    def add_service_merchant(self, merchant_id, first_service_date=None):
        """✅ NEW: Track vehicle service merchant relationships"""
        if merchant_id not in self.vehicle_service_merchants:
            self.vehicle_service_merchants.append(merchant_id)
            self.add_frequent_merchant(merchant_id, first_service_date)

    def _handle_agent_community_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced agent-to-agent transfers"""
        if (self.fellow_agents and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 150):
            
            recipient = random.choice(self.fellow_agents)
            
            # Amount based on current financial situation
            base_amount = random.uniform(100, 800)
            
            # Economic class adjustments
            economic_multiplier = {
                'Lower': random.uniform(0.6, 1.0),
                'Lower_Middle': random.uniform(0.8, 1.2),
                'Middle': random.uniform(1.0, 1.4)
            }.get(self.economic_class, 1.0)
            
            # Adjust based on current balance
            max_sendable = self.balance * 0.20  # Max 20% of balance
            final_amount = min(base_amount * economic_multiplier, max_sendable)
            
            if final_amount >= 50:
                # ✅ NEW: Realistic channel selection
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Peer Support Transfer',
                    'channel': channel,
                    'transaction_category': 'peer_support'
                })

    def _handle_family_remittances(self, date, events, context):
        """✅ UPDATED: Enhanced family remittance handling"""
        # Send money home weekly (typically Sundays or Mondays)
        if (self.family_back_home and 
            date.weekday() in [0, 6] and  # Monday or Sunday
            random.random() < self.family_remittance_chance and
            self.balance > 1000):
            
            current_date_key = date.strftime("%Y-%W")  # Weekly key
            if self.last_family_remittance_date != current_date_key:
                recipient = random.choice(self.family_back_home)
                
                # Remittances typically 25-45% of available balance
                remittance_percentage = random.uniform(0.25, 0.45)
                
                # Adjust based on personality
                if self.financial_personality == 'Saver':
                    remittance_percentage *= random.uniform(1.1, 1.4)
                elif self.financial_personality == 'Over_Spender':
                    remittance_percentage *= random.uniform(0.8, 1.0)
                
                remittance_amount = self.balance * remittance_percentage
                
                if remittance_amount >= 500:
                    # ✅ NEW: Select appropriate channel for remittances
                    if remittance_amount > 50000:
                        channel = random.choice(['IMPS', 'NEFT'])
                    else:
                        channel = RealisticP2PStructure.select_realistic_channel()
                    
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': recipient, 
                        'amount': round(remittance_amount, 2), 
                        'desc': 'Family Remittance',
                        'channel': channel,
                        'transaction_category': 'family_remittance'
                    })
                    self.last_family_remittance_date = current_date_key

    def _handle_operational_sharing(self, date, events, context):
        """✅ UPDATED: Operational cost sharing among delivery agents"""
        # Fuel sharing with enhanced realism
        if (self.fellow_agents and 
            random.random() < self.fuel_sharing_chance and
            self.balance > 200):
            
            recipient = random.choice(self.fellow_agents)
            fuel_share_amount = random.uniform(80, 250)  # Realistic fuel sharing
            
            # Ensure affordability
            max_fuel_share = self.balance * 0.15
            final_amount = min(fuel_share_amount, max_fuel_share)
            
            if final_amount >= 50:
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Fuel Cost Sharing',
                    'channel': channel,
                    'transaction_category': 'fuel_sharing'
                })
        
        # Vehicle maintenance sharing
        if (self.fellow_agents and 
            random.random() < self.vehicle_maintenance_sharing and
            self.balance > 400):
            
            recipient = random.choice(self.fellow_agents)
            maintenance_share = random.uniform(150, 600)
            
            # Maximum 25% of balance for maintenance sharing
            max_share = self.balance * 0.25
            final_amount = min(maintenance_share, max_share)
            
            if final_amount >= 100:
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Maintenance Cost Sharing',
                    'channel': channel,
                    'transaction_category': 'maintenance_sharing'
                })

    def _handle_emergency_support(self, date, events, context):
        """✅ UPDATED: Emergency support with realistic constraints"""
        if (self.fellow_agents and 
            random.random() < self.emergency_help_chance and
            self.balance > 600):
            
            recipient = random.choice(self.fellow_agents)
            emergency_amount = random.uniform(200, 1000)
            
            # Maximum 35% of balance for emergency
            max_emergency = self.balance * 0.35
            final_amount = min(emergency_amount, max_emergency)
            
            if final_amount >= 150:
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Emergency Financial Support',
                    'channel': channel,
                    'transaction_category': 'emergency_support'
                })

    def _handle_peer_network_transfers(self, date, events, context):
        """✅ UPDATED: Broader peer network support"""
        if (self.peer_network and 
            random.random() < self.agent_support_chance and
            self.balance > 300):
            
            recipient = random.choice(self.peer_network)
            peer_amount = random.uniform(75, 500)
            
            # Maximum 18% for peer transfers
            max_peer = self.balance * 0.18
            final_amount = min(peer_amount, max_peer)
            
            if final_amount >= 50:
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Peer Network Support',
                    'channel': channel,
                    'transaction_category': 'peer_support'
                })

    def get_delivery_agent_features(self):
        """✅ ENHANCED: Comprehensive delivery agent features"""
        return {
            'platform_employer_count': len(self.delivery_platforms),
            'primary_platform_tenure': self.get_employment_tenure_months(),
            'platform_income_consistency': self.daily_payout_consistency,
            'fuel_station_relationships': len(self.fuel_stations),
            'service_merchant_relationships': len(self.vehicle_service_merchants),
            'peer_network_size': len(self.fellow_agents) + len(self.peer_network),
            'family_dependency_score': len(self.family_back_home),
            'operational_sharing_activity': self.fuel_sharing_chance + self.vehicle_maintenance_sharing,
            'last_payout_recency': (datetime.now().date() - self.last_platform_payout_date).days if self.last_platform_payout_date else 999,
            'cod_handling_frequency': self.cod_settlement_chance,
            'total_company_relationships': len(self.delivery_platforms)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with company salary tracking"""
        events = []
        
        # Handle all income sources (including company salary tracking)
        self._handle_income_and_settlements(date, events)
        
        # Handle fixed expenses
        self._handle_fixed_debits(date, events)
        
        # Handle operational spending with merchant tracking
        self._handle_operational_spending(date, events)
        
        # Handle P2P transfers
        self._handle_agent_community_transfers(date, events, context)
        self._handle_family_remittances(date, events, context)
        self._handle_operational_sharing(date, events, context)
        self._handle_emergency_support(date, events, context)
        self._handle_peer_network_transfers(date, events, context)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events, daily_spend_chance=0.6)
        
        return events
