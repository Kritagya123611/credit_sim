import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class DailyWageLaborer(BaseAgent):
    """
    Enhanced Daily Wage Laborer agent for Phase 2: Contractor salary source tracking
    Includes contractor companies as employers, realistic informal employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Lower', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Daily Wage Laborer"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "7000-15000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Informal_Labor",
            "employment_verification": "Not_Verified",
            "income_type": "Cash_Deposit, Wages",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Daily",
            "savings_retention_rate": "Near_Zero",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": False,
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "N/A",
            "mobile_plan_type": "Prepaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.25, 0.60), 3),
            "ip_consistency_score": round(random.uniform(0.15, 0.45), 3),
            "sim_churn_rate": random.choice(["High", "Very_High"]),
            "primary_digital_channels": ["Cash", "UPI"],
            "login_pattern": "Irregular",
            "ecommerce_activity_level": "None",
            "ecommerce_avg_ticket_size": "N/A",
            
            # ✅ NEW: Heterogeneous graph connections specific to DailyWageLaborer
            "industry_sector": "Construction_Labor",
            "company_size": "Small_Contractor",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Contractors as employers (salary source tracking)
        self.contractors = []  # Construction contractors as company nodes
        self.primary_contractor_id = None  # Main work source contractor
        self.contractor_relationships = {}  # Track contractor work patterns

        # ✅ NEW: Employment relationship tracking
        self.work_consistency = random.uniform(0.6, 0.8)  # Work availability
        self.contractor_payment_reliability = random.uniform(0.7, 0.9)  # Payment consistency
        self.last_wage_date = None

        # Financial calculations with more variation
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_income = random.uniform(min_mod, max_mod)

        # Work patterns with enhanced tracking
        self.daily_work_chance = random.uniform(0.70, 0.80) * (1 + (class_config['loan_propensity'] * 0.2))
        self.daily_wage_amount = self.avg_monthly_income / random.uniform(20, 25)  # 20-25 working days
        
        # Remittance patterns with more variation
        remittance_base = random.uniform(0.55, 0.85)
        personality_multiplier = 1.15 if financial_personality == 'Saver' else 1.0
        self.remittance_percentage = remittance_base * personality_multiplier
        self.recharge_chance = random.uniform(0.03, 0.07)
        
        # ✅ Enhanced P2P networks for Daily Wage Laborers
        self.worker_network = []  # Fellow laborers and construction workers
        self.family_recipient = None  # Single family member for regular remittances
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.contractors = []  # Construction contractors as company nodes
        self.local_merchants = []  # Local shops for essentials
        self.remittance_agents = []  # Money transfer agents as merchant nodes
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.12, 0.18) * personality_config.get('spend_chance_mod', 1.0)
        self.emergency_help_chance = random.uniform(0.02, 0.04)
        self.community_support_chance = random.uniform(0.06, 0.10)
        
        # Temporal tracking with enhanced features
        self.last_remittance_day = None
        self.work_cycles = []  # Track work availability patterns
        self.seasonal_patterns = []  # Construction work varies by season

        self.balance = random.uniform(30, 250)  # Very low balance range

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Daily wage laborers typically have 1-2 devices (basic phone, sometimes backup)"""
        device_options = [1, 2]
        weights = [0.7, 0.3]  # Most have just 1 device
        return random.choices(device_options, weights=weights)[0]

    def assign_contractors(self, contractor_company_ids):
        """✅ NEW: Assign contractor companies as employers for salary tracking"""
        self.contractors = contractor_company_ids
        
        if contractor_company_ids:
            # Assign primary contractor as main employer
            self.primary_contractor_id = random.choice(contractor_company_ids)
            self.assign_employer(
                company_id=self.primary_contractor_id,
                employment_start_date=datetime.now().date() - timedelta(days=random.randint(30, 730))
            )
            
            # Set up contractor relationships
            for contractor_id in contractor_company_ids:
                self.contractor_relationships[contractor_id] = {
                    'work_type': random.choice(['Construction', 'Loading', 'Cleaning', 'General_Labor']),
                    'payment_reliability': random.uniform(0.6, 0.9),
                    'daily_wage_rate': random.uniform(200, 500)
                }

    def _handle_contractor_wage_payment(self, date, events):
        """✅ NEW: Handle daily wage from contractor companies"""
        if (random.random() < self.daily_work_chance and 
            random.random() < self.contractor_payment_reliability):
            
            # Determine wage source
            wage_amount = self.daily_wage_amount * random.uniform(0.8, 1.2)
            
            # ✅ NEW: 70% from contractors, 30% cash work
            if (self.contractors and 
                random.random() < 0.7):
                
                contractor_id = self.primary_contractor_id or random.choice(self.contractors)
                
                # Add seasonal variations for construction work
                month_multiplier = {
                    12: 0.8,  # December - winter slowdown
                    1: 0.7,   # January - post-holiday low
                    2: 0.8,   # February - winter
                    6: 1.2,   # June - pre-monsoon rush
                    7: 0.9,   # July - monsoon impact
                    8: 0.9,   # August - monsoon
                    10: 1.3,  # October - post-monsoon construction boom
                    11: 1.2,  # November - peak construction
                }.get(date.month, 1.0)
                
                final_wage = wage_amount * month_multiplier
                
                # ✅ NEW: Log as salary transaction from contractor
                txn = self.log_salary_transaction(
                    amount=final_wage,
                    date=date,
                    company_id=contractor_id
                )
                
                if txn:
                    txn['transaction_category'] = 'contractor_daily_wage'
                    txn['company_type'] = 'construction_contractor'
                    events.append(txn)
                    self.last_wage_date = date
            else:
                # Cash work without formal contractor
                txn = self.log_transaction(
                    "CREDIT", "Cash Daily Wage", wage_amount, date, channel="Cash_Deposit"
                )
                if txn:
                    events.append(txn)
                    self.last_wage_date = date

    def _handle_daily_income(self, date, events):
        """✅ UPDATED: Enhanced daily income with contractor tracking"""
        self._handle_contractor_wage_payment(date, events)

    def _handle_cash_withdrawals(self, date, events):
        """✅ UPDATED: Enhanced cash withdrawals for daily expenses"""
        if (self.balance > 100 and 
            random.random() < 0.35):  # 35% chance of cash withdrawal
            
            # Keep minimal cash for survival, withdraw rest for daily use
            cash_out_percentage = random.uniform(0.6, 0.8)
            cash_out_amount = self.balance * cash_out_percentage
            
            if cash_out_amount > 50:
                txn = self.log_transaction(
                    "DEBIT", "Daily Cash Withdrawal", cash_out_amount, date, channel="ATM"
                )
                if txn:
                    events.append(txn)

    def add_remittance_agent(self, agent_id, first_transaction_date=None):
        """✅ NEW: Track money transfer agent relationships"""
        if agent_id not in self.remittance_agents:
            self.remittance_agents.append(agent_id)
            self.add_frequent_merchant(agent_id, first_transaction_date)

    def _handle_family_remittances(self, date, events, context):
        """✅ UPDATED: Enhanced family remittances with agent tracking"""
        current_day_key = date.strftime("%Y-%m-%d")
        
        if (self.family_recipient and 
            self.balance >= self.daily_wage_amount * 0.8 and  # Ensure we just got paid
            self.last_remittance_day != current_day_key and
            random.random() < 0.45):  # 45% chance of sending money after work
            
            # Calculate remittance with variation
            base_remittance = self.daily_wage_amount * self.remittance_percentage
            
            # Adjust based on economic pressures
            economic_multiplier = {
                'Lower': random.uniform(0.8, 1.0),
                'Lower_Middle': random.uniform(0.9, 1.1)
            }.get(self.economic_class, 1.0)
            
            remittance_amount = base_remittance * economic_multiplier
            
            if remittance_amount >= 50:  # Minimum threshold
                # ✅ NEW: Use remittance agent for larger amounts
                if (remittance_amount > 1000 and 
                    self.remittance_agents and
                    random.random() < 0.4):  # 40% use agents for large amounts
                    
                    agent_id = random.choice(self.remittance_agents)
                    
                    # First pay the remittance agent fee
                    agent_fee = remittance_amount * random.uniform(0.015, 0.025)  # 1.5-2.5% fee
                    agent_txn = self.log_merchant_transaction(
                        merchant_id=agent_id,
                        amount=agent_fee,
                        description="Money Transfer Agent Fee",
                        date=date,
                        channel="Cash"
                    )
                    if agent_txn:
                        events.append(agent_txn)
                    
                    channel = 'Agent_Transfer'
                else:
                    channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': self.family_recipient, 
                    'amount': round(remittance_amount, 2), 
                    'desc': 'Family Remittance Transfer',
                    'channel': channel,
                    'transaction_category': 'family_remittance'
                })
                self.last_remittance_day = current_day_key

    def _handle_worker_community_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced worker community transfers"""
        if (self.worker_network and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 100):
            
            recipient = random.choice(self.worker_network)
            
            # Small amounts typical for this economic group
            base_amount = random.uniform(50, 400)
            
            # Adjust based on current balance and economic class
            balance_factor = min(self.balance / 500, 1.0)  # Scale by available balance
            economic_multiplier = {
                'Lower': random.uniform(0.7, 1.0),
                'Lower_Middle': random.uniform(0.9, 1.2)
            }.get(self.economic_class, 1.0)
            
            # Don't send more than 25% of balance
            max_sendable = self.balance * 0.25
            final_amount = min(base_amount * balance_factor * economic_multiplier, max_sendable)
            
            if final_amount >= 30:  # Minimum viable transfer
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Worker Community Support',
                    'channel': channel,
                    'transaction_category': 'worker_support'
                })

    def _handle_community_support(self, date, events, context):
        """✅ UPDATED: Enhanced community support transfers"""
        if (self.worker_network and 
            random.random() < self.community_support_chance and 
            self.balance > 150):
            
            recipient = random.choice(self.worker_network)
            
            # Community support amounts based on financial personality
            base_support = random.uniform(80, 250)
            
            # Personality adjustments
            if self.financial_personality == 'Saver':
                base_support *= random.uniform(1.1, 1.4)  # Savers are more community-oriented
            
            # Ensure we don't send more than we can afford
            max_support = self.balance * 0.4  # Maximum 40% for community help
            final_amount = min(base_support, max_support)
            
            if final_amount >= 50:  # Minimum for meaningful support
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Community Mutual Support',
                    'channel': channel,
                    'transaction_category': 'community_support'
                })

    def _handle_emergency_help(self, date, events, context):
        """✅ UPDATED: Enhanced emergency help for fellow workers"""
        if (self.worker_network and 
            random.random() < self.emergency_help_chance and 
            self.balance > 200):
            
            recipient = random.choice(self.worker_network)
            
            # Emergency amounts - larger but still constrained
            emergency_amount = random.uniform(150, 500)
            
            # Adjust based on financial personality
            if self.financial_personality == 'Saver':
                emergency_amount *= random.uniform(1.2, 1.6)
            
            # Can't give more than 50% of balance in emergency
            max_emergency = self.balance * 0.5
            final_amount = min(emergency_amount, max_emergency)
            
            if final_amount >= 100:  # Minimum for emergency
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Emergency Worker Support',
                    'channel': channel,
                    'transaction_category': 'emergency_help'
                })

    def _handle_recharge(self, date, events):
        """✅ UPDATED: Enhanced mobile recharges with merchant tracking"""
        if random.random() < self.recharge_chance:
            # Small recharge amounts typical for this economic group
            recharge_options = [10, 20, 49, 79, 99]
            recharge_amount = random.choice(recharge_options)
            
            recharge_merchant_id = f"mobile_recharge_{hash(self.agent_id) % 100}"
            
            txn = self.log_merchant_transaction(
                merchant_id=recharge_merchant_id,
                amount=recharge_amount,
                description="Basic Mobile Recharge",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def _handle_essential_purchases(self, date, events, context):
        """✅ UPDATED: Enhanced essential purchases from local merchants"""
        # Weekly essential purchases
        if (date.weekday() == 6 and  # Sunday purchases
            random.random() < 0.4 and  # 40% chance
            self.balance > 100):
            
            # Essential purchases - food, basic necessities
            purchase_amount = random.uniform(60, 250)
            
            # Adjust based on economic class
            if self.economic_class == 'Lower_Middle':
                purchase_amount *= random.uniform(1.1, 1.3)
            
            # Limit to what they can afford
            max_purchase = self.balance * 0.35
            final_amount = min(purchase_amount, max_purchase)
            
            if final_amount >= 40:
                # ✅ NEW: Enhanced local merchant tracking
                local_merchant_id = f"local_merchant_{hash(self.agent_id) % 200}"
                self.add_frequent_merchant(local_merchant_id, date)
                
                txn = self.log_merchant_transaction(
                    merchant_id=local_merchant_id,
                    amount=final_amount,
                    description="Weekly Essential Groceries",
                    date=date,
                    channel="Cash"
                )
                if txn:
                    events.append(txn)

    def get_daily_wage_laborer_features(self):
        """✅ ENHANCED: Comprehensive daily wage laborer features"""
        return {
            'contractor_employer_count': len(self.contractors),
            'primary_contractor_tenure': self.get_employment_tenure_months(),
            'work_consistency_score': self.work_consistency,
            'contractor_payment_reliability': self.contractor_payment_reliability,
            'remittance_agent_relationships': len(self.remittance_agents),
            'local_merchant_relationships': len(self.local_merchants),
            'worker_network_size': len(self.worker_network),
            'family_dependency_score': 1 if self.family_recipient else 0,
            'remittance_ratio': self.remittance_percentage,
            'cash_dependency_score': 0.85,  # Very high cash dependency
            'informal_employment_score': 1.0,  # Fully informal
            'last_wage_recency': (datetime.now().date() - self.last_wage_date).days if self.last_wage_date else 999,
            'seasonal_work_vulnerability': 0.8,  # High vulnerability to seasonal changes
            'total_company_relationships': len(self.contractors)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with contractor salary tracking"""
        events = []
        
        # Handle all income sources (including contractor salary tracking)
        self._handle_daily_income(date, events)
        
        # Handle family and community transfers
        self._handle_family_remittances(date, events, context)
        self._handle_worker_community_transfers(date, events, context)
        self._handle_community_support(date, events, context)
        self._handle_emergency_help(date, events, context)
        
        # Handle essential purchases and expenses
        self._handle_essential_purchases(date, events, context)
        self._handle_cash_withdrawals(date, events)
        self._handle_recharge(date, events)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
