import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class MigrantWorker(BaseAgent):
    """
    Enhanced Migrant Worker agent for Phase 2: Contractor salary source tracking
    Includes contractor companies as employers, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Lower', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Migrant Worker"

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
            "income_type": "Wages",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Weekly_or_Monthly",
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
            "device_consistency_score": round(random.uniform(0.35, 0.65), 3),
            "ip_consistency_score": round(random.uniform(0.25, 0.55), 3),
            "sim_churn_rate": random.choice(["High", "Very_High"]),
            "primary_digital_channels": ["UPI", "IMPS"],
            "login_pattern": "Remittance_Cycle",
            "ecommerce_activity_level": "None",
            "ecommerce_avg_ticket_size": "N/A",
            
            # ✅ NEW: Heterogeneous graph connections specific to MigrantWorker
            "industry_sector": "Labor_Migration",
            "company_size": "Small_Contractor",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Contractors as employers (salary source tracking)
        self.contractors = []  # Labor contractors as company nodes
        self.primary_contractor_id = None  # Main work source contractor
        self.contractor_relationships = {}  # Track contractor work patterns

        # ✅ NEW: Employment relationship tracking
        self.work_consistency = random.uniform(0.65, 0.85)  # Work availability
        self.contractor_payment_reliability = random.uniform(0.75, 0.9)  # Payment consistency
        self.last_wage_date = None
        self.migration_tenure = random.randint(6, 60)  # 6 months to 5 years

        # Geographic attributes with enhanced tracking
        self.home_state = random.choice(["Uttar Pradesh", "Bihar", "Odisha", "Rajasthan", "West Bengal", "Jharkhand"])
        self.work_city = random.choice(["Mumbai", "Delhi", "Bengaluru", "Surat", "Pune", "Chennai", "Hyderabad"])

        # Financial calculations with more variation
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_mod, max_mod)
        
        # Pay cycle patterns with enhanced tracking
        self.pay_cycle = "monthly" if economic_class == 'Lower_Middle' else random.choice(["weekly", "bi_weekly"])
        self.weekly_wage = self.monthly_income / random.uniform(4, 4.5)
        self.monthly_pay_day = random.randint(1, 5)
        self.weekly_pay_day = random.randint(5, 6)  # Friday or Saturday
        
        # Remittance patterns with more variation
        remittance_base = random.uniform(0.55, 0.90)
        personality_multiplier = 1.15 if financial_personality == 'Saver' else 1.0
        self.remittance_percentage = remittance_base * personality_multiplier
        self.recharge_chance = random.uniform(0.05, 0.09)

        # ✅ Enhanced P2P networks for Migrant Workers
        self.family_back_home = []  # Family in home state for remittances
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.contractors = []  # Contractors and employers as company nodes
        self.remittance_agents = []  # Money transfer agents as merchant nodes
        self.fellow_migrants = []  # Other migrant workers from same region
        self.local_merchants = []  # Local shops in work city
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.25, 0.35) * personality_config.get('spend_chance_mod', 1.0)
        self.emergency_transfer_chance = random.uniform(0.03, 0.07)
        self.festival_months = [3, 10, 11]  # Holi, Diwali, etc.
        
        # Temporal tracking with enhanced features
        self.last_remittance_date = None
        self.migration_cycles = []  # Track work migration patterns
        self.seasonal_employment_patterns = []  # Track seasonal work

        self.balance = random.uniform(50, 600)  # Very low balance range

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Migrant workers typically have 1-2 devices (basic phone, sometimes backup)"""
        device_options = [1, 2]
        weights = [0.6, 0.4]  # Most have just 1 device
        return random.choices(device_options, weights=weights)[0]

    def assign_contractors(self, contractor_company_ids):
        """✅ NEW: Assign contractor companies as employers for salary tracking"""
        self.contractors = contractor_company_ids
        
        if contractor_company_ids:
            # Assign primary contractor as main employer
            self.primary_contractor_id = random.choice(contractor_company_ids)
            self.assign_employer(
                company_id=self.primary_contractor_id,
                employment_start_date=datetime.now().date() - timedelta(days=self.migration_tenure * 30)
            )
            
            # Set up contractor relationships
            for contractor_id in contractor_company_ids:
                self.contractor_relationships[contractor_id] = {
                    'work_type': random.choice(['Construction', 'Factory', 'Loading', 'Housekeeping', 'Agriculture']),
                    'payment_reliability': random.uniform(0.7, 0.9),
                    'wage_rate': random.uniform(250, 600)
                }

    def _handle_contractor_wage_payment(self, date, events):
        """✅ NEW: Handle wage payments from contractor companies"""
        is_payday = False
        wage_amount = 0

        # Determine if it's payday based on cycle
        if self.pay_cycle == "weekly" and date.weekday() == self.weekly_pay_day:
            is_payday = True
            wage_amount = self.weekly_wage
        elif self.pay_cycle == "bi_weekly" and date.weekday() == self.weekly_pay_day and date.day <= 15:
            is_payday = True
            wage_amount = self.monthly_income / 2
        elif self.pay_cycle == "monthly" and date.day == self.monthly_pay_day:
            is_payday = True
            wage_amount = self.monthly_income

        if (is_payday and 
            random.random() < self.contractor_payment_reliability):
            
            # Add variation to wage amount
            wage_amount *= random.uniform(0.9, 1.1)
            
            # ✅ NEW: 85% from contractors, 15% direct cash work
            if (self.contractors and 
                random.random() < 0.85):
                
                contractor_id = self.primary_contractor_id or random.choice(self.contractors)
                
                # Add seasonal variations for migrant work
                month_multiplier = {
                    12: 0.8,  # December - construction slowdown
                    1: 0.7,   # January - winter low
                    2: 0.8,   # February - still winter
                    4: 1.2,   # April - construction peak
                    5: 1.1,   # May - summer work
                    10: 1.3,  # October - post-monsoon construction boom
                    11: 1.2,  # November - peak season
                }.get(date.month, 1.0)
                
                final_wage = wage_amount * month_multiplier
                
                # ✅ NEW: Log as salary transaction from contractor
                txn = self.log_salary_transaction(
                    amount=final_wage,
                    date=date,
                    company_id=contractor_id
                )
                
                if txn:
                    txn['transaction_category'] = 'migrant_contractor_wage'
                    txn['company_type'] = 'labor_contractor'
                    txn['work_location'] = self.work_city
                    events.append(txn)
                    self.last_wage_date = date
            else:
                # Direct cash work without formal contractor
                txn = self.log_transaction(
                    "CREDIT", f"Daily Cash Work ({self.work_city})", wage_amount, date, channel="Cash_Deposit"
                )
                if txn:
                    events.append(txn)
                    self.last_wage_date = date

    def _handle_income(self, date, events):
        """✅ UPDATED: Enhanced income handling with contractor tracking"""
        self._handle_contractor_wage_payment(date, events)

    def _handle_cash_withdrawals(self, date, events):
        """✅ UPDATED: Enhanced cash withdrawals for daily expenses"""
        if (self.balance > 300 and 
            random.random() < 0.4):  # 40% chance of cash withdrawal
            
            # Cash out most of remaining balance after remittances for daily survival
            cash_out_percentage = random.uniform(0.5, 0.8)
            cash_out_amount = self.balance * cash_out_percentage
            
            if cash_out_amount > 100:
                txn = self.log_transaction(
                    "DEBIT", f"Daily Cash Withdrawal ({self.work_city})", cash_out_amount, date, channel="ATM"
                )
                if txn:
                    events.append(txn)

    def add_remittance_agent(self, agent_id, first_transaction_date=None):
        """✅ NEW: Track remittance agent relationships"""
        if agent_id not in self.remittance_agents:
            self.remittance_agents.append(agent_id)
            self.add_frequent_merchant(agent_id, first_transaction_date)

    def _handle_regular_remittances(self, date, events, context):
        """✅ UPDATED: Enhanced regular remittances with agent tracking"""
        # Send money home after payday (within 2-3 days of receiving wage)
        is_remittance_window = False
        
        if self.pay_cycle == "weekly":
            # Send within 2 days of weekly payday
            days_since_payday = (date.weekday() - self.weekly_pay_day) % 7
            is_remittance_window = days_since_payday <= 2
        elif self.pay_cycle == "bi_weekly":
            # Send within 2 days of bi-weekly payday
            if date.day <= 17 and date.weekday() == (self.weekly_pay_day + 2) % 7:
                is_remittance_window = True
        elif self.pay_cycle == "monthly":
            # Send within 3 days of monthly payday
            if date.day >= self.monthly_pay_day and date.day <= self.monthly_pay_day + 3:
                is_remittance_window = True

        if (is_remittance_window and 
            self.family_back_home and
            self.balance >= 400):  # Minimum balance for remittance
            
            # Check if already sent this cycle
            current_date_key = f"{date.strftime('%Y-%m')}-{self.pay_cycle}"
            if self.last_remittance_date != current_date_key:
                recipient = random.choice(self.family_back_home)
                
                # Calculate remittance based on available balance and percentage
                base_remittance = self.balance * self.remittance_percentage
                
                # Adjust for economic pressures
                economic_multiplier = {
                    'Lower': random.uniform(0.85, 1.0),
                    'Lower_Middle': random.uniform(0.9, 1.1)
                }.get(self.economic_class, 1.0)
                
                available_for_remittance = base_remittance * economic_multiplier
                
                if available_for_remittance >= 500:  # Minimum threshold
                    # ✅ NEW: Use remittance agent for larger amounts
                    if (available_for_remittance > 3000 and 
                        self.remittance_agents and
                        random.random() < 0.6):  # 60% use agents for large amounts
                        
                        agent_id = random.choice(self.remittance_agents)
                        
                        # Pay remittance agent fee first
                        agent_fee = available_for_remittance * random.uniform(0.012, 0.018)  # 1.2-1.8% fee
                        agent_txn = self.log_merchant_transaction(
                            merchant_id=agent_id,
                            amount=agent_fee,
                            description=f"Remittance Agent Fee ({self.home_state})",
                            date=date,
                            channel="Cash"
                        )
                        if agent_txn:
                            events.append(agent_txn)
                        
                        channel = 'Agent_Transfer'
                    else:
                        # ✅ NEW: Select realistic channel based on amount
                        if available_for_remittance > 50000:
                            channel = random.choice(['IMPS', 'NEFT'])
                        else:
                            channel = RealisticP2PStructure.select_realistic_channel()
                    
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': recipient, 
                        'amount': round(available_for_remittance, 2), 
                        'desc': f'Family Remittance ({self.home_state})',
                        'channel': channel,
                        'transaction_category': 'family_remittance'
                    })
                    self.last_remittance_date = current_date_key

    def _handle_additional_family_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced additional family transfers with festival adjustments"""
        if (self.family_back_home and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 600):
            
            recipient = random.choice(self.family_back_home)
            
            # Base amount for non-remittance transfers
            base_amount = random.uniform(300, 1200)
            
            # Economic class adjustments
            if self.economic_class == 'Lower_Middle':
                base_amount *= random.uniform(1.1, 1.3)
            
            # Increase during festival months
            if date.month in self.festival_months:
                base_amount *= random.uniform(1.4, 2.2)
            
            # Only send if sufficient balance
            if self.balance > base_amount + 300:  # Keep buffer for survival
                # ✅ NEW: Select realistic channel
                if base_amount > 50000:
                    channel = random.choice(['IMPS', 'NEFT'])
                else:
                    channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(base_amount, 2), 
                    'desc': f'Festival Support ({self.home_state})',
                    'channel': channel,
                    'transaction_category': 'festival_support'
                })

    def _handle_emergency_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced emergency transfers with realistic constraints"""
        if (self.family_back_home and 
            random.random() < self.emergency_transfer_chance and 
            self.balance > 1200):  # Need significant balance for emergency
            
            recipient = random.choice(self.family_back_home)
            # Emergency transfers are typically larger but constrained by balance
            emergency_percentage = random.uniform(0.4, 0.8)
            emergency_amount = self.balance * emergency_percentage
            
            # Adjust based on financial personality
            if self.financial_personality == 'Saver':
                emergency_amount *= random.uniform(1.1, 1.4)
            
            # ✅ NEW: Select realistic channel based on amount
            if emergency_amount > 75000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(emergency_amount, 2), 
                'desc': f'Emergency Family Support ({self.home_state})',
                'channel': channel,
                'transaction_category': 'emergency_transfer'
            })

    def _handle_fellow_migrant_support(self, date, events, context):
        """✅ UPDATED: Enhanced support for fellow migrant workers from same region"""
        if (self.fellow_migrants and 
            random.random() < 0.12 and  # 12% chance of helping fellow migrants
            self.balance > 600):
            
            recipient = random.choice(self.fellow_migrants)
            
            # Small amounts for fellow migrant support
            support_amount = random.uniform(150, 600)
            
            # Adjust based on shared regional background
            if hasattr(recipient, 'home_state') and recipient.home_state == self.home_state:
                support_amount *= random.uniform(1.2, 1.5)  # More support for same state
            
            # Ensure affordability
            max_support = self.balance * 0.25
            final_amount = min(support_amount, max_support)
            
            if final_amount >= 100:  # Minimum support amount
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': recipient, 
                    'amount': round(final_amount, 2), 
                    'desc': f'Fellow Migrant Support ({self.work_city})',
                    'channel': channel,
                    'transaction_category': 'migrant_support'
                })

    def _handle_recharge(self, date, events):
        """✅ UPDATED: Enhanced mobile recharges with merchant tracking"""
        if random.random() < self.recharge_chance:
            # Small recharge amounts typical for migrant workers
            recharge_options = [49, 79, 99, 149]
            recharge_amount = random.choice(recharge_options)
            
            recharge_merchant_id = f"telecom_recharge_{hash(self.agent_id) % 100}"
            
            txn = self.log_merchant_transaction(
                merchant_id=recharge_merchant_id,
                amount=recharge_amount,
                description=f"Mobile Recharge ({self.work_city})",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def _handle_essential_purchases(self, date, events, context):
        """✅ UPDATED: Enhanced essential purchases from local merchants"""
        # Weekly essential purchases in work city
        if (date.weekday() == 6 and  # Sunday purchases
            random.random() < 0.3 and  # 30% chance
            self.balance > 400):
            
            # Essential purchases in work city
            purchase_amount = random.uniform(120, 500)
            
            # Adjust based on economic class
            if self.economic_class == 'Lower_Middle':
                purchase_amount *= random.uniform(1.1, 1.3)
            
            # Limit to what they can afford
            max_purchase = self.balance * 0.3
            final_amount = min(purchase_amount, max_purchase)
            
            if final_amount >= 80:
                # ✅ NEW: Enhanced local merchant tracking
                local_merchant_id = f"migrant_merchant_{self.work_city.lower()}_{hash(self.agent_id) % 200}"
                self.add_frequent_merchant(local_merchant_id, date)
                
                txn = self.log_merchant_transaction(
                    merchant_id=local_merchant_id,
                    amount=final_amount,
                    description=f"Weekly Essentials ({self.work_city})",
                    date=date,
                    channel="Cash"
                )
                if txn:
                    events.append(txn)

    def get_migrant_worker_features(self):
        """✅ ENHANCED: Comprehensive migrant worker features"""
        return {
            'contractor_employer_count': len(self.contractors),
            'primary_contractor_tenure': self.get_employment_tenure_months(),
            'migration_tenure_months': self.migration_tenure,
            'work_consistency_score': self.work_consistency,
            'contractor_payment_reliability': self.contractor_payment_reliability,
            'remittance_agent_relationships': len(self.remittance_agents),
            'local_merchant_relationships': len(self.local_merchants),
            'fellow_migrant_network_size': len(self.fellow_migrants),
            'family_dependency_score': len(self.family_back_home),
            'work_city': self.work_city,
            'home_state': self.home_state,
            'geographic_mobility_score': 1.0,  # High mobility due to migration
            'remittance_ratio': self.remittance_percentage,
            'pay_cycle_frequency': self.pay_cycle,
            'informal_employment_score': 1.0,  # Fully informal
            'cash_dependency_score': 0.9,  # Very high cash dependency
            'last_wage_recency': (datetime.now().date() - self.last_wage_date).days if self.last_wage_date else 999,
            'interstate_migration_indicator': 1.0,  # Always interstate migrants
            'total_company_relationships': len(self.contractors)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with contractor salary tracking"""
        events = []
        
        # Handle all income sources (including contractor salary tracking)
        self._handle_income(date, events)
        
        # Handle family and community transfers
        self._handle_regular_remittances(date, events, context)
        self._handle_additional_family_transfers(date, events, context)
        self._handle_emergency_transfers(date, events, context)
        self._handle_fellow_migrant_support(date, events, context)
        
        # Handle essential purchases and expenses
        self._handle_essential_purchases(date, events, context)
        self._handle_cash_withdrawals(date, events)
        self._handle_recharge(date, events)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
