import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class Student(BaseAgent):
    """
    Enhanced Student agent for Phase 2: Educational institution salary source tracking
    Includes educational institutions as allowance sources, realistic student relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Over_Spender'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Student"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "3000-10000"
        min_allowance, max_allowance = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_allowance * income_multiplier)}-{int(max_allowance * income_multiplier)}"
        
        profile_attributes = {
            "archetype_name": archetype_name, 
            "risk_profile": risk_profile_category, 
            "risk_score": risk_score,
            "economic_class": economic_class, 
            "financial_personality": financial_personality,
            "employment_status": "Not_Applicable", 
            "employment_verification": "Not_Applicable",
            "income_type": "Allowance", 
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Irregular", 
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False, 
            "investment_types": [],
            "has_loan_emi": False, 
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False, 
            "insurance_types": [],
            "utility_payment_status": "N/A", 
            "mobile_plan_type": "Prepaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.68, 0.87), 3),
            "ip_consistency_score": round(random.uniform(0.35, 0.65), 3),
            "sim_churn_rate": random.choice(["Medium", "High"]),
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Late_Night_Activity", 
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "Low",
            
            # ✅ NEW: Heterogeneous graph connections specific to Student
            "industry_sector": "Education_Student",
            "company_size": "Not_Applicable",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Educational institutions as allowance sources (salary source tracking)
        self.educational_institutions = []  # Colleges, universities as company nodes
        self.primary_institution_id = None  # Main educational institution
        self.institution_hierarchy = {}  # Track scholarship types and allowance sources

        # ✅ NEW: Student relationship tracking
        self.student_tenure = random.randint(6, 48)  # 6 months to 4 years
        self.allowance_consistency = random.uniform(0.6, 0.85)  # Irregular allowance pattern
        self.last_allowance_date = None
        self.scholarship_patterns = {}

        # Financial calculations with more variation
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.allowance_amount = random.uniform(min_mod, max_mod)
        self.allowance_days = sorted(random.sample(range(2, 28), random.randint(1, 3)))

        # Spending patterns with more variation
        self.daily_spend_chance = random.uniform(0.70, 0.80) * personality_config.get('spend_chance_mod', 1.0)
        self.recharge_chance = random.uniform(0.06, 0.10)
        self.bnpl_chance = random.uniform(0.12, 0.18) if financial_personality == 'Over_Spender' else random.uniform(0.03, 0.07)
        
        # ✅ Enhanced P2P networks for Students
        self.contacts = []  # General friend network
        self.study_group = []  # Study group members for academic expenses
        self.hostel_friends = []  # Hostel/roommate network for shared expenses
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.educational_institutions = []  # Schools, colleges as company nodes
        self.student_service_providers = []  # Canteens, stationery shops as merchant nodes
        self.family_members = []  # Family for allowance source tracking
        self.online_platforms = []  # OTT, gaming, food delivery as merchant nodes
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.22, 0.28) * personality_config.get('spend_chance_mod', 1.0)
        self.study_group_transfer_chance = random.uniform(0.12, 0.18)  # Academic-related transfers
        self.hostel_sharing_chance = random.uniform(0.15, 0.21)  # Shared hostel expenses
        self.emergency_help_chance = random.uniform(0.06, 0.10)  # Emergency peer support
        
        # Temporal patterns with more variation
        self.exam_months = [4, 10, 11]  # Exam seasons
        self.festival_months = [3, 10, 11]  # Festival seasons
        
        # Temporal tracking with enhanced features
        self.last_allowance_date = None
        self.academic_cycles = []  # Track semester/exam patterns
        self.social_activity_patterns = []

        # BNPL tracking with enhanced features
        self.bnpl_repayments = {}
        self.active_bnpl_count = 0
        self.max_bnpl_limit = random.uniform(2000, 8000)
        
        self.balance = random.uniform(50, 600)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Students typically have 1-3 devices (phone, sometimes laptop/tablet)"""
        device_options = [1, 2, 3]
        weights = [0.3, 0.5, 0.2]  # Most have 2 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_educational_institutions(self, institution_company_ids):
        """✅ NEW: Assign educational institutions as allowance sources for salary tracking"""
        self.educational_institutions = institution_company_ids
        
        if institution_company_ids:
            # Assign primary institution as main allowance source
            self.primary_institution_id = random.choice(institution_company_ids)
            self.assign_employer(
                company_id=self.primary_institution_id,
                employment_start_date=datetime.now().date() - timedelta(days=self.student_tenure * 30)
            )
            
            # Set up institution hierarchy
            for inst_id in institution_company_ids:
                self.institution_hierarchy[inst_id] = {
                    'allowance_type': random.choice(['Merit_Scholarship', 'Need_Based_Aid', 'Research_Stipend', 'Family_Support']),
                    'allowance_multiplier': random.uniform(0.7, 1.5),  # Multiplier for base allowance
                    'payment_reliability': random.uniform(0.6, 0.9)
                }

    def _handle_educational_institution_allowance_payment(self, date, events):
        """✅ NEW: Handle allowance from educational institutions/family"""
        if (date.day in self.allowance_days and 
            random.random() < self.allowance_consistency):
            
            institution_id = self.primary_institution_id or (
                random.choice(self.educational_institutions) if self.educational_institutions else None
            )
            
            # Calculate allowance with student patterns
            base_allowance = self.allowance_amount
            
            # Add institution-specific variations
            if institution_id:
                allowance_multiplier = self.institution_hierarchy.get(institution_id, {}).get('allowance_multiplier', 1.0)
                base_allowance *= allowance_multiplier
            
            # Add seasonal variations (lower during holidays)
            month_multiplier = {
                12: 0.7,  # December - holiday reduction
                1: 0.8,   # January - post-holiday
                5: 0.9,   # May - summer break
                6: 0.8,   # June - summer break
            }.get(date.month, 1.0)
            
            final_allowance = base_allowance * month_multiplier
            
            # ✅ NEW: Log as salary transaction from educational institution
            if institution_id and random.random() < 0.4:  # 40% from institution, 60% from family
                txn = self.log_salary_transaction(
                    amount=final_allowance,
                    date=date,
                    company_id=institution_id
                )
                if txn:
                    txn['transaction_category'] = 'educational_allowance'
                    txn['company_type'] = 'educational_institution'
                    txn['allowance_type'] = self.institution_hierarchy.get(institution_id, {}).get('allowance_type', 'Unknown')
                    events.append(txn)
            else:
                # Family allowance (most common)
                if self.family_members:
                    family_member = random.choice(self.family_members)
                    txn = self.log_salary_transaction(
                        amount=final_allowance,
                        date=date,
                        company_id=family_member
                    )
                    if txn:
                        txn['transaction_category'] = 'family_allowance'
                        events.append(txn)
                else:
                    txn = self.log_transaction(
                        "CREDIT", "Family Allowance", final_allowance, date, channel="P2P"
                    )
                    if txn:
                        events.append(txn)
            
            self.last_allowance_date = date
            return final_allowance
        
        return 0

    def _handle_income(self, date, events):
        """✅ UPDATED: Enhanced allowance handling with institution tracking"""
        self._handle_educational_institution_allowance_payment(date, events)

    def add_student_service_provider(self, provider_id, first_transaction_date=None):
        """✅ NEW: Track student service provider relationships"""
        if provider_id not in self.student_service_providers:
            self.student_service_providers.append(provider_id)
            self.add_frequent_merchant(provider_id, first_transaction_date)

    def _handle_spending(self, date, events):
        """✅ UPDATED: Enhanced spending with merchant tracking"""
        # Handle BNPL repayments with enhanced tracking
        if date.date() in self.bnpl_repayments:
            amount_due = self.bnpl_repayments.pop(date.date())
            self.active_bnpl_count = max(0, self.active_bnpl_count - 1)
            
            bnpl_provider_id = f"bnpl_provider_{hash(self.agent_id) % 100}"
            
            txn = self.log_merchant_transaction(
                merchant_id=bnpl_provider_id,
                amount=amount_due,
                description="Student BNPL Repayment",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

        # Handle mobile recharge with enhanced patterns
        if random.random() < self.recharge_chance:
            recharge_options = [99, 149, 199, 239, 299]
            recharge_amount = random.choice(recharge_options)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                recharge_amount = random.choice([199, 299, 399])
            elif self.economic_class == 'Lower':
                recharge_amount = random.choice([79, 99, 149])
            
            telecom_provider_id = f"student_telecom_{hash(self.agent_id) % 50}"
            
            txn = self.log_merchant_transaction(
                merchant_id=telecom_provider_id,
                amount=recharge_amount,
                description="Student Mobile Recharge",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

        # Handle regular spending with enhanced categories
        if random.random() < self.daily_spend_chance:
            spending_categories = [
                ("Food_Delivery", random.uniform(80, 400)),
                ("Cab_Service", random.uniform(50, 250)),
                ("OTT_Subscription", random.uniform(99, 299)),
                ("Groceries", random.uniform(120, 600)),
                ("Gaming_Purchase", random.uniform(50, 500)),
                ("Stationery", random.uniform(30, 200)),
                ("Coffee_Snacks", random.uniform(40, 180)),
                ("Movies_Entertainment", random.uniform(150, 500))
            ]
            
            spend_category, spend_amount = random.choice(spending_categories)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                spend_amount *= random.uniform(1.2, 1.8)
            elif self.economic_class == 'Lower':
                spend_amount *= random.uniform(0.6, 0.8)
            
            # BNPL decision with enhanced logic
            use_bnpl = (
                random.random() < self.bnpl_chance and 
                spend_amount > 200 and  # Minimum for BNPL
                self.active_bnpl_count < 3 and  # Max 3 active BNPL
                spend_category in ["Food_Delivery", "Gaming_Purchase", "Movies_Entertainment"]
            )
            
            if use_bnpl:
                # BNPL transaction - schedule future repayment
                repayment_date = date.date() + timedelta(days=random.randint(14, 30))
                self.bnpl_repayments[repayment_date] = self.bnpl_repayments.get(repayment_date, 0) + spend_amount
                self.active_bnpl_count += 1
                
                # ✅ NEW: Enhanced BNPL provider tracking
                bnpl_provider_id = f"student_bnpl_{spend_category}_{hash(self.agent_id + str(date)) % 200}"
                self.add_student_service_provider(bnpl_provider_id, date)
                
                # Log as BNPL transaction
                txn = self.log_merchant_transaction(
                    merchant_id=bnpl_provider_id,
                    amount=0,  # No immediate charge
                    description=f"Student BNPL - {spend_category.replace('_', ' ')}",
                    date=date,
                    channel="BNPL"
                )
                if txn:
                    events.append(txn)
            else:
                # Regular payment
                service_provider_id = f"student_{spend_category}_{hash(self.agent_id + str(date)) % 300}"
                self.add_student_service_provider(service_provider_id, date)
                
                txn = self.log_merchant_transaction(
                    merchant_id=service_provider_id,
                    amount=spend_amount,
                    description=f"Student {spend_category.replace('_', ' ')}",
                    date=date,
                    channel="UPI"
                )
                if txn:
                    events.append(txn)

    def _handle_peer_group_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced peer group transfers with realistic channels"""
        if (self.contacts and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 300):
            
            recipient = random.choice(self.contacts)
            
            # Students typically send smaller amounts
            base_amount = random.uniform(100, 600)
            
            # Economic class adjustments
            economic_multiplier = {
                'Lower': random.uniform(0.5, 0.8),
                'Lower_Middle': random.uniform(0.8, 1.1),
                'Middle': random.uniform(1.0, 1.3),
                'Upper_Middle': random.uniform(1.3, 1.8),
                'High': random.uniform(1.6, 2.2)
            }.get(self.economic_class, 1.0)
            
            # Financial personality adjustments
            if self.financial_personality == 'Over_Spender':
                base_amount *= random.uniform(1.2, 1.6)
            elif self.financial_personality == 'Saver':
                base_amount *= random.uniform(0.7, 1.0)
            
            final_amount = base_amount * economic_multiplier
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(final_amount, 2), 
                'desc': 'Student Peer Transfer',
                'channel': channel,
                'transaction_category': 'peer_transfer'
            })

    def _handle_study_group_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced academic-related transfers"""
        if (self.study_group and 
            random.random() < self.study_group_transfer_chance and
            self.balance > 200):
            
            study_mate = random.choice(self.study_group)
            
            # Academic expenses are typically small but frequent
            academic_amount = random.uniform(50, 300)
            
            # Higher amounts during exam months
            if date.month in self.exam_months:
                academic_amount *= random.uniform(1.4, 2.2)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                academic_amount *= random.uniform(1.2, 1.6)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': study_mate, 
                'amount': round(academic_amount, 2), 
                'desc': 'Study Group Academic Expense',
                'channel': channel,
                'transaction_category': 'academic_expense'
            })

    def _handle_hostel_sharing_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced shared hostel/accommodation expenses"""
        if (self.hostel_friends and 
            random.random() < self.hostel_sharing_chance and
            self.balance > 250):
            
            hostel_mate = random.choice(self.hostel_friends)
            
            # Shared expenses (food, utilities, cleaning, etc.)
            shared_amount = random.uniform(120, 600)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                shared_amount *= random.uniform(1.3, 1.8)
            elif self.economic_class == 'Lower':
                shared_amount *= random.uniform(0.7, 0.9)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': hostel_mate, 
                'amount': round(shared_amount, 2), 
                'desc': 'Hostel Shared Expense',
                'channel': channel,
                'transaction_category': 'hostel_sharing'
            })

    def _handle_emergency_peer_support(self, date, events, context):
        """✅ UPDATED: Enhanced emergency support within student network"""
        if (self.contacts and 
            random.random() < self.emergency_help_chance and
            self.balance > 400):  # Need decent balance for emergency help
            
            friend = random.choice(self.contacts)
            
            # Emergency amounts for students
            emergency_amount = random.uniform(200, 800)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                emergency_amount *= random.uniform(1.4, 2.2)
            
            # Personality adjustments
            if self.financial_personality == 'Over_Spender':
                emergency_amount *= random.uniform(1.2, 1.6)  # More generous
            
            # Can't give more than 35% of balance in emergency
            max_emergency = self.balance * 0.35
            final_amount = min(emergency_amount, max_emergency)
            
            if final_amount >= 100:  # Minimum for meaningful emergency help
                # ✅ NEW: Select realistic channel
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': friend, 
                    'amount': round(final_amount, 2), 
                    'desc': 'Student Emergency Support',
                    'channel': channel,
                    'transaction_category': 'emergency_support'
                })

    def _handle_festival_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced festival-related transfers"""
        if (date.month in self.festival_months and 
            date.day <= 5 and
            self.contacts and 
            random.random() < (self.p2p_transfer_chance * 2.2) and
            self.balance > 200):
            
            friend = random.choice(self.contacts)
            
            # Festival transfers are typically smaller gifts
            festival_amount = random.uniform(150, 500)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                festival_amount *= random.uniform(1.3, 2.0)
            elif self.economic_class == 'Lower':
                festival_amount *= random.uniform(0.7, 0.9)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': friend, 
                'amount': round(festival_amount, 2), 
                'desc': 'Festival Gift Transfer',
                'channel': channel,
                'transaction_category': 'festival_transfer'
            })

    def _handle_educational_expenses(self, date, events, context):
        """✅ UPDATED: Enhanced educational institution payments"""
        if (random.random() < 0.08 and  # 8% chance of educational expense
            self.educational_institutions and
            self.balance > 800):
            
            # Educational expenses (fees, books, supplies)
            expense_categories = [
                ("Course_Fees", random.uniform(1000, 5000)),
                ("Books_Supplies", random.uniform(300, 1500)),
                ("Lab_Fees", random.uniform(500, 2000)),
                ("Exam_Fees", random.uniform(200, 800))
            ]
            
            category, expense_amount = random.choice(expense_categories)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                expense_amount *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Enhanced educational institution tracking
            institution_id = random.choice(self.educational_institutions)
            
            txn = self.log_merchant_transaction(
                merchant_id=institution_id,
                amount=expense_amount,
                description=f"Educational {category.replace('_', ' ')}",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

    def get_student_specific_features(self):
        """✅ ENHANCED: Comprehensive student-specific features"""
        return {
            'educational_institution_employer_count': len(self.educational_institutions),
            'primary_institution_tenure': self.get_employment_tenure_months(),
            'student_experience_months': self.student_tenure,
            'service_provider_relationships': len(self.student_service_providers),
            'peer_network_size': len(self.contacts),
            'study_group_size': len(self.study_group),
            'hostel_network_size': len(self.hostel_friends),
            'family_dependency_count': len(self.family_members),
            'allowance_irregularity_score': 1.0 - (len(self.allowance_days) / 30.0),
            'bnpl_usage_score': self.bnpl_chance,
            'active_bnpl_obligations': self.active_bnpl_count,
            'social_activity_level': self.p2p_transfer_chance,
            'academic_support_activity': self.study_group_transfer_chance,
            'last_allowance_recency': (datetime.now().date() - self.last_allowance_date).days if self.last_allowance_date else 999,
            'spending_impulsivity_score': self.daily_spend_chance,
            'total_company_relationships': len(self.educational_institutions)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with educational institution allowance tracking"""
        events = []
        
        # Handle all income sources (including educational institution allowance tracking)
        self._handle_income(date, events)
        
        # Handle spending and BNPL
        self._handle_spending(date, events)
        
        # Handle P2P transfers
        self._handle_peer_group_transfers(date, events, context)
        self._handle_study_group_transfers(date, events, context)
        self._handle_hostel_sharing_transfers(date, events, context)
        self._handle_emergency_peer_support(date, events, context)
        self._handle_festival_transfers(date, events, context)
        
        # Handle educational expenses
        self._handle_educational_expenses(date, events, context)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
