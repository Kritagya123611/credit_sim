import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class Homemaker(BaseAgent):
    """
    Enhanced Homemaker agent for Phase 2: Household support and family allowance tracking
    Includes household head relationships as income sources, realistic family behaviors,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Homemaker"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "10000-30000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Not_Applicable",
            "employment_verification": "Not_Applicable",
            "income_type": "Family_Support",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False,
            "investment_types": [],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Child_Education_Plan"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Prepaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.55, 0.85), 3),
            "ip_consistency_score": round(random.uniform(0.95, 0.99), 3),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["UPI", "Mobile_Banking"],
            "login_pattern": "Infrequent",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Medium",
            
            # ✅ NEW: Heterogeneous graph connections specific to Homemaker
            "industry_sector": "Household_Domestic",
            "company_size": "Not_Applicable",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Household head as income source (salary source tracking)
        self.household_head_id = None  # Primary income earner as "employer"
        self.household_relationship_type = None  # spouse, parent, etc.
        self.allowance_schedule = {}  # Track allowance patterns

        # ✅ NEW: Household relationship tracking
        self.allowance_consistency = random.uniform(0.95, 1.0)  # Very consistent allowances
        self.monthly_allowance_day = random.randint(1, 5)  # Beginning of month
        self.last_allowance_date = None

        # Financial calculations with more variation
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_allowance = random.uniform(min_mod, max_mod)

        # Fixed expenses with enhanced tracking
        self.loan_emi_amount = self.monthly_allowance * random.uniform(0.25, 0.35)
        self.insurance_premium = self.monthly_allowance * random.uniform(0.08, 0.12)
        self.utility_bill_amount = self.monthly_allowance * random.uniform(0.12, 0.18)
        self.weekly_grocery_day = random.randint(4, 6)  # Friday to Sunday
        self.school_fee_months = [1, 4, 7, 10]
        self.occasional_spend_chance = random.uniform(0.06, 0.10) * personality_config.get('spend_chance_mod', 1.0)
        self.shared_device_id = None

        # ✅ Enhanced P2P networks for Homemakers
        self.social_circle = []  # Community and friends
        self.extended_family = []  # Extended family members
        self.children_contacts = []  # School/education related contacts
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.household_head_id = None  # Primary income earner (company node or agent)
        self.children_schools = []  # School institutions as company nodes
        self.local_merchants = []  # Grocery stores, pharmacies, etc.
        self.community_groups = []  # Social/religious groups as special nodes
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.15, 0.21) * personality_config.get('spend_chance_mod', 1.0)
        self.social_support_chance = random.uniform(0.10, 0.14)
        self.family_help_chance = random.uniform(0.06, 0.10)
        self.children_expense_chance = random.uniform(0.12, 0.18)
        
        # Temporal patterns with enhanced seasonality
        self.festival_months = [3, 10, 11]  # Festival seasons
        self.school_activity_months = [6, 12]  # School event seasons
        
        # Temporal tracking with enhanced features
        self.last_allowance_date = None
        self.seasonal_spending_patterns = []
        self.community_activity_cycles = []

        self.balance = random.uniform(self.monthly_allowance * 0.03, self.monthly_allowance * 0.25)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Homemakers typically have 1-2 devices (phone, sometimes tablet for children)"""
        device_options = [1, 2, 3]
        weights = [0.5, 0.4, 0.1]  # Most have 1-2 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_household_head(self, head_company_id, relationship_type='spouse'):
        """✅ NEW: Assign household head as income source for allowance tracking"""
        self.household_head_id = head_company_id
        self.household_relationship_type = relationship_type
        
        # Set up allowance schedule
        self.allowance_schedule = {
            'payout_day': self.monthly_allowance_day,
            'consistency': self.allowance_consistency,
            'relationship_type': relationship_type
        }
        
        # Assign as employer for salary tracking
        self.assign_employer(
            company_id=head_company_id,
            employment_start_date=datetime.now().date() - timedelta(days=random.randint(365, 3650))
        )

    def add_children_school(self, school_company_id, enrollment_date=None):
        """✅ NEW: Track children's school relationships as company nodes"""
        if school_company_id not in self.children_schools:
            self.children_schools.append(school_company_id)
            if enrollment_date:
                self.relationship_start_dates[f'school_{school_company_id}'] = enrollment_date

    def add_community_group(self, group_id, join_date=None):
        """✅ NEW: Track community group memberships"""
        if group_id not in self.community_groups:
            self.community_groups.append(group_id)
            if join_date:
                self.relationship_start_dates[f'community_{group_id}'] = join_date

    def _handle_household_allowance_payment(self, date, events):
        """✅ NEW: Handle monthly allowance from household head"""
        if (date.day == self.monthly_allowance_day and 
            random.random() < self.allowance_consistency):
            
            # Calculate allowance with slight variation
            base_allowance = self.monthly_allowance
            
            # Add occasional bonus (festivals, special occasions)
            if date.month in self.festival_months:
                base_allowance *= random.uniform(1.2, 1.8)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                base_allowance *= random.uniform(1.1, 1.4)
            
            # ✅ NEW: Log as salary transaction from household head
            if self.household_head_id:
                txn = self.log_salary_transaction(
                    amount=base_allowance,
                    date=date,
                    company_id=self.household_head_id
                )
                if txn:
                    txn['transaction_category'] = 'household_allowance'
                    txn['company_type'] = 'household_head'
                    txn['relationship_type'] = self.household_relationship_type
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", "Family Support Transfer", base_allowance, date, channel="P2P"
                )
                if txn:
                    events.append(txn)
            
            self.last_allowance_date = date
            return base_allowance
        
        return 0

    def _handle_monthly_income_and_fixed_costs(self, date, events):
        """✅ UPDATED: Enhanced monthly income and expenses with company tracking"""
        # Household allowance payment
        allowance_amount = self._handle_household_allowance_payment(date, events)

        # EMI payment with variation
        emi_day = random.randint(8, 12)
        if self.has_loan_emi and date.day == emi_day:
            # Add slight variation to EMI amount
            emi_variation = random.uniform(0.98, 1.02)
            actual_emi = self.loan_emi_amount * emi_variation
            
            loan_merchant_id = f"family_loan_bank_{hash(self.agent_id) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=loan_merchant_id,
                amount=actual_emi,
                description="Home/Car Loan EMI (Family Co-payment)",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)
            
        # Utility bills with enhanced tracking
        utility_day = random.randint(13, 17)
        if date.day == utility_day:
            # Add variation to utility amount
            utility_variation = random.uniform(0.9, 1.1)
            actual_utility = self.utility_bill_amount * utility_variation
            
            utility_merchant_id = f"household_utility_{hash(self.agent_id) % 500}"
            
            txn = self.log_merchant_transaction(
                merchant_id=utility_merchant_id,
                amount=actual_utility,
                description="Household Utility Bills",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

        # Insurance premium with enhanced tracking
        insurance_day = random.randint(18, 22)
        if self.has_insurance_payments and date.day == insurance_day:
            # Add variation to insurance amount
            insurance_variation = random.uniform(0.95, 1.05)
            actual_insurance = self.insurance_premium * insurance_variation
            
            insurance_merchant_id = f"child_education_insurance_{hash(self.agent_id) % 300}"
            
            txn = self.log_merchant_transaction(
                merchant_id=insurance_merchant_id,
                amount=actual_insurance,
                description="Child Education Insurance Premium",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)

    def _handle_household_spending(self, date, events):
        """✅ UPDATED: Enhanced household spending with merchant tracking"""
        # Weekly groceries with enhanced variation
        if date.weekday() == self.weekly_grocery_day:
            grocery_amount = self.monthly_allowance * random.uniform(0.08, 0.16)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                grocery_amount *= random.uniform(1.2, 1.6)
            elif self.economic_class == 'Lower':
                grocery_amount *= random.uniform(0.7, 0.9)
            
            grocery_merchant_id = f"grocery_store_{hash(self.agent_id) % 200}"
            
            # ✅ NEW: Enhanced local merchant tracking
            self.add_frequent_merchant(grocery_merchant_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=grocery_merchant_id,
                amount=grocery_amount,
                description="Weekly Family Groceries",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

        # School fees with enhanced tracking
        school_fee_day = random.randint(3, 7)
        if date.month in self.school_fee_months and date.day == school_fee_day:
            fee_amount = self.monthly_allowance * random.uniform(0.4, 1.8)
            
            # Economic class adjustments for school fees
            if self.economic_class in ['High', 'Upper_Middle']:
                fee_amount *= random.uniform(1.5, 2.5)
            
            # ✅ NEW: Use children's school if available
            if self.children_schools:
                school_id = random.choice(self.children_schools)
                txn = self.log_merchant_transaction(
                    merchant_id=school_id,
                    amount=fee_amount,
                    description="Children School Fees",
                    date=date,
                    channel="Netbanking"
                )
            else:
                txn = self.log_transaction(
                    "DEBIT", "School Fees", fee_amount, date, channel="Netbanking"
                )
            
            if txn:
                events.append(txn)

        # Occasional spending with enhanced categories
        if random.random() < self.occasional_spend_chance:
            spending_categories = [
                ("Kids_Clothing", random.uniform(800, 3000)),
                ("Home_Goods", random.uniform(500, 2500)),
                ("Online_Pharmacy", random.uniform(300, 1500)),
                ("Family_Entertainment", random.uniform(600, 2000))
            ]
            
            category, amount = random.choice(spending_categories)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                amount *= random.uniform(1.3, 2.0)
            
            ecommerce_merchant_id = f"ecommerce_{category}_{hash(self.agent_id + str(date)) % 500}"
            
            txn = self.log_merchant_transaction(
                merchant_id=ecommerce_merchant_id,
                amount=amount,
                description=f"Family {category.replace('_', ' ')}",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def _handle_social_p2p_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced social circle transfers with seasonal patterns"""
        if (self.social_circle and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 400):
            
            recipient = random.choice(self.social_circle)
            
            # Homemakers typically send moderate amounts for social activities
            base_amount = random.uniform(200, 1500)
            
            # Economic class adjustments
            economic_multiplier = {
                'Lower': random.uniform(0.6, 0.9),
                'Lower_Middle': random.uniform(0.8, 1.2),
                'Middle': random.uniform(1.0, 1.4),
                'Upper_Middle': random.uniform(1.3, 2.0),
                'High': random.uniform(1.8, 2.8)
            }.get(self.economic_class, 1.0)
            
            final_amount = base_amount * economic_multiplier
            
            # Increase during festival months
            if date.month in self.festival_months:
                final_amount *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(final_amount, 2), 
                'desc': 'Social Community Transfer',
                'channel': channel,
                'transaction_category': 'social_transfer'
            })

    def _handle_family_support_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced family support with economic considerations"""
        if (self.extended_family and 
            random.random() < self.family_help_chance and
            self.balance > 800):
            
            recipient = random.choice(self.extended_family)
            
            # Family support amounts based on household allowance
            support_amount = self.monthly_allowance * random.uniform(0.08, 0.20)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                support_amount *= random.uniform(1.2, 1.8)
            elif self.economic_class == 'Lower':
                support_amount *= random.uniform(0.7, 1.0)
            
            # ✅ NEW: Select realistic channel based on amount
            if support_amount > 75000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(support_amount, 2), 
                'desc': 'Extended Family Support',
                'channel': channel,
                'transaction_category': 'family_support'
            })

    def _handle_children_related_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced children/education transfers with school context"""
        if (self.children_contacts and 
            random.random() < self.children_expense_chance and
            self.balance > 300):
            
            recipient = random.choice(self.children_contacts)
            
            # Children-related expenses are typically smaller but frequent
            base_amount = random.uniform(150, 1200)
            
            # Higher amounts during school activity months
            if date.month in self.school_activity_months:
                base_amount *= random.uniform(1.4, 2.2)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                base_amount *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(base_amount, 2), 
                'desc': 'Children Activity Expense',
                'channel': channel,
                'transaction_category': 'children_expense'
            })

    def _handle_community_support(self, date, events, context):
        """✅ UPDATED: Enhanced community support with group context"""
        if (self.social_circle and 
            random.random() < self.social_support_chance and
            self.balance > 600):
            
            recipient = random.choice(self.social_circle)
            
            # Community support amounts
            support_amount = random.uniform(300, 1200)
            
            # Adjust based on economic class and personality
            if self.economic_class in ['High', 'Upper_Middle']:
                support_amount *= random.uniform(1.3, 2.0)
            
            if self.financial_personality == 'Saver':
                support_amount *= random.uniform(0.8, 1.1)
            elif self.financial_personality == 'Over_Spender':
                support_amount *= random.uniform(1.1, 1.4)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(support_amount, 2), 
                'desc': 'Community Support Transfer',
                'channel': channel,
                'transaction_category': 'community_support'
            })

    def _handle_seasonal_community_activities(self, date, events, context):
        """✅ UPDATED: Handle seasonal community and religious activities"""
        if (date.month in self.festival_months and
            self.community_groups and
            random.random() < 0.15 and  # 15% chance during festival months
            self.balance > 800):
            
            # Festival contributions to community groups
            contribution_amount = self.monthly_allowance * random.uniform(0.03, 0.12)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                contribution_amount *= random.uniform(1.5, 2.5)
            
            # ✅ NEW: Enhanced community group contribution tracking
            group_id = random.choice(self.community_groups)
            
            txn = self.log_merchant_transaction(
                merchant_id=group_id,
                amount=contribution_amount,
                description="Community Festival Contribution",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def get_homemaker_specific_features(self):
        """✅ ENHANCED: Comprehensive homemaker-specific features"""
        return {
            'has_household_head': self.household_head_id is not None,
            'household_head_tenure': self.get_employment_tenure_months(),
            'allowance_consistency_score': self.allowance_consistency,
            'children_school_relationships': len(self.children_schools),
            'community_group_memberships': len(self.community_groups),
            'social_circle_size': len(self.social_circle),
            'family_support_obligations': len(self.extended_family),
            'children_related_contacts': len(self.children_contacts),
            'allowance_dependency_score': 1.0,  # Fully dependent on allowance
            'community_activity_level': len(self.community_groups) * 0.2,
            'last_allowance_recency': (datetime.now().date() - self.last_allowance_date).days if self.last_allowance_date else 999,
            'total_company_relationships': len(self.children_schools) + (1 if self.household_head_id else 0)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with household head allowance tracking"""
        events = []
        
        # Handle all income sources (including household head allowance tracking)
        self._handle_monthly_income_and_fixed_costs(date, events)
        
        # Handle household spending
        self._handle_household_spending(date, events)
        
        # Handle P2P transfers
        self._handle_social_p2p_transfers(date, events, context)
        self._handle_family_support_transfers(date, events, context)
        self._handle_children_related_transfers(date, events, context)
        self._handle_community_support(date, events, context)
        
        # Handle seasonal activities
        self._handle_seasonal_community_activities(date, events, context)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
