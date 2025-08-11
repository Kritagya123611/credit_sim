import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class PoliceOfficer(BaseAgent):
    """
    Enhanced Police Officer agent for Phase 2: Government department salary source tracking
    Includes police department companies as employers, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Police"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "30000-50000"
        min_sal, max_sal = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_sal * income_multiplier)}-{int(max_sal * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Salaried",
            "employment_verification": "GOVT_Verified",
            "income_type": "Uniformed_Services_Salary",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "Medium",
            "has_investment_activity": True,
            "investment_types": ["LIC", "FD"],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["LIC", "Government_Schemes"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.89, 0.98), 3),
            "ip_consistency_score": round(random.uniform(0.72, 0.88), 3),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Mobile_Banking", "UPI"],
            "login_pattern": "Shift_Work_Night_Activity",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Medium",
            
            # ✅ NEW: Heterogeneous graph connections specific to PoliceOfficer
            "industry_sector": "Government_Security",
            "company_size": "Large_Government",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Police departments as employers (salary source tracking)
        self.police_departments = []  # State Police, Central Forces as company nodes
        self.primary_department_id = None  # Main employer department
        self.department_hierarchy = {}  # Track department ranks and roles

        # ✅ NEW: Employment relationship tracking
        self.police_service_tenure = random.randint(24, 360)  # 2-30 years in service
        self.salary_consistency = random.uniform(0.98, 1.0)  # Very high consistency
        self.last_salary_date = None
        self.last_allowance_date = None

        # Financial calculations with more variation
        self.salary_day = random.randint(28, 31)  # End of month government salary
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_mod, max_mod)

        # ✅ ENHANCED: More varied percentages based on personality and rank
        self.remittance_percentage = random.uniform(0.25, 0.35) * (1.2 if financial_personality == 'Saver' else 1)
        self.emi_percentage = random.uniform(0.18, 0.22)
        self.investment_percentage = random.uniform(0.08, 0.12) * personality_config.get('invest_chance_mod', 1.0)
        self.insurance_percentage = random.uniform(0.07, 0.09)
        
        # Spending patterns with more variation
        self.daily_spend_chance = random.uniform(0.55, 0.65) * personality_config.get('spend_chance_mod', 1.0)
        
        # ✅ Enhanced P2P networks for Police Officers
        self.family_member_recipient = None  # Primary family recipient
        self.professional_network = []  # Fellow officers and security personnel
        self.extended_family = []  # Extended family for support
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.police_departments = []  # Police department as company nodes
        self.government_service_providers = []  # Pension, medical services
        self.security_equipment_vendors = []  # Uniform, equipment suppliers
        
        # P2P transfer probabilities with more variation
        self.professional_transfer_chance = random.uniform(0.06, 0.10)  # Professional transfers
        self.extended_family_support_chance = random.uniform(0.08, 0.12)  # Extended family support
        self.emergency_help_chance = random.uniform(0.04, 0.06)  # Emergency help to colleagues

        # Temporal tracking with enhanced features
        self.last_duty_allowance_date = None
        self.shift_patterns = []  # Track duty shift patterns
        self.duty_allowance_cycles = []  # Track duty allowance patterns

        self.balance = random.uniform(self.salary_amount * 0.12, self.salary_amount * 0.45)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Police officers typically have 2-3 devices (personal phone, service phone, sometimes tablet)"""
        device_options = [2, 3, 4]
        weights = [0.4, 0.5, 0.1]  # Most have 2-3 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_police_departments(self, department_company_ids):
        """✅ NEW: Assign police departments as employers for salary tracking"""
        self.police_departments = department_company_ids
        
        if department_company_ids:
            # Assign primary department as main employer
            self.primary_department_id = random.choice(department_company_ids)
            self.assign_employer(
                company_id=self.primary_department_id,
                employment_start_date=datetime.now().date() - timedelta(days=self.police_service_tenure * 30)
            )
            
            # Set up department hierarchy
            for dept_id in department_company_ids:
                self.department_hierarchy[dept_id] = {
                    'rank': random.choice(['Constable', 'Head_Constable', 'ASI', 'SI', 'Inspector', 'ACP']),
                    'unit': random.choice(['Traffic', 'Crime_Branch', 'Law_Order', 'Special_Branch']),
                    'posting_type': random.choice(['District', 'Headquarters', 'Special_Unit'])
                }

    def _handle_police_department_salary_payment(self, date, events):
        """✅ NEW: Handle monthly salary from police department"""
        if (date.day == self.salary_day and 
            random.random() < self.salary_consistency):
            
            department_id = self.primary_department_id or (
                random.choice(self.police_departments) if self.police_departments else None
            )
            
            # Calculate salary with police service patterns
            base_salary = self.salary_amount
            
            # Add service-based increments
            years_service = self.police_service_tenure // 12
            increment_multiplier = (1.08 ** min(years_service, 15))  # 8% yearly increment cap at 15 years
            
            # Add police-specific allowances
            da_percentage = random.uniform(0.12, 0.20)  # 12-20% DA
            hra_percentage = random.uniform(0.08, 0.15)  # 8-15% HRA
            duty_allowance_percentage = random.uniform(0.05, 0.10)  # 5-10% duty allowance
            
            final_salary = base_salary * increment_multiplier * (1 + da_percentage + hra_percentage + duty_allowance_percentage)
            
            # ✅ NEW: Log as salary transaction from police department
            if department_id:
                txn = self.log_salary_transaction(
                    amount=final_salary,
                    date=date,
                    company_id=department_id
                )
                if txn:
                    txn['transaction_category'] = 'police_department_salary'
                    txn['company_type'] = 'police_department'
                    txn['officer_rank'] = self.department_hierarchy.get(department_id, {}).get('rank', 'Unknown')
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", "Police Salary Deposit", final_salary, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)
            
            self.last_salary_date = date
            return final_salary
        
        return 0

    def _handle_fixed_monthly_events(self, date, events, context):
        """✅ UPDATED: Enhanced salary and fixed payments with department tracking"""
        # Police department salary payment
        salary_amount = self._handle_police_department_salary_payment(date, events)
        
        if salary_amount > 0:
            # ✅ UPDATED: Family remittance with realistic channel selection
            if self.family_member_recipient:
                remittance_amount = salary_amount * self.remittance_percentage
                
                # ✅ NEW: Select realistic channel based on amount
                if remittance_amount > 100000:
                    channel = random.choice(['NEFT', 'RTGS'])
                elif remittance_amount > 50000:
                    channel = random.choice(['IMPS', 'NEFT'])
                else:
                    channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': self.family_member_recipient, 
                    'amount': round(remittance_amount, 2), 
                    'desc': 'Police Officer Family Support',
                    'channel': channel,
                    'transaction_category': 'family_remittance'
                })

        # Loan EMI with variation
        emi_day = random.randint(8, 12)
        if self.has_loan_emi and date.day == emi_day:
            # Add slight variation to EMI amount
            emi_variation = random.uniform(0.98, 1.02)
            emi_amount = (self.salary_amount * self.emi_percentage) * emi_variation
            
            loan_provider_id = f"police_officer_loan_{hash(self.agent_id) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=loan_provider_id,
                amount=emi_amount,
                description="Police Officer Housing Loan EMI",
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
            
            insurance_provider_id = f"police_insurance_{hash(self.agent_id) % 300}"
            
            txn = self.log_merchant_transaction(
                merchant_id=insurance_provider_id,
                amount=insurance_total,
                description="Police LIC/Government Insurance",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)

        # Investment with enhanced tracking
        investment_day = random.randint(18, 22)
        if self.has_investment_activity and date.day == investment_day:
            # Add variation to investment amount
            investment_variation = random.uniform(0.9, 1.1)
            invest_amt = (self.salary_amount * self.investment_percentage) * investment_variation
            
            investment_provider_id = f"police_investment_{hash(self.agent_id) % 200}"
            
            txn = self.log_merchant_transaction(
                merchant_id=investment_provider_id,
                amount=invest_amt,
                description="Police FD/PPF Investment",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

    def add_government_service_provider(self, service_id, first_service_date=None):
        """✅ NEW: Track government service provider relationships"""
        if service_id not in self.government_service_providers:
            self.government_service_providers.append(service_id)
            self.add_frequent_merchant(service_id, first_service_date)

    def _handle_professional_network_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced professional network transfers"""
        if (self.professional_network and 
            random.random() < self.professional_transfer_chance and
            self.balance > 5000):
            
            colleague = random.choice(self.professional_network)
            
            # Professional transfers (shared expenses, mutual aid, etc.)
            transfer_amount = random.uniform(1500, 5000)
            
            # Adjust based on economic class and seniority
            if self.economic_class in ['Upper_Middle', 'High']:
                transfer_amount *= random.uniform(1.2, 1.8)
            
            # Senior officers tend to contribute more
            if self.police_service_tenure > 120:  # 10+ years
                transfer_amount *= random.uniform(1.1, 1.4)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': colleague, 
                'amount': round(transfer_amount, 2), 
                'desc': 'Police Officer Professional Transfer',
                'channel': channel,
                'transaction_category': 'professional_transfer'
            })

    def _handle_extended_family_support(self, date, events, context):
        """✅ UPDATED: Enhanced extended family support"""
        if (self.extended_family and 
            random.random() < self.extended_family_support_chance and
            self.balance > 8000):
            
            family_member = random.choice(self.extended_family)
            
            # Extended family support amounts
            support_amount = self.salary_amount * random.uniform(0.06, 0.15)
            
            # Adjust based on financial personality and seniority
            if self.financial_personality == 'Saver':
                support_amount *= random.uniform(1.1, 1.4)  # Savers are more supportive
            
            # Senior officers support more
            if self.police_service_tenure > 180:  # 15+ years
                support_amount *= random.uniform(1.2, 1.6)
            
            # ✅ NEW: Select realistic channel
            if support_amount > 75000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': family_member, 
                'amount': round(support_amount, 2), 
                'desc': 'Extended Family Support',
                'channel': channel,
                'transaction_category': 'family_support'
            })

    def _handle_emergency_colleague_help(self, date, events, context):
        """✅ UPDATED: Enhanced emergency help to fellow officers"""
        if (self.professional_network and 
            random.random() < self.emergency_help_chance and
            self.balance > 10000):  # Need good balance for emergency help
            
            colleague = random.choice(self.professional_network)
            
            # Emergency help amounts
            emergency_amount = random.uniform(3000, 8000)
            
            # Adjust based on economic class and service tenure
            if self.economic_class in ['Upper_Middle', 'High']:
                emergency_amount *= random.uniform(1.2, 1.8)
                
            if self.police_service_tenure > 120:  # Senior officers help more
                emergency_amount *= random.uniform(1.1, 1.5)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': colleague, 
                'amount': round(emergency_amount, 2), 
                'desc': 'Police Emergency Support',
                'channel': channel,
                'transaction_category': 'emergency_help'
            })

    def _handle_duty_related_expenses(self, date, events, context):
        """✅ UPDATED: Enhanced duty-related expenses and equipment purchases"""
        if (random.random() < 0.12 and  # 12% chance of duty-related expense
            self.balance > 3000):
            
            # Duty-related expenses (uniform, equipment maintenance)
            expense_amount = random.uniform(600, 2500)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                expense_amount *= random.uniform(1.2, 1.6)
            
            # ✅ NEW: Enhanced security equipment vendor tracking
            vendor_id = f"police_equipment_{hash(self.agent_id + str(date)) % 500}"
            self.add_frequent_merchant(vendor_id, date)
            
            expense_types = [
                "Police Uniform/Equipment",
                "Duty Gear Maintenance",
                "Training Materials",
                "Service Equipment"
            ]
            
            txn = self.log_merchant_transaction(
                merchant_id=vendor_id,
                amount=expense_amount,
                description=random.choice(expense_types),
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def _handle_daily_spending(self, date, events):
        """✅ UPDATED: Enhanced daily spending with shift-work patterns"""
        if random.random() < self.daily_spend_chance:
            is_night_shift = random.random() < 0.35  # 35% chance of night shift
            
            if is_night_shift:
                description = "Night Duty - Food/Tea"
                amount = random.uniform(120, 400)
                merchant_id = f"night_duty_food_{hash(self.agent_id) % 200}"
            else:
                description = "Daily Expense - Groceries/Misc"
                amount = random.uniform(250, 1000)
                merchant_id = f"daily_grocery_{hash(self.agent_id) % 300}"
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                amount *= random.uniform(1.2, 1.5)
            
            # ✅ NEW: Enhanced merchant tracking
            txn = self.log_merchant_transaction(
                merchant_id=merchant_id,
                amount=amount,
                description=description,
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def get_police_officer_features(self):
        """✅ ENHANCED: Comprehensive police officer features"""
        return {
            'police_department_employer_count': len(self.police_departments),
            'primary_department_tenure': self.get_employment_tenure_months(),
            'police_service_years': self.police_service_tenure // 12,
            'service_provider_relationships': len(self.government_service_providers),
            'equipment_vendor_relationships': len(self.security_equipment_vendors),
            'professional_network_size': len(self.professional_network),
            'family_support_obligations': len(self.extended_family) + (1 if self.family_member_recipient else 0),
            'salary_consistency_score': self.salary_consistency,  # Very high for police
            'duty_reliability_score': 1.0,  # Police officers are very reliable
            'shift_work_pattern_score': 1.0,  # Indicates irregular hours
            'last_salary_recency': (datetime.now().date() - self.last_salary_date).days if self.last_salary_date else 999,
            'service_rank_seniority': self.police_service_tenure / 360,  # Normalized seniority
            'total_company_relationships': len(self.police_departments)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with police department salary tracking"""
        events = []
        
        # Handle all income sources (including police department salary tracking)
        self._handle_fixed_monthly_events(date, events, context)
        
        # Handle P2P transfers
        self._handle_professional_network_transfers(date, events, context)
        self._handle_extended_family_support(date, events, context)
        self._handle_emergency_colleague_help(date, events, context)
        
        # Handle duty-related expenses and daily spending
        self._handle_duty_related_expenses(date, events, context)
        self._handle_daily_spending(date, events)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
