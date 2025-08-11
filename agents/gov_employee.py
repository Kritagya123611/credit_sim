import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class GovernmentEmployee(BaseAgent):
    """
    Enhanced Government Employee agent for Phase 2: Government department salary source tracking
    Includes government department companies as employers, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Middle', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Government Employee"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "35000-70000"
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
            "income_type": "Government_Salary",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "High",
            "has_investment_activity": True,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["LIC", "Government_Schemes"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.92, 0.99), 3),
            "ip_consistency_score": round(random.uniform(0.89, 0.98), 3),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Low",
            "ecommerce_avg_ticket_size": "Medium",
            
            # ✅ NEW: Heterogeneous graph connections specific to GovernmentEmployee
            "industry_sector": "Government_Public",
            "company_size": "Large_Government",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Government departments as employers (salary source tracking)
        self.government_departments = []  # Finance Ministry, Defense, Railways as company nodes
        self.primary_department_id = None  # Main employer department
        self.department_hierarchy = {}  # Track department hierarchy and roles

        # ✅ NEW: Employment relationship tracking
        self.government_service_tenure = random.randint(24, 360)  # 2-30 years
        self.salary_consistency = 1.0  # Government salaries are 100% consistent
        self.last_salary_date = None
        self.pension_eligibility_years = max(0, self.government_service_tenure - 240)  # 20+ years

        # Financial calculations with enhanced tracking
        self.salary_day = random.randint(28, 31)  # End of month government salary
        min_sal_mod, max_sal_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal_mod, max_sal_mod)
        
        # ✅ ENHANCED: More varied percentages based on personality and tenure
        self.emi_percentage = random.uniform(0.25, 0.35)
        self.investment_percentage = random.uniform(0.08, 0.12) * personality_config.get('invest_chance_mod', 1.0)
        self.insurance_percentage = random.uniform(0.06, 0.10)
        self.utility_bill_percentage = random.uniform(0.04, 0.06)
        self.remittance_percentage = random.uniform(0.15, 0.25) * (1.2 if financial_personality == 'Saver' else 1)

        # Spending patterns with more variation
        self.ecommerce_spend_chance = random.uniform(0.03, 0.07) * personality_config.get('spend_chance_mod', 1.0)
        
        # ✅ Enhanced P2P networks for Government Employees
        self.family_member_recipient = None  # Primary family remittance recipient
        self.professional_network = []  # Government colleagues
        self.extended_family = []  # Extended family for support
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.government_departments = []  # Department company nodes
        self.government_service_providers = []  # Pension, PF, medical services
        self.government_banks = []  # Preferred government banks (SBI, PNB, etc.)
        
        # P2P transfer probabilities with more variation
        self.family_remittance_chance = random.uniform(0.20, 0.30)  # Regular family support
        self.professional_transfer_chance = random.uniform(0.06, 0.10)  # Professional transfers
        self.extended_family_support_chance = random.uniform(0.10, 0.14)  # Extended family support
        self.government_service_payment_chance = random.uniform(0.03, 0.07)  # Service payments

        # Temporal tracking with enhanced features
        self.last_salary_date = None
        self.pension_contribution_tracking = []
        self.annual_increment_tracking = []  # Track yearly salary increments

        self.balance = random.uniform(self.salary_amount * 0.25, self.salary_amount * 0.9)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Government employees typically have 2-3 devices (phone, laptop, sometimes tablet)"""
        device_options = [2, 3, 4]
        weights = [0.4, 0.5, 0.1]  # Most have 2-3 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_government_departments(self, department_company_ids):
        """✅ NEW: Assign government department companies as employers for salary tracking"""
        self.government_departments = department_company_ids
        
        if department_company_ids:
            # Assign primary department as main employer
            self.primary_department_id = random.choice(department_company_ids)
            self.assign_employer(
                company_id=self.primary_department_id,
                employment_start_date=datetime.now().date() - timedelta(days=self.government_service_tenure * 30)
            )
            
            # Set up department hierarchy
            for dept_id in department_company_ids:
                self.department_hierarchy[dept_id] = {
                    'grade': random.choice(['Group_A', 'Group_B', 'Group_C', 'Group_D']),
                    'pay_scale': random.randint(6, 9),  # Government pay scales
                    'posting_type': random.choice(['Permanent', 'Deputation', 'Contractual'])
                }

    def assign_government_banks(self, bank_company_ids):
        """✅ NEW: Assign preferred government banks for salary accounts"""
        self.government_banks = bank_company_ids
        for bank_id in bank_company_ids:
            self.relationship_start_dates[f'bank_{bank_id}'] = datetime.now().date()

    def _handle_government_salary_payment(self, date, events):
        """✅ NEW: Handle monthly salary from government department"""
        if (date.day == self.salary_day and 
            random.random() < self.salary_consistency):  # 100% consistency for govt
            
            department_id = self.primary_department_id or (
                random.choice(self.government_departments) if self.government_departments else None
            )
            
            # Calculate salary with government increments
            base_salary = self.salary_amount
            
            # Annual increment simulation (typically 3% for government)
            years_served = self.government_service_tenure // 12
            increment_multiplier = (1.03 ** min(years_served, 10))  # Cap at 10 years
            
            # Add dearness allowance and other government benefits
            da_percentage = random.uniform(0.15, 0.25)  # 15-25% DA
            hra_percentage = random.uniform(0.08, 0.12)  # 8-12% HRA
            
            final_salary = base_salary * increment_multiplier * (1 + da_percentage + hra_percentage)
            
            # ✅ NEW: Log as salary transaction from government department
            if department_id:
                txn = self.log_salary_transaction(
                    amount=final_salary,
                    date=date,
                    company_id=department_id
                )
                if txn:
                    txn['transaction_category'] = 'government_salary'
                    txn['company_type'] = 'government_department'
                    txn['pay_grade'] = self.department_hierarchy.get(department_id, {}).get('grade', 'Unknown')
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", "Government Salary Deposit", final_salary, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)
            
            self.last_salary_date = date
            return final_salary
        
        return 0

    def _handle_recurring_events(self, date, events, context):
        """✅ UPDATED: Enhanced recurring events with company salary tracking"""
        # Government salary payment
        salary_amount = self._handle_government_salary_payment(date, events)
        
        if salary_amount > 0:
            # ✅ UPDATED: Monthly family remittance with realistic channel selection
            if (self.family_member_recipient and 
                random.random() < self.family_remittance_chance):
                
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
                    'desc': 'Government Employee Family Support',
                    'channel': channel,
                    'transaction_category': 'family_remittance'
                })
        
        # EMI payment with variation
        emi_day = random.randint(3, 7)
        if self.has_loan_emi and date.day == emi_day:
            emi_amount = self.salary_amount * self.emi_percentage
            # Add slight variation
            emi_amount *= random.uniform(0.98, 1.02)
            
            # ✅ NEW: Enhanced loan tracking
            loan_provider_id = f"govt_bank_loan_{hash(self.agent_id) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=loan_provider_id,
                amount=emi_amount,
                description="Government Employee Housing Loan EMI",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)

        # Insurance payment with enhanced tracking
        insurance_day = random.randint(8, 12)
        if self.has_insurance_payments and date.day == insurance_day:
            insurance_total = self.salary_amount * self.insurance_percentage
            
            # ✅ NEW: Enhanced insurance tracking
            insurance_provider_id = f"lic_govt_scheme_{hash(self.agent_id) % 500}"
            
            txn = self.log_merchant_transaction(
                merchant_id=insurance_provider_id,
                amount=insurance_total,
                description="LIC/Government Insurance Premium",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)

        # Investment with enhanced tracking
        investment_day = random.randint(13, 17)
        if self.has_investment_activity and date.day == investment_day:
            invest_amt = self.salary_amount * self.investment_percentage
            
            # ✅ NEW: Enhanced investment tracking
            investment_provider_id = f"govt_investment_{hash(self.agent_id) % 200}"
            
            txn = self.log_merchant_transaction(
                merchant_id=investment_provider_id,
                amount=invest_amt,
                description="PPF/NSC/Government Securities",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)
            
        # Utility bills with enhanced tracking
        utility_day = random.randint(18, 22)
        if date.day == utility_day:
            bill_amount = self.salary_amount * self.utility_bill_percentage
            
            # ✅ NEW: Enhanced utility tracking
            utility_provider_id = f"govt_utility_{hash(self.agent_id) % 300}"
            
            txn = self.log_merchant_transaction(
                merchant_id=utility_provider_id,
                amount=bill_amount,
                description="Government Quarters Utility Bill",
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
            self.balance > 8000):
            
            colleague = random.choice(self.professional_network)
            
            # Professional transfers (office collections, shared expenses, etc.)
            transfer_amount = random.uniform(500, 4000)
            
            # Adjust based on economic class and seniority
            if self.economic_class in ['Upper_Middle', 'High']:
                transfer_amount *= random.uniform(1.2, 2.0)
            
            # Senior officers tend to contribute more
            if self.government_service_tenure > 180:  # 15+ years
                transfer_amount *= random.uniform(1.1, 1.5)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': colleague, 
                'amount': round(transfer_amount, 2), 
                'desc': 'Government Office Collection',
                'channel': channel,
                'transaction_category': 'professional_transfer'
            })

    def _handle_extended_family_support(self, date, events, context):
        """✅ UPDATED: Enhanced extended family support"""
        if (self.extended_family and 
            random.random() < self.extended_family_support_chance and
            self.balance > 12000):
            
            family_member = random.choice(self.extended_family)
            
            # Extended family support amounts
            support_amount = self.salary_amount * random.uniform(0.04, 0.12)
            
            # Adjust based on financial personality and tenure
            if self.financial_personality == 'Saver':
                support_amount *= random.uniform(1.1, 1.4)  # Savers are more family-supportive
            
            # Senior government employees support more
            if self.government_service_tenure > 120:  # 10+ years
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

    def _handle_government_service_payments(self, date, events, context):
        """✅ UPDATED: Payments to government service providers"""
        if (random.random() < self.government_service_payment_chance and
            self.balance > 3000):
            
            # Government service payments (pension contributions, PF, medical, etc.)
            service_amount = self.salary_amount * random.uniform(0.015, 0.04)
            
            # ✅ NEW: Enhanced government service provider tracking
            service_id = f"govt_service_{hash(self.agent_id + str(date)) % 500}"
            self.add_government_service_provider(service_id, date)
            
            service_types = [
                "Employee Provident Fund Contribution",
                "Government Medical Service",
                "Pension Fund Contribution",
                "GPF Advance Payment"
            ]
            
            txn = self.log_merchant_transaction(
                merchant_id=service_id,
                amount=service_amount,
                description=random.choice(service_types),
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

    def _handle_daily_spending(self, date, events):
        """✅ UPDATED: Enhanced e-commerce spending with merchant tracking"""
        if random.random() < self.ecommerce_spend_chance:
            ecommerce_categories = [
                ("Books_Education", random.uniform(300, 1500)),
                ("Household_Essentials", random.uniform(800, 3000)),
                ("Government_Store", random.uniform(500, 2000)),
                ("Medical_Pharmacy", random.uniform(200, 1200))
            ]
            
            category, ecommerce_amt = random.choice(ecommerce_categories)
            
            # Economic class adjustments
            if self.economic_class in ['Upper_Middle', 'High']:
                ecommerce_amt *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Enhanced e-commerce merchant tracking
            ecommerce_merchant_id = f"ecommerce_{category}_{hash(self.agent_id + str(date)) % 500}"
            
            txn = self.log_merchant_transaction(
                merchant_id=ecommerce_merchant_id,
                amount=ecommerce_amt,
                description=f"E-commerce ({category.replace('_', ' ')})",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def get_government_employee_features(self):
        """✅ ENHANCED: Comprehensive government employee features"""
        return {
            'department_employer_count': len(self.government_departments),
            'primary_department_tenure': self.get_employment_tenure_months(),
            'government_service_years': self.government_service_tenure // 12,
            'service_provider_relationships': len(self.government_service_providers),
            'government_bank_relationships': len(self.government_banks),
            'professional_network_size': len(self.professional_network),
            'family_support_obligations': len(self.extended_family) + (1 if self.family_member_recipient else 0),
            'salary_consistency_score': self.salary_consistency,  # Always 1.0 for government
            'payment_reliability_score': 1.0,  # Government employees are very reliable
            'pension_eligibility_years': self.pension_eligibility_years,
            'last_salary_recency': (datetime.now().date() - self.last_salary_date).days if self.last_salary_date else 999,
            'total_company_relationships': len(self.government_departments) + len(self.government_banks)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with government department salary tracking"""
        events = []
        
        # Handle all income sources (including government department salary tracking)
        self._handle_recurring_events(date, events, context)
        
        # Handle P2P transfers
        self._handle_professional_network_transfers(date, events, context)
        self._handle_extended_family_support(date, events, context)
        
        # Handle government service payments
        self._handle_government_service_payments(date, events, context)
        
        # Handle daily spending
        self._handle_daily_spending(date, events)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
