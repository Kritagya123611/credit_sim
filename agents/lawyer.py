import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class Lawyer(BaseAgent):
    """
    Enhanced Lawyer agent for Phase 2: Law firm salary source tracking
    Includes law firm companies as employers, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Middle', financial_personality='Rational_Investor'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Lawyer"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "50000-200000"
        min_monthly, max_monthly = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_monthly * income_multiplier)}-{int(max_monthly * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-Employed_Professional",
            "employment_verification": "Professional_License_Verified",
            "income_type": "Professional_Fees",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Lumpy",
            "savings_retention_rate": "Medium",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life", "Prof_Indemnity"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "High-Value_Postpaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.85, 0.95), 3),
            "ip_consistency_score": round(random.uniform(0.78, 0.92), 3),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Cards"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "High",
            
            # ✅ NEW: Heterogeneous graph connections specific to Lawyer
            "industry_sector": "Legal_Professional",
            "company_size": "Small_Practice",
        }
        
        super().__init__(**profile_attributes)

        # ✅ NEW: Law firms as employers (salary source tracking)
        self.law_firms = []  # Partner law firms as company nodes
        self.primary_law_firm_id = None  # Main income source law firm
        self.law_firm_hierarchy = {}  # Track law firm roles and partnerships

        # ✅ NEW: Employment relationship tracking
        self.legal_practice_tenure = random.randint(24, 480)  # 2-40 years in practice
        self.income_consistency = random.uniform(0.8, 0.95)  # Payment reliability
        self.last_fee_date = None
        self.last_bonus_date = None

        # Financial calculations with more variation
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_income = random.uniform(min_mod, max_mod)

        # Income patterns - lumpy payments with enhanced tracking
        self.payout_months = sorted(random.sample(range(1, 13), k=random.randint(2, 5)))
        self.lump_sum_payment = self.avg_monthly_income * (12 / len(self.payout_months))
        self.has_large_cash_reserve = False

        # Fixed expenses with variation
        self.junior_retainer_fee = self.avg_monthly_income * random.uniform(0.35, 0.65)
        self.loan_emi_amount = self.avg_monthly_income * random.uniform(0.25, 0.35)
        self.prof_indemnity_premium = self.avg_monthly_income * random.uniform(0.04, 0.06)

        # Behavior modifiers with enhanced tracking
        self.spend_chance_mod = personality_config.get('spend_chance_mod', 1.0)
        self.invest_chance_mod = personality_config.get('invest_chance_mod', 1.0)
        
        # ✅ Enhanced P2P networks for Lawyers
        self.professional_network = []  # Fellow lawyers, consultants
        self.junior_associate = None    # Single junior/associate for retainer payments
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.law_firms = []  # Associated law firms as company nodes
        self.court_systems = []  # Court systems as institutional nodes
        self.legal_service_providers = []  # Legal research, software as merchant nodes
        self.client_companies = []  # Corporate clients as company nodes
        
        # P2P transfer probabilities with more variation
        self.p2p_transfer_chance = random.uniform(0.10, 0.14) * personality_config.get('spend_chance_mod', 1.0)
        self.professional_transfer_chance = random.uniform(0.06, 0.10)
        self.client_refund_chance = random.uniform(0.03, 0.07)
        
        # Professional payment cycles with variation
        self.retainer_payment_day = random.randint(3, 7)

        # Temporal tracking with enhanced features
        self.last_large_payout_date = None
        self.case_cycles = []  # Track case settlement patterns
        self.client_payment_patterns = []

        self.balance = random.uniform(self.avg_monthly_income * 1.2, self.avg_monthly_income * 3.5)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Lawyers typically have 2-4 devices (phone, laptop, tablet, work phone)"""
        device_options = [2, 3, 4]
        weights = [0.3, 0.5, 0.2]  # Most have 3 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_law_firms(self, law_firm_company_ids):
        """✅ NEW: Assign law firms as employers for salary tracking"""
        self.law_firms = law_firm_company_ids
        
        if law_firm_company_ids:
            # Assign primary law firm as main employer
            self.primary_law_firm_id = random.choice(law_firm_company_ids)
            self.assign_employer(
                company_id=self.primary_law_firm_id,
                employment_start_date=datetime.now().date() - timedelta(days=self.legal_practice_tenure * 30)
            )
            
            # Set up law firm hierarchy and roles
            for firm_id in law_firm_company_ids:
                self.law_firm_hierarchy[firm_id] = {
                    'role': random.choice(['Partner', 'Senior_Associate', 'Of_Counsel', 'Consultant']),
                    'practice_area': random.choice(['Corporate', 'Litigation', 'Family', 'Criminal', 'Tax']),
                    'profit_sharing': random.choice([True, False])
                }

    def assign_client_companies(self, client_company_ids):
        """✅ NEW: Assign client companies for fee tracking"""
        self.client_companies = client_company_ids
        for client_id in client_company_ids:
            self.relationship_start_dates[f'client_{client_id}'] = datetime.now().date()

    def _handle_law_firm_fee_payment(self, date, events):
        """✅ NEW: Handle fees/salary from law firms"""
        if (date.month in self.payout_months and 
            date.day == 15 and 
            random.random() < self.income_consistency):
            
            firm_id = self.primary_law_firm_id or (
                random.choice(self.law_firms) if self.law_firms else None
            )
            
            # Calculate fee with law practice patterns
            base_fee = self.lump_sum_payment
            
            # Add partnership profit sharing
            if (firm_id and 
                self.law_firm_hierarchy.get(firm_id, {}).get('profit_sharing', False)):
                base_fee *= random.uniform(1.2, 1.8)
            
            # Add case success bonuses
            if random.random() < 0.3:  # 30% chance of case bonus
                base_fee *= random.uniform(1.1, 1.5)
            
            # Seasonal variations (legal calendar patterns)
            month_multiplier = {
                12: 0.8,  # December - holidays
                1: 0.9,   # January - slow start
                6: 1.2,   # June - mid-year settlements
                9: 1.1,   # September - back from summer
            }.get(date.month, 1.0)
            
            final_fee = base_fee * month_multiplier
            
            # ✅ NEW: Log as salary transaction from law firm
            if firm_id:
                txn = self.log_salary_transaction(
                    amount=final_fee,
                    date=date,
                    company_id=firm_id
                )
                if txn:
                    txn['transaction_category'] = 'law_firm_fee'
                    txn['company_type'] = 'law_firm'
                    txn['lawyer_role'] = self.law_firm_hierarchy.get(firm_id, {}).get('role', 'Unknown')
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", "Legal Fee Payment", final_fee, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)
            
            self.has_large_cash_reserve = True
            self.last_large_payout_date = date
            self.last_fee_date = date
            return final_fee
        
        return 0

    def _handle_lumpy_income(self, date, events):
        """✅ UPDATED: Enhanced lumpy income with company salary tracking"""
        # Law firm fee payments
        fee_amount = self._handle_law_firm_fee_payment(date, events)

        # Direct client payments (in addition to law firm fees)
        if (random.random() < 0.3 and  # 30% chance of direct client payment
            date.day in [5, 20] and
            self.client_companies):
            
            client_id = random.choice(self.client_companies)
            client_payment = self.avg_monthly_income * random.uniform(0.2, 0.8)
            
            # ✅ NEW: Log as salary transaction from client company
            txn = self.log_salary_transaction(
                amount=client_payment,
                date=date,
                company_id=client_id
            )
            
            if txn:
                txn['transaction_category'] = 'direct_client_fee'
                txn['company_type'] = 'client_company'
                events.append(txn)

    def _handle_recurring_debits(self, date, events):
        """✅ UPDATED: Enhanced recurring payments with merchant tracking"""
        # Loan EMI with variation
        emi_day = random.randint(8, 12)
        if self.has_loan_emi and date.day == emi_day:
            # Add slight variation to EMI amount
            emi_variation = random.uniform(0.98, 1.02)
            actual_emi = self.loan_emi_amount * emi_variation
            
            loan_provider_id = f"legal_practice_loan_{hash(self.agent_id) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=loan_provider_id,
                amount=actual_emi,
                description="Legal Practice Office Loan EMI",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)

        # Professional indemnity insurance (annual)
        insurance_day = random.randint(18, 22)
        if (self.has_insurance_payments and 
            date.month == 7 and 
            date.day == insurance_day):
            
            # Add variation to insurance amount
            insurance_variation = random.uniform(0.95, 1.05)
            actual_insurance = self.prof_indemnity_premium * insurance_variation
            
            insurance_provider_id = f"legal_indemnity_insurance_{hash(self.agent_id) % 300}"
            
            txn = self.log_merchant_transaction(
                merchant_id=insurance_provider_id,
                amount=actual_insurance,
                description="Professional Indemnity Insurance",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

    def add_legal_service_provider(self, provider_id, first_service_date=None):
        """✅ NEW: Track legal service provider relationships"""
        if provider_id not in self.legal_service_providers:
            self.legal_service_providers.append(provider_id)
            self.add_frequent_merchant(provider_id, first_service_date)

    def add_client_company(self, company_id, first_case_date=None):
        """✅ NEW: Track corporate client relationships"""
        if company_id not in self.client_companies:
            self.client_companies.append(company_id)
            if first_case_date:
                self.relationship_start_dates[f'client_{company_id}'] = first_case_date

    def _handle_professional_retainer_payments(self, date, events, context):
        """✅ UPDATED: Enhanced retainer payments with realistic channels"""
        # Monthly retainer payment to junior associate
        if (date.day == self.retainer_payment_day and 
            self.junior_associate and 
            self.balance > self.junior_retainer_fee):
            
            # Add variation to retainer amount
            retainer_variation = random.uniform(0.95, 1.05)
            actual_retainer = self.junior_retainer_fee * retainer_variation
            
            # ✅ NEW: Select realistic channel based on amount
            if actual_retainer > 100000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif actual_retainer > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self,
                'recipient': self.junior_associate,
                'amount': round(actual_retainer, 2),
                'desc': 'Junior Associate Retainer',
                'channel': channel,
                'transaction_category': 'professional_retainer'
            })

    def _handle_professional_network_transfers(self, date, events, context):
        """✅ UPDATED: Enhanced professional network transfers"""
        if (self.professional_network and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 8000):
            
            recipient = random.choice(self.professional_network)
            
            # Lawyers typically send higher amounts in professional context
            base_amount = random.uniform(3000, 12000)
            
            # Economic class adjustments
            economic_multiplier = {
                'Lower_Middle': random.uniform(0.7, 1.0),
                'Middle': random.uniform(0.9, 1.3),
                'Upper_Middle': random.uniform(1.2, 1.8),
                'High': random.uniform(1.6, 2.5)
            }.get(self.economic_class, 1.0)
            
            # Increase amounts if they have large cash reserves
            if self.has_large_cash_reserve:
                base_amount *= random.uniform(1.3, 2.2)
            
            # Experience-based adjustments
            if self.legal_practice_tenure > 120:  # 10+ years experience
                base_amount *= random.uniform(1.1, 1.5)
            
            final_amount = base_amount * economic_multiplier
            
            # ✅ NEW: Select realistic channel based on amount
            if final_amount > 200000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif final_amount > 100000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(final_amount, 2), 
                'desc': 'Legal Professional Transfer',
                'channel': channel,
                'transaction_category': 'professional_transfer'
            })

    def _handle_professional_networking_transfers(self, date, events, context):
        """✅ UPDATED: Additional professional transfers (referral fees, shared costs)"""
        if (self.professional_network and 
            random.random() < self.professional_transfer_chance and 
            self.balance > 15000):
            
            recipient = random.choice(self.professional_network)
            
            # Professional networking amounts are typically higher for lawyers
            amount = random.uniform(2000, 8000)
            
            # Higher amounts during large cash reserve periods
            if self.has_large_cash_reserve:
                amount *= random.uniform(1.4, 2.8)
            
            # Adjust based on economic class and experience
            if self.economic_class in ['High', 'Upper_Middle']:
                amount *= random.uniform(1.3, 2.0)
                
            if self.legal_practice_tenure > 180:  # 15+ years
                amount *= random.uniform(1.2, 1.6)
            
            # ✅ NEW: Select realistic channel based on amount
            if amount > 150000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif amount > 75000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(amount, 2), 
                'desc': 'Legal Referral/Shared Costs',
                'channel': channel,
                'transaction_category': 'professional_networking'
            })

    def _handle_client_refunds(self, date, events, context):
        """✅ UPDATED: Enhanced client refunds and professional service payments"""
        if (self.professional_network and 
            random.random() < self.client_refund_chance and
            self.has_large_cash_reserve and
            self.balance > 20000):
            
            recipient = random.choice(self.professional_network)
            
            # Client refunds are typically larger amounts for lawyers
            refund_amount = random.uniform(8000, 35000)
            
            # Adjust based on economic class and practice success
            if self.economic_class in ['High', 'Upper_Middle']:
                refund_amount *= random.uniform(1.4, 2.8)
            
            # Senior lawyers handle larger refunds
            if self.legal_practice_tenure > 120:
                refund_amount *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Select appropriate channel for larger refunds
            if refund_amount > 200000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif refund_amount > 100000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(refund_amount, 2), 
                'desc': 'Client Refund/Settlement',
                'channel': channel,
                'transaction_category': 'client_refund'
            })

    def _handle_legal_service_payments(self, date, events, context):
        """✅ UPDATED: Payments to legal service providers"""
        if (random.random() < 0.18 and  # 18% chance
            self.balance > 8000):
            
            # Legal service payments (research, software, court fees)
            service_amount = self.avg_monthly_income * random.uniform(0.02, 0.08)
            
            # Adjust based on practice area and economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                service_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Enhanced legal service provider tracking
            service_id = f"legal_service_{hash(self.agent_id + str(date)) % 500}"
            self.add_legal_service_provider(service_id, date)
            
            service_types = [
                "Legal Research Database",
                "Court Filing Fees",
                "Legal Software Subscription",
                "Professional Development"
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

    def _handle_spending_and_investment(self, date, events):
        """✅ UPDATED: Enhanced spending and investments after large payouts"""
        # Large investments after receiving payments
        if (self.has_large_cash_reserve and 
            self.has_investment_activity and 
            random.random() < (0.6 * self.invest_chance_mod)):
            
            investment_amount = self.balance * random.uniform(0.25, 0.65)
            investment_type = random.choice(self.investment_types)
            
            # ✅ NEW: Enhanced investment tracking
            investment_provider_id = f"legal_investment_{hash(self.agent_id) % 300}"
            
            txn = self.log_merchant_transaction(
                merchant_id=investment_provider_id,
                amount=investment_amount,
                description=f"Legal Practice Investment - {investment_type}",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)
                self.has_large_cash_reserve = False

        # Professional and lifestyle spending
        if random.random() < (0.45 * self.spend_chance_mod):
            spending_categories = [
                ("Fine_Dining", random.uniform(2000, 8000)),
                ("Travel_Booking", random.uniform(10000, 40000)),
                ("Professional_Books", random.uniform(1000, 5000)),
                ("Legal_Software", random.uniform(3000, 15000)),
                ("Court_Fees", random.uniform(500, 3000))
            ]
            
            category, amount = random.choice(spending_categories)
            
            # Economic class adjustments
            if self.economic_class in ['High', 'Upper_Middle']:
                amount *= random.uniform(1.4, 2.2)
            
            # ✅ NEW: Enhanced merchant tracking
            spend_merchant_id = f"professional_{category}_{hash(self.agent_id + str(date)) % 500}"
            
            txn = self.log_merchant_transaction(
                merchant_id=spend_merchant_id,
                amount=amount,
                description=f"Professional {category.replace('_', ' ')}",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def get_lawyer_specific_features(self):
        """✅ ENHANCED: Comprehensive lawyer-specific features"""
        return {
            'law_firm_employer_count': len(self.law_firms),
            'primary_law_firm_tenure': self.get_employment_tenure_months(),
            'legal_practice_experience_years': self.legal_practice_tenure // 12,
            'client_company_relationships': len(self.client_companies),
            'legal_service_provider_relationships': len(self.legal_service_providers),
            'professional_network_size': len(self.professional_network),
            'has_junior_associate': self.junior_associate is not None,
            'income_lumpiness_score': 12 / len(self.payout_months),  # Higher = more lumpy
            'cash_reserve_cycles': len(self.case_cycles),
            'professional_payment_reliability': 1.0,  # Lawyers are typically reliable
            'last_fee_recency': (datetime.now().date() - self.last_fee_date).days if self.last_fee_date else 999,
            'partnership_profit_sharing': any(self.law_firm_hierarchy.get(f, {}).get('profit_sharing', False) for f in self.law_firms),
            'total_company_relationships': len(self.law_firms) + len(self.client_companies)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with law firm salary tracking"""
        events = []
        
        # Handle all income sources (including law firm salary tracking)
        self._handle_lumpy_income(date, events)
        
        # Handle recurring expenses
        self._handle_recurring_debits(date, events)
        
        # Handle P2P transfers
        self._handle_professional_retainer_payments(date, events, context)
        self._handle_professional_network_transfers(date, events, context)
        self._handle_professional_networking_transfers(date, events, context)
        self._handle_client_refunds(date, events, context)
        
        # Handle service payments and spending
        self._handle_legal_service_payments(date, events, context)
        self._handle_spending_and_investment(date, events)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
