import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

class Doctor(BaseAgent):
    """
    Enhanced Doctor agent for Phase 2: Hospital salary source tracking
    Includes hospital companies as employers, realistic employment relationships,
    and enhanced behavioral diversity for GNN fraud detection.
    """
    def __init__(self, economic_class='Upper_Middle', financial_personality='Rational_Investor'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Doctor"

        # Risk calculation
        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        # Income calculation
        base_income_range = "70000-300000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-Employed_Professional",
            "employment_verification": "Medical_License_Verified",
            "income_type": "Professional_Fees",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Lumpy",
            "savings_retention_rate": "High",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life", "Prof_Indemnity"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "High-Value_Postpaid",
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            "device_consistency_score": round(random.uniform(0.85, 0.96), 3),
            "ip_consistency_score": round(random.uniform(0.85, 0.96), 3),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Cards"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
            
            # ✅ NEW: Heterogeneous graph connections specific to Doctor
            "industry_sector": "Healthcare_Medical",
            "company_size": "Small_Practice",
        }
        
        super().__init__(**profile_attributes)

        # Financial calculations with more variation
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_income = random.uniform(min_mod, max_mod)  # More realistic variation

        # ✅ NEW: Hospital companies as employers (salary source tracking)
        self.hospital_affiliations = []  # Apollo, Fortis, Max as company nodes
        self.primary_hospital_id = None  # Main income source hospital
        self.hospital_payout_schedule = {}  # Track payout schedules per hospital

        # ✅ NEW: Employment relationship tracking
        self.consultation_schedule_consistency = random.uniform(0.8, 0.95)  # Schedule reliability
        self.monthly_hospital_payout_day = random.randint(28, 31)  # End of month payouts
        self.last_hospital_payout_date = None

        # Income patterns with enhanced tracking
        self.consultation_chance = random.uniform(0.80, 0.90)  # More variation
        self.avg_consultation_fee = self.avg_monthly_income / random.uniform(35, 45)  # More realistic
        self.large_payout_chance = random.uniform(0.20, 0.30)  # Personality-based

        # Fixed expenses with more variation
        self.clinic_rent_amount = self.avg_monthly_income * random.uniform(0.15, 0.25)
        self.sip_amount = self.avg_monthly_income * random.uniform(0.20, 0.30) * personality_config.get('invest_chance_mod', 1.0)
        self.loan_emi_amount = self.avg_monthly_income * random.uniform(0.25, 0.35)
        self.prof_indemnity_premium = self.avg_monthly_income * random.uniform(0.04, 0.06)
        self.high_end_spend_chance = random.uniform(0.08, 0.12) * personality_config.get('spend_chance_mod', 1.0)
        
        # ✅ Enhanced P2P networks for Doctors
        self.service_providers = []  # Medical equipment, lab services
        self.professional_network = []  # Other healthcare professionals
        self.family_dependents = []  # Family members for support
        
        # ✅ NEW: Enhanced heterogeneous connections
        self.hospital_affiliations = []  # Hospital/clinic company nodes
        self.medical_equipment_vendors = []  # Equipment merchant relationships
        self.pharmaceutical_suppliers = []  # Medicine/supply merchants
        self.lab_service_providers = []  # Diagnostic lab relationships
        
        # P2P transfer probabilities with more variation
        self.p2p_payment_chance = random.uniform(0.18, 0.22)
        self.professional_transfer_chance = random.uniform(0.10, 0.14)
        self.family_support_chance = random.uniform(0.12, 0.18)
        self.equipment_purchase_chance = random.uniform(0.06, 0.10)
        self.lab_payment_chance = random.uniform(0.15, 0.21)

        # Temporal tracking with enhanced cycles
        self.last_equipment_purchase_date = None
        self.consultation_peak_periods = []  # Track busy periods
        self.seasonal_income_patterns = {}  # Track seasonal variations

        self.balance = random.uniform(self.avg_monthly_income * 0.8, self.avg_monthly_income * 2.5)

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Doctors typically have 2-4 devices (phone, tablet, laptop, clinic systems)"""
        device_options = [2, 3, 4]
        weights = [0.3, 0.5, 0.2]  # Most have 3 devices
        return random.choices(device_options, weights=weights)[0]

    def assign_hospital_affiliations(self, hospital_company_ids):
        """✅ NEW: Assign hospital companies as employers for salary tracking"""
        self.hospital_affiliations = hospital_company_ids
        
        if hospital_company_ids:
            # Assign primary hospital as main employer
            self.primary_hospital_id = random.choice(hospital_company_ids)
            self.assign_employer(
                company_id=self.primary_hospital_id,
                employment_start_date=datetime.now().date() - timedelta(days=random.randint(365, 3650))
            )
            
            # Set up payout schedules for each hospital
            for hospital_id in hospital_company_ids:
                self.hospital_payout_schedule[hospital_id] = {
                    'monthly_payout_day': random.randint(28, 31),
                    'consultation_fee_share': random.uniform(0.6, 0.8),
                    'procedure_fee_share': random.uniform(0.7, 0.9)
                }

    def _handle_hospital_salary_payouts(self, date, events):
        """✅ NEW: Handle hospital payouts as salary from employer companies"""
        # Monthly hospital payouts as salary (end of month)
        if (self.hospital_affiliations and 
            date.day == self.monthly_hospital_payout_day and
            random.random() < self.consultation_schedule_consistency):
            
            hospital_id = self.primary_hospital_id or random.choice(self.hospital_affiliations)
            
            # Calculate monthly payout with variance
            base_amount = self.avg_monthly_income * random.uniform(0.4, 0.7)  # Hospital share
            
            # Add seasonal variations (medical tourism, health checkups)
            month_multiplier = {
                12: 1.2,  # December - year-end health checkups
                11: 1.1,  # November - pre-winter checkups
                1: 0.9,   # January - post-holiday low
                4: 1.1,   # April - summer prep
            }.get(date.month, 1.0)
            
            final_amount = base_amount * month_multiplier
            
            # ✅ NEW: Log as salary transaction from company
            txn = self.log_salary_transaction(
                amount=final_amount,
                date=date,
                company_id=hospital_id
            )
            
            if txn:
                txn['transaction_category'] = 'hospital_salary_payout'
                txn['company_type'] = 'healthcare_provider'
                events.append(txn)
                self.last_hospital_payout_date = date

    def _handle_income(self, date, events):
        """✅ UPDATED: Enhanced income handling with company salary tracking"""
        # Hospital salary payouts
        self._handle_hospital_salary_payouts(date, events)

        # Daily consultation income (private practice)
        if (date.weekday() < 5 and  # Weekdays
            random.random() < self.consultation_chance):
            
            daily_consult_income = self.avg_consultation_fee * random.uniform(0.6, 1.4)
            
            # ✅ NEW: Some consultations through hospital (tracked as secondary income)
            if (self.hospital_affiliations and 
                random.random() < 0.4):  # 40% hospital consultations
                
                hospital_id = random.choice(self.hospital_affiliations)
                txn = self.log_salary_transaction(
                    amount=daily_consult_income,
                    date=date,
                    company_id=hospital_id
                )
                if txn:
                    txn['transaction_category'] = 'hospital_consultation_fee'
                    events.append(txn)
            else:
                # Private consultation
                txn = self.log_transaction(
                    "CREDIT", "Private Consultation Fees", daily_consult_income, date, channel="UPI"
                )
                if txn:
                    events.append(txn)

        # Large medical payouts (surgeries, procedures)
        if (date.day in [15, 30] and 
            random.random() < self.large_payout_chance):
            
            payout_amount = self.avg_monthly_income * random.uniform(1.2, 2.8)
            procedure_type = random.choice(["Surgery Fee", "Procedure Fee", "Emergency Consultation"])
            
            # ✅ NEW: Large payouts mostly from hospital affiliations
            if (self.hospital_affiliations and 
                random.random() < 0.7):  # 70% from hospitals
                
                hospital_id = random.choice(self.hospital_affiliations)
                txn = self.log_salary_transaction(
                    amount=payout_amount,
                    date=date,
                    company_id=hospital_id
                )
                if txn:
                    txn['transaction_category'] = 'major_procedure_fee'
                    events.append(txn)
            else:
                txn = self.log_transaction(
                    "CREDIT", procedure_type, payout_amount, date, channel="Bank_Transfer"
                )
                if txn:
                    events.append(txn)

    def _handle_professional_and_fixed_expenses(self, date, events):
        """✅ UPDATED: Enhanced professional expenses with merchant tracking"""
        # Clinic rent with variation
        clinic_rent_day = random.randint(3, 7)  # More realistic rent payment days
        if date.day == clinic_rent_day:
            # Add some variation to rent amount
            rent_variation = random.uniform(0.95, 1.05)
            actual_rent = self.clinic_rent_amount * rent_variation
            
            clinic_merchant_id = f"clinic_rental_{hash(self.agent_id) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=clinic_merchant_id,
                amount=actual_rent,
                description="Medical Clinic Rent",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

        # Loan EMI with enhanced tracking
        emi_day = random.randint(8, 12)
        if self.has_loan_emi and date.day == emi_day:
            # Add variation for processing fees
            emi_variation = random.uniform(0.98, 1.02)
            actual_emi = self.loan_emi_amount * emi_variation
            
            txn = self.log_transaction(
                "DEBIT", "Medical Equipment Loan EMI", actual_emi, date, channel="Auto_Debit"
            )
            if txn:
                events.append(txn)
            
        # Investment SIP with enhanced merchant tracking
        sip_day = random.randint(13, 17)
        if self.has_investment_activity and date.day == sip_day:
            # Add variation to SIP amount
            sip_variation = random.uniform(0.9, 1.1)
            actual_sip = self.sip_amount * sip_variation
            
            investment_merchant_id = f"mutual_fund_{hash(self.agent_id) % 500}"
            
            txn = self.log_merchant_transaction(
                merchant_id=investment_merchant_id,
                amount=actual_sip,
                description="Healthcare Sector Mutual Fund SIP",
                date=date,
                channel="Auto_Debit"
            )
            if txn:
                events.append(txn)

        # Professional indemnity insurance (annual/quarterly)
        if (self.has_insurance_payments and 
            date.month == 6 and 
            date.day == random.randint(18, 22)):
            
            insurance_merchant_id = f"medical_insurance_{hash(self.agent_id) % 200}"
            
            txn = self.log_merchant_transaction(
                merchant_id=insurance_merchant_id,
                amount=self.prof_indemnity_premium,
                description="Professional Indemnity Insurance",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)

    def _handle_discretionary_spending(self, date, events):
        """✅ UPDATED: Enhanced high-end spending with merchant diversity"""
        if random.random() < self.high_end_spend_chance:
            spend_categories = [
                ("Fine_Dining", random.uniform(3000, 12000)),
                ("Luxury_Goods", random.uniform(8000, 35000)),
                ("Travel_Booking", random.uniform(15000, 50000)),
                ("Premium_Electronics", random.uniform(20000, 80000)),
                ("Medical_Conference", random.uniform(10000, 30000))
            ]
            
            spend_category, spend_amount = random.choice(spend_categories)
            
            # Economic class adjustments
            if self.economic_class == 'High':
                spend_amount *= random.uniform(1.5, 2.5)
            
            # ✅ NEW: Enhanced merchant tracking
            luxury_merchant_id = f"premium_{spend_category}_{hash(self.agent_id + str(date)) % 1000}"
            
            txn = self.log_merchant_transaction(
                merchant_id=luxury_merchant_id,
                amount=spend_amount,
                description=f"Premium {spend_category.replace('_', ' ')}",
                date=date,
                channel="Card"
            )
            if txn:
                events.append(txn)

    def add_medical_equipment_vendor(self, vendor_id, first_purchase_date=None):
        """✅ NEW: Track medical equipment vendor relationships"""
        if vendor_id not in self.medical_equipment_vendors:
            self.medical_equipment_vendors.append(vendor_id)
            self.add_frequent_merchant(vendor_id, first_purchase_date)

    def add_lab_service_provider(self, lab_id, first_service_date=None):
        """✅ NEW: Track diagnostic lab service relationships"""
        if lab_id not in self.lab_service_providers:
            self.lab_service_providers.append(lab_id)
            self.add_frequent_merchant(lab_id, first_service_date)

    def _handle_professional_service_payments(self, date, events, context):
        """✅ UPDATED: Professional service payments with enhanced tracking"""
        monthly_payment_day = random.randint(23, 27)
        if (date.day == monthly_payment_day and 
            self.service_providers and 
            random.random() < self.p2p_payment_chance and
            self.balance > 15000):
            
            provider = random.choice(self.service_providers)
            amount = self.avg_monthly_income * random.uniform(0.08, 0.18)
            
            # Economic class adjustments
            if self.economic_class == 'High':
                amount *= random.uniform(1.3, 1.8)
            
            # ✅ NEW: Select realistic channel based on amount
            if amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': provider, 
                'amount': round(amount, 2), 
                'desc': 'Medical Service Payment',
                'channel': channel,
                'transaction_category': 'professional_service'
            })

    def _handle_professional_network_transfers(self, date, events, context):
        """✅ UPDATED: Medical professional network transfers"""
        if (self.professional_network and 
            random.random() < self.professional_transfer_chance and
            self.balance > 20000):
            
            colleague = random.choice(self.professional_network)
            
            # Professional transfers (referrals, consultations, joint procedures)
            base_amount = self.avg_monthly_income * random.uniform(0.04, 0.12)
            
            # Economic class and personality adjustments
            if self.economic_class == 'High':
                base_amount *= random.uniform(1.4, 2.2)
                
            if self.financial_personality == 'Rational_Investor':
                base_amount *= random.uniform(1.0, 1.3)
            
            # ✅ NEW: Select realistic channel
            if base_amount > 75000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': colleague, 
                'amount': round(base_amount, 2), 
                'desc': 'Professional Referral Payment',
                'channel': channel,
                'transaction_category': 'professional_referral'
            })

    def _handle_family_support_transfers(self, date, events, context):
        """✅ UPDATED: Family support with enhanced patterns"""
        if (self.family_dependents and 
            date.day == 1 and  # Monthly family support
            random.random() < self.family_support_chance and
            self.balance > 30000):
            
            family_member = random.choice(self.family_dependents)
            
            # Family support amounts based on income and dependents
            support_percentage = random.uniform(0.12, 0.25)
            support_amount = self.avg_monthly_income * support_percentage
            
            # Adjust based on number of dependents
            if len(self.family_dependents) > 2:
                support_amount *= random.uniform(1.2, 1.5)
            
            # Personality adjustments
            if self.financial_personality == 'Saver':
                support_amount *= random.uniform(0.9, 1.1)
            elif self.financial_personality == 'Over_Spender':
                support_amount *= random.uniform(1.1, 1.4)
            
            # ✅ NEW: Select realistic channel
            if support_amount > 100000:
                channel = random.choice(['IMPS', 'NEFT'])
            elif support_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': family_member, 
                'amount': round(support_amount, 2), 
                'desc': 'Family Financial Support',
                'channel': channel,
                'transaction_category': 'family_support'
            })

    def _handle_medical_equipment_purchases(self, date, events, context):
        """✅ UPDATED: Medical equipment and supply purchases"""
        equipment_purchase_day = random.randint(18, 22)
        if (date.day == equipment_purchase_day and 
            random.random() < self.equipment_purchase_chance and
            self.balance > 40000):
            
            # Equipment purchases are significant investments
            equipment_amount = self.avg_monthly_income * random.uniform(0.25, 1.2)
            
            # Higher amounts for high economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                equipment_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Enhanced medical equipment vendor tracking
            equipment_vendor_id = f"medical_equipment_{hash(self.agent_id + str(date)) % 1000}"
            self.add_medical_equipment_vendor(equipment_vendor_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=equipment_vendor_id,
                amount=equipment_amount,
                description="Professional Medical Equipment",
                date=date,
                channel="Netbanking"
            )
            if txn:
                events.append(txn)
                self.last_equipment_purchase_date = date

    def _handle_lab_service_payments(self, date, events, context):
        """✅ UPDATED: Payments to diagnostic lab services"""
        if (random.random() < self.lab_payment_chance and
            self.balance > 8000):
            
            # Regular lab service payments with variation
            lab_payment = self.avg_monthly_income * random.uniform(0.015, 0.06)
            
            # ✅ NEW: Enhanced lab service relationship tracking
            lab_id = f"diagnostic_lab_{hash(self.agent_id) % 200}"
            self.add_lab_service_provider(lab_id, date)
            
            txn = self.log_merchant_transaction(
                merchant_id=lab_id,
                amount=lab_payment,
                description="Diagnostic Lab Services",
                date=date,
                channel="UPI"
            )
            if txn:
                events.append(txn)

    def get_doctor_specific_features(self):
        """✅ ENHANCED: Comprehensive doctor-specific features"""
        return {
            'hospital_employer_count': len(self.hospital_affiliations),
            'primary_hospital_tenure': self.get_employment_tenure_months(),
            'consultation_schedule_consistency': self.consultation_schedule_consistency,
            'equipment_vendor_relationships': len(self.medical_equipment_vendors),
            'lab_service_relationships': len(self.lab_service_providers),
            'professional_network_size': len(self.professional_network),
            'service_provider_count': len(self.service_providers),
            'family_support_obligations': len(self.family_dependents),
            'consultation_frequency_score': self.consultation_chance,
            'equipment_investment_activity': self.equipment_purchase_chance,
            'last_hospital_payout_recency': (datetime.now().date() - self.last_hospital_payout_date).days if self.last_hospital_payout_date else 999,
            'total_company_relationships': len(self.hospital_affiliations)
        }

    def act(self, date: datetime, **context):
        """✅ UPDATED: Enhanced behavior with hospital salary tracking"""
        events = []
        
        # Handle all income sources (including hospital salary tracking)
        self._handle_income(date, events)
        
        # Handle professional and fixed expenses
        self._handle_professional_and_fixed_expenses(date, events)
        
        # Handle discretionary spending
        self._handle_discretionary_spending(date, events)
        
        # Handle P2P transfers
        self._handle_professional_service_payments(date, events, context)
        self._handle_professional_network_transfers(date, events, context)
        self._handle_family_support_transfers(date, events, context)
        
        # Handle equipment and service purchases
        self._handle_medical_equipment_purchases(date, events, context)
        self._handle_lab_service_payments(date, events, context)
        
        # Handle daily living expenses
        self._handle_daily_living_expenses(date, events)
        
        return events
