import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure  # ✅ NEW: Import for realistic P2P handling

class Doctor(BaseAgent):
    """
    A multi-dimensional profile for a Doctor.
    Behavior is modified by economic_class and financial_personality.
    Updated with realistic P2P transaction handling.
    """
    def __init__(self, economic_class='Upper_Middle', financial_personality='Rational_Investor'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Doctor / Healthcare Worker"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

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
            "device_consistency_score": round(random.uniform(0.88, 0.96), 2),
            "ip_consistency_score": round(random.uniform(0.88, 0.96), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Cards"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
        }
        
        super().__init__(**profile_attributes)

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_income = (min_mod + max_mod) / 2

        self.consultation_chance = 0.85
        self.avg_consultation_fee = self.avg_monthly_income / 40
        self.large_payout_chance = 0.25

        self.clinic_rent_amount = self.avg_monthly_income * 0.20
        self.sip_amount = self.avg_monthly_income * 0.25 * personality_config['invest_chance_mod']
        self.loan_emi_amount = self.avg_monthly_income * 0.30
        self.prof_indemnity_premium = self.avg_monthly_income * 0.5
        self.high_end_spend_chance = 0.10 * personality_config['spend_chance_mod']
        
        # ✅ Enhanced P2P attributes - Doctors have professional networks
        self.service_providers = []  # To be populated by simulation engine
        self.professional_network = []  # Other healthcare professionals
        self.family_dependents = []  # Family members for support
        
        self.p2p_payment_chance = 0.20  # Professional service payments
        self.professional_transfer_chance = 0.12  # Transfers to medical colleagues
        self.family_support_chance = 0.15  # Supporting family members

        self.balance = random.uniform(self.avg_monthly_income, self.avg_monthly_income * 2)

    def _handle_income(self, date, events):
        """Handles consultation fees and large medical payouts."""
        if date.weekday() < 5 and random.random() < self.consultation_chance:
            daily_consult_income = self.avg_consultation_fee * random.uniform(0.8, 1.2)
            txn = self.log_transaction("CREDIT", "Daily Consultation Fees", daily_consult_income, date, channel="UPI")
            if txn: events.append(txn)
        
        if date.day == 15 and random.random() < self.large_payout_chance:
            payout_amount = self.avg_monthly_income * random.uniform(1.5, 3.0)
            source = random.choice(["Gateway Payout", "Surgery Fee"])
            txn = self.log_transaction("CREDIT", source, payout_amount, date, channel="Bank Transfer")
            if txn: events.append(txn)

    def _handle_professional_and_fixed_expenses(self, date, events):
        """Handles professional and fixed monthly expenses."""
        if date.day == 5:
            txn = self.log_transaction("DEBIT", "Clinic Rent", self.clinic_rent_amount, date, channel="Netbanking")
            if txn: events.append(txn)

        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", self.loan_emi_amount, date, channel="Auto_Debit")
            if txn: events.append(txn)
            
        if self.has_investment_activity and date.day == 15:
            txn = self.log_transaction("DEBIT", "Mutual Fund SIP", self.sip_amount, date, channel="Auto_Debit")
            if txn: events.append(txn)

        if self.has_insurance_payments and date.month == 6 and date.day == 20:
             txn = self.log_transaction("DEBIT", "Professional Indemnity Insurance", self.prof_indemnity_premium, date, channel="Netbanking")
             if txn: events.append(txn)

    def _handle_discretionary_spending(self, date, events):
        """Handles high-end discretionary spending typical for doctors."""
        if random.random() < self.high_end_spend_chance:
            spend_category = random.choice(["Fine Dining", "Luxury Goods", "Travel Booking", "Electronics"])
            spend_amount = random.uniform(5000, 25000)
            txn = self.log_transaction("DEBIT", f"Card Spend - {spend_category}", spend_amount, date, channel="Card")
            if txn: events.append(txn)

    def _handle_professional_service_payments(self, date, events, context):
        """✅ UPDATED: Handles payments to service providers with realistic channels."""
        if (date.day == 25 and 
            self.service_providers and 
            random.random() < self.p2p_payment_chance and
            self.balance > 10000):
            
            provider = random.choice(self.service_providers)
            amount = self.avg_monthly_income * random.uniform(0.1, 0.2)
            
            # ✅ NEW: Select realistic channel based on amount
            if amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])  # High amounts use secure channels
            elif amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': provider, 
                'amount': round(amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_professional_network_transfers(self, date, events, context):
        """✅ NEW: Handles transfers to other medical professionals."""
        if (self.professional_network and 
            random.random() < self.professional_transfer_chance and
            self.balance > 15000):
            
            colleague = random.choice(self.professional_network)
            
            # Professional transfers (consultation referrals, joint procedures, etc.)
            base_amount = self.avg_monthly_income * random.uniform(0.05, 0.15)
            
            # Adjust based on economic class
            if self.economic_class == 'High':
                base_amount *= random.uniform(1.5, 2.0)
            
            # ✅ NEW: Select realistic channel
            if base_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': colleague, 
                'amount': round(base_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_family_support_transfers(self, date, events, context):
        """✅ NEW: Handles family support transfers."""
        if (self.family_dependents and 
            date.day == 1 and  # Monthly family support on 1st
            random.random() < self.family_support_chance and
            self.balance > 20000):
            
            family_member = random.choice(self.family_dependents)
            
            # Family support amounts based on income
            support_amount = self.avg_monthly_income * random.uniform(0.15, 0.30)
            
            # Adjust based on financial personality
            if self.financial_personality == 'Saver':
                support_amount *= random.uniform(0.8, 1.0)
            elif self.financial_personality == 'Over_Spender':
                support_amount *= random.uniform(1.2, 1.5)
            
            # ✅ NEW: Select realistic channel
            if support_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': family_member, 
                'amount': round(support_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling with realistic channels."""
        events = []
        self._handle_income(date, events)
        self._handle_professional_and_fixed_expenses(date, events)
        self._handle_discretionary_spending(date, events)
        self._handle_professional_service_payments(date, events, context)  # ✅ Updated with realistic channels
        self._handle_professional_network_transfers(date, events, context)  # ✅ NEW: Professional network transfers
        self._handle_family_support_transfers(date, events, context)        # ✅ NEW: Family support transfers
        self._handle_daily_living_expenses(date, events)
        return events
