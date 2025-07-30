import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure  # ✅ NEW: Import for realistic P2P handling

class Lawyer(BaseAgent):
    """
    A multi-dimensional profile for a Lawyer or Consultant.
    Behavior is modified by economic_class and financial_personality.
    Updated with realistic P2P transaction handling.
    """
    def __init__(self, economic_class='Middle', financial_personality='Rational_Investor'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Lawyer / Consultant"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

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
            "device_consistency_score": round(random.uniform(0.88, 0.95), 2),
            "ip_consistency_score": round(random.uniform(0.82, 0.92), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Cards"],
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "High",
        }
        
        super().__init__(**profile_attributes)

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        avg_monthly = (min_mod + max_mod) / 2

        self.payout_months = sorted(random.sample(range(1, 13), k=random.randint(2, 4)))
        self.lump_sum_payment = avg_monthly * (12 / len(self.payout_months))
        self.has_large_cash_reserve = False

        self.junior_retainer_fee = avg_monthly * random.uniform(0.4, 0.6)
        self.loan_emi_amount = avg_monthly * 0.30
        self.prof_indemnity_premium = avg_monthly * 0.5

        self.spend_chance_mod = personality_config['spend_chance_mod']
        self.invest_chance_mod = personality_config['invest_chance_mod']
        
        # ✅ Enhanced P2P attributes - Lawyers have professional and personal networks
        self.professional_network = []  # To be populated by simulation engine
        self.junior_associate = None    # Single junior/associate for retainer payments
        
        self.p2p_transfer_chance = 0.12 * personality_config.get('spend_chance_mod', 1.0)
        self.professional_transfer_chance = 0.08
        self.client_refund_chance = 0.05
        
        # Professional payment cycles
        self.retainer_payment_day = 5

        self.balance = random.uniform(avg_monthly * 1.5, avg_monthly * 3.0)

    def _handle_lumpy_income(self, date, events):
        """Simulates receiving large, infrequent payments."""
        if date.month in self.payout_months and date.day == 15:
            payout = self.lump_sum_payment * random.uniform(0.8, 1.2)
            txn = self.log_transaction("CREDIT", "Client/Project Fee Received", payout, date, channel="Bank Transfer")
            if txn:
                events.append(txn)
                self.has_large_cash_reserve = True

    def _handle_recurring_debits(self, date, events):
        """Handles regular monthly expenses (non-P2P)."""
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", self.loan_emi_amount, date, channel="Auto_Debit")
            if txn: events.append(txn)

        if self.has_insurance_payments and date.month == 7 and date.day == 20:
            txn = self.log_transaction("DEBIT", "Professional Indemnity Insurance", self.prof_indemnity_premium, date, channel="Netbanking")
            if txn: events.append(txn)

    def _handle_professional_retainer_payments(self, date, events, context):
        """✅ UPDATED: Handles regular retainer payments with realistic channels."""
        # Monthly retainer payment to junior associate
        if (date.day == self.retainer_payment_day and 
            self.junior_associate and 
            self.balance > self.junior_retainer_fee):
            
            # ✅ NEW: Select realistic channel based on amount
            if self.junior_retainer_fee > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self,
                'recipient': self.junior_associate,
                'amount': round(self.junior_retainer_fee, 2),
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_professional_network_transfers(self, date, events, context):
        """✅ UPDATED: Handles professional network transfers with realistic channels."""
        if (self.professional_network and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 5000):
            
            recipient = random.choice(self.professional_network)
            
            # Lawyers typically send higher amounts in professional context
            base_amount = random.uniform(2000, 8000)
            
            # Increase amounts if they have large cash reserves
            if self.has_large_cash_reserve:
                base_amount *= random.uniform(1.2, 2.0)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                base_amount *= random.uniform(1.3, 2.0)
            elif self.economic_class in ['Lower', 'Lower_Middle']:
                base_amount *= random.uniform(0.6, 0.8)
            
            # ✅ NEW: Select realistic channel based on amount
            if base_amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])  # Very high amounts use secure channels
            elif base_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(base_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_professional_networking_transfers(self, date, events, context):
        """✅ UPDATED: Handles additional professional network transfers with realistic channels."""
        # Additional professional transfers (referral fees, shared costs, etc.)
        if (self.professional_network and 
            random.random() < self.professional_transfer_chance and 
            self.balance > 10000):
            
            recipient = random.choice(self.professional_network)
            
            # Professional networking amounts are typically moderate to high
            amount = random.uniform(1500, 5000)
            
            # Higher amounts during large cash reserve periods
            if self.has_large_cash_reserve:
                amount *= random.uniform(1.5, 2.5)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                amount *= random.uniform(1.4, 2.2)
            
            # ✅ NEW: Select realistic channel based on amount
            if amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_client_refunds(self, date, events, context):
        """✅ NEW: Handles client refunds and professional service payments."""
        if (self.professional_network and 
            random.random() < self.client_refund_chance and
            self.has_large_cash_reserve and
            self.balance > 15000):
            
            recipient = random.choice(self.professional_network)
            
            # Client refunds are typically larger amounts
            refund_amount = random.uniform(5000, 25000)
            
            # Adjust based on economic class and cash reserves
            if self.economic_class in ['High', 'Upper_Middle']:
                refund_amount *= random.uniform(1.5, 3.0)
            
            # ✅ NEW: Select appropriate channel for larger refunds
            if refund_amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])
            elif refund_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(refund_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_spending_and_investment(self, date, events):
        """Simulates spending and large investments, often after a payout."""
        if self.has_large_cash_reserve:
            if self.has_investment_activity and random.random() < (0.5 * self.invest_chance_mod):
                investment_amount = self.balance * random.uniform(0.3, 0.6)
                investment_type = random.choice(self.investment_types)
                txn = self.log_transaction("DEBIT", f"Lump-Sum Investment - {investment_type}", investment_amount, date, channel="Netbanking")
                if txn:
                    events.append(txn)
                    self.has_large_cash_reserve = False

        if random.random() < (0.4 * self.spend_chance_mod):
            spend_category = random.choice(["Fine Dining", "Travel Booking", "Professional Books", "Legal Software", "Court Fees"])
            amount = random.uniform(1000, 8000)
            txn = self.log_transaction("DEBIT", f"Card Spend - {spend_category}", amount, date, channel="Card")
            if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling with realistic channels."""
        events = []
        self._handle_lumpy_income(date, events)
        self._handle_recurring_debits(date, events)
        self._handle_professional_retainer_payments(date, events, context)     # ✅ Updated with realistic channels
        self._handle_professional_network_transfers(date, events, context)     # ✅ Updated with realistic channels
        self._handle_professional_networking_transfers(date, events, context)  # ✅ Updated with realistic channels
        self._handle_client_refunds(date, events, context)                     # ✅ NEW: Client refunds
        self._handle_spending_and_investment(date, events)
        self._handle_daily_living_expenses(date, events)
        return events
