import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np


class ContentCreator(BaseAgent):
    """
    A multi-dimensional profile for a Content Creator.
    Behavior is modified by economic_class and financial_personality.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Risk_Addict'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Content Creator / Influencer"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "20000-100000"
        min_inc, max_inc = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_inc * income_multiplier)}-{int(max_inc * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Self-Employed",
            "employment_verification": "ITR_Inconsistent",
            "income_type": "Sponsorships, Platform_Payouts",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Erratic_High_Variance",
            "savings_retention_rate": "Low",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "Mostly_On_Time",
            "has_insurance_payments": False,
            "insurance_types": [],
            "utility_payment_status": "Occasionally_Late",
            "mobile_plan_type": "High-Value_Postpaid",
            "device_consistency_score": round(random.uniform(0.50, 0.70), 2),
            "ip_consistency_score": round(random.uniform(0.40, 0.60), 2),
            "sim_churn_rate": "High",
            "primary_digital_channels": ["All"],
            "login_pattern": "Geographically_Dynamic",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
        }
        
        super().__init__(**profile_attributes)

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.avg_monthly_income = random.uniform(min_mod, max_mod)

        self.platform_payout_chance = 0.25 * personality_config['invest_chance_mod']
        self.sponsorship_chance = 0.03
        self.has_sponsorship_funds = False

        self.loan_emi_amount = self.avg_monthly_income * 0.20
        self.software_subscription = random.uniform(2000, 4000)
        self.utility_bill_day = random.randint(20, 28)
        self.late_payment_chance = 0.3
        self.spend_chance_mod = personality_config['spend_chance_mod']

        # ✅ Enhanced P2P attributes - Content creators have diverse networks
        self.collaborators = []  # To be populated by simulation engine
        self.creator_network = []  # Other content creators
        self.brand_contacts = []  # Brand representatives for payments
        self.freelancer_network = []  # Editors, designers, etc.

        self.p2p_transfer_chance = 0.22 * personality_config.get('spend_chance_mod', 1.0)  # Higher frequency
        self.collaboration_payment_chance = 0.18  # Paying collaborators
        self.freelancer_payment_chance = 0.15  # Paying editors, designers
        self.creator_support_chance = 0.08  # Supporting other creators
        self.brand_advance_chance = 0.05  # Giving advances to brands/agencies

        # Track different payment cycles
        self.last_collaboration_payment_date = None
        self.monthly_freelancer_payment_day = random.randint(5, 10)

        self.balance = random.uniform(5000, 20000)

    def _handle_income(self, date, events):
        """Handles erratic income from sponsorships and platform payouts."""
        if random.random() < self.sponsorship_chance:
            sponsorship_amount = self.avg_monthly_income * random.uniform(2.0, 5.0)
            txn = self.log_transaction("CREDIT", "Brand Sponsorship", sponsorship_amount, date, channel="Netbanking")
            if txn:
                events.append(txn)
                self.has_sponsorship_funds = True

        if random.random() < self.platform_payout_chance:
            payout_amount = self.avg_monthly_income * random.uniform(0.1, 0.4)
            source = random.choice(["YouTube AdSense", "Instagram Bonus", "Affiliate Payout"])
            txn = self.log_transaction("CREDIT", source, payout_amount, date, channel="Netbanking")
            if txn: events.append(txn)

    def _handle_fixed_and_professional_expenses(self, date, events):
        """Handles fixed monthly expenses."""
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Equipment Loan EMI", self.loan_emi_amount, date, channel="Auto_Debit")
            if txn: events.append(txn)
            
        if date.day == 5:
            txn = self.log_transaction("DEBIT", "SaaS Subscription (Adobe/Canva)", self.software_subscription, date, channel="Card")
            if txn: events.append(txn)

        payment_day = self.utility_bill_day + (random.randint(1, 5) if random.random() < self.late_payment_chance else 0)
        if date.day == payment_day:
            utility_bill_amount = random.uniform(2000, 5000)
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", utility_bill_amount, date, channel="UPI")
            if txn: events.append(txn)

    def _handle_dynamic_spending(self, date, events):
        """Handles variable spending based on income spikes."""
        spend_chance_multiplier = 5.0 if self.has_sponsorship_funds else 1.0

        if self.has_sponsorship_funds and self.has_investment_activity and random.random() < 0.6:
            spend_category = random.choice(self.investment_types + ["New Camera/Laptop Gear", "Content Trip Booking"])
            amount = self.balance * random.uniform(0.3, 0.7)
            txn = self.log_transaction("DEBIT", spend_category, amount, date, channel="Netbanking")
            if txn:
                events.append(txn)
                self.has_sponsorship_funds = False

        if random.random() < (0.2 * spend_chance_multiplier * self.spend_chance_mod):
             ecommerce_amt = random.uniform(2000, 15000)
             txn = self.log_transaction("DEBIT", "E-commerce (Fashion/Props)", ecommerce_amt, date, channel="Card")
             if txn: events.append(txn)

    def _handle_collaboration_payments(self, date, events, context):
        """✅ Enhanced: Handles payments to collaborators and partners."""
        if (self.collaborators and 
            random.random() < self.collaboration_payment_chance and
            self.balance > 5000):  # Ensure sufficient balance
            
            collaborator = random.choice(self.collaborators)
            
            # Payment amounts vary based on type of collaboration
            base_amount = self.avg_monthly_income * random.uniform(0.1, 0.3)
            
            # Higher payments when having sponsorship funds
            if self.has_sponsorship_funds:
                base_amount *= random.uniform(1.5, 2.5)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                base_amount *= random.uniform(1.3, 2.0)
            
            collaboration_type = random.choice([
                'Video Collaboration Fee',
                'Co-creator Payment',
                'Guest Appearance Fee',
                'Joint Content Revenue',
                'Partnership Share',
                'Collab Project Payment'
            ])
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': collaborator, 
                'amount': round(base_amount, 2), 
                'desc': collaboration_type,
                'channel': 'UPI'
            })

    def _handle_freelancer_payments(self, date, events, context):
        """✅ NEW: Handles payments to editors, designers, and other freelancers."""
        if (self.freelancer_network and 
            date.day == self.monthly_freelancer_payment_day and
            random.random() < self.freelancer_payment_chance and
            self.balance > 3000):
            
            freelancer = random.choice(self.freelancer_network)
            
            # Monthly freelancer payments
            payment_amount = self.avg_monthly_income * random.uniform(0.15, 0.35)
            
            # Higher payments during sponsorship periods
            if self.has_sponsorship_funds:
                payment_amount *= random.uniform(1.2, 1.8)
            
            freelancer_service = random.choice([
                'Video Editing Fee',
                'Graphic Design Payment',
                'Thumbnail Creation',
                'Content Writing Fee',
                'Photography Payment',
                'Audio Editing Fee',
                'Animation Service'
            ])
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': freelancer, 
                'amount': round(payment_amount, 2), 
                'desc': freelancer_service,
                'channel': 'UPI'
            })

    def _handle_creator_network_transfers(self, date, events, context):
        """✅ NEW: Handles transfers within creator community."""
        if (self.creator_network and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 2000):
            
            recipient = random.choice(self.creator_network)
            
            # Creator-to-creator transfers
            base_amount = random.uniform(1000, 5000)
            
            # Increase during high-income periods
            if self.has_sponsorship_funds:
                base_amount *= random.uniform(1.5, 2.0)
            
            # Adjust based on financial personality
            if self.financial_personality == 'Risk_Addict':
                base_amount *= random.uniform(1.2, 1.8)
            elif self.financial_personality == 'Saver':
                base_amount *= random.uniform(0.6, 1.0)
            
            transfer_desc = random.choice([
                'Creator Support',
                'Equipment Sharing',
                'Joint Project Fund',
                'Creator Emergency Help',
                'Content Collaboration',
                'Influencer Network Support',
                'Creative Community Aid'
            ])
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(base_amount, 2), 
                'desc': transfer_desc,
                'channel': 'UPI'
            })

    def _handle_brand_advance_payments(self, date, events, context):
        """✅ NEW: Handles advance payments to brands/agencies."""
        if (self.brand_contacts and 
            random.random() < self.brand_advance_chance and
            self.balance > 10000 and
            self.has_sponsorship_funds):  # Only when flush with cash
            
            brand_contact = random.choice(self.brand_contacts)
            
            # Advance payments are typically larger
            advance_amount = self.avg_monthly_income * random.uniform(0.2, 0.5)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                advance_amount *= random.uniform(1.5, 2.5)
            
            advance_desc = random.choice([
                'Brand Campaign Advance',
                'Sponsorship Prepayment',
                'Marketing Agency Advance',
                'Content Series Advance',
                'Brand Partnership Deposit'
            ])
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': brand_contact, 
                'amount': round(advance_amount, 2), 
                'desc': advance_desc,
                'channel': 'UPI'
            })

    def _handle_creator_support_activities(self, date, events, context):
        """✅ NEW: Handles supporting other creators during tough times."""
        if (self.creator_network and 
            random.random() < self.creator_support_chance and
            self.balance > 8000):  # Need good balance for support activities
            
            recipient = random.choice(self.creator_network)
            
            # Support amounts based on current financial status
            support_amount = random.uniform(2000, 8000)
            
            # More generous during sponsorship periods
            if self.has_sponsorship_funds:
                support_amount *= random.uniform(1.3, 2.0)
            
            # Risk addicts are more generous with support
            if self.financial_personality == 'Risk_Addict':
                support_amount *= random.uniform(1.2, 1.6)
            
            support_desc = random.choice([
                'Creator Emergency Fund',
                'Equipment Help',
                'Channel Boost Support',
                'Creative Crisis Help',
                'Influencer Solidarity',
                'Community Support Fund'
            ])
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(support_amount, 2), 
                'desc': support_desc,
                'channel': 'UPI'
            })

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling."""
        events = []
        self._handle_income(date, events)
        self._handle_fixed_and_professional_expenses(date, events)
        self._handle_dynamic_spending(date, events)
        self._handle_collaboration_payments(date, events, context)      # ✅ Enhanced collaboration payments
        self._handle_freelancer_payments(date, events, context)         # ✅ Monthly freelancer payments
        self._handle_creator_network_transfers(date, events, context)   # ✅ Creator community transfers
        self._handle_brand_advance_payments(date, events, context)      # ✅ Brand advance payments
        self._handle_creator_support_activities(date, events, context)  # ✅ Creator support activities
        self._handle_daily_living_expenses(date, events)
        return events
