import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure  # ✅ NEW: Import for realistic P2P handling

class TechProfessional(BaseAgent):
    """
    A multi-dimensional profile for a Tech Professional.
    Behavior is modified by economic_class and financial_personality.
    Updated with realistic P2P transaction handling.
    """
    def __init__(self, economic_class='Upper_Middle', financial_personality='Rational_Investor'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Tech Professional"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "60000-200000"
        min_sal, max_sal = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_sal * income_multiplier)}-{int(max_sal * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name,
            "risk_profile": risk_profile_category,
            "risk_score": risk_score,
            "economic_class": economic_class,
            "financial_personality": financial_personality,
            "employment_status": "Salaried",
            "employment_verification": "EPFO_Verified",
            "income_type": "Salary_IT",
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date_with_Bonus",
            "savings_retention_rate": "Medium",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME",
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life"],
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "High-Value_Postpaid",
            "device_consistency_score": round(random.uniform(0.92, 0.98), 2),
            "ip_consistency_score": round(random.uniform(0.60, 0.75), 2),
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["All"],
            "login_pattern": "Geographically_Dynamic",
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "High",
        }
        
        super().__init__(**profile_attributes)

        self.salary_day = random.randint(1, 5)
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_mod, max_mod)
        
        self.stock_investment_amount = self.salary_amount * random.uniform(0.15, 0.25) * personality_config['invest_chance_mod']
        self.crypto_investment_amount = self.salary_amount * random.uniform(0.05, 0.10) * (2 if financial_personality == 'Risk_Addict' else 1)
        self.loan_emi_amount = self.salary_amount * 0.20
        self.saas_subscription_amount = random.uniform(500, 2000)

        self.is_traveling = False
        self.travel_start_day = 0
        self.travel_duration = 0
        
        self.annual_bonus_month = random.choice([3, 4])
        self.has_received_bonus_this_year = False

        # ✅ Enhanced P2P attributes - Tech professionals have diverse networks
        self.contacts = []  # To be populated by simulation engine
        self.professional_network = []  # Tech colleagues, freelancers, etc.
        self.family_dependents = []  # Family members they support
        
        self.p2p_transfer_chance = 0.18 * personality_config.get('spend_chance_mod', 1.0)
        self.professional_transfer_chance = 0.12
        self.family_support_chance = 0.10
        
        # Special transfer occasions
        self.bonus_sharing_chance = 0.25
        self.has_shared_bonus_this_year = False

        self.balance = random.uniform(self.salary_amount * 0.5, self.salary_amount)

    def _handle_income(self, date, events):
        """Handles monthly salary and large annual bonuses/stock sales."""
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Salary Credit (IT)", self.salary_amount, date, channel="Bank Transfer")
            if txn: events.append(txn)
        
        if date.month == self.annual_bonus_month and date.day == self.salary_day and not self.has_received_bonus_this_year:
            bonus_amount = self.salary_amount * random.uniform(3.0, 6.0)
            txn = self.log_transaction("CREDIT", "Annual Bonus/RSU Sale", bonus_amount, date, channel="Bank Transfer")
            if txn: events.append(txn)
            self.has_received_bonus_this_year = True
        
        if date.month == 1: 
            self.has_received_bonus_this_year = False
            self.has_shared_bonus_this_year = False

    def _handle_fixed_debits(self, date, events):
        """Handles recurring payments for loans, utilities, and investments."""
        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", self.loan_emi_amount, date, channel="Auto_Debit")
            if txn: events.append(txn)
            
        if self.has_investment_activity:
            if "Stocks" in self.investment_types and date.day == 5:
                txn = self.log_transaction("DEBIT", "Stock Investment (Zerodha)", self.stock_investment_amount, date, channel="Netbanking")
                if txn: events.append(txn)
            if "Crypto" in self.investment_types and date.day == 15:
                txn = self.log_transaction("DEBIT", "Crypto Investment (WazirX)", self.crypto_investment_amount, date, channel="Netbanking")
                if txn: events.append(txn)

        if date.day == 20:
            txn = self.log_transaction("DEBIT", "SaaS Subscriptions (Cloud/VPN)", self.saas_subscription_amount, date, channel="Card")
            if txn: events.append(txn)

    def _handle_dynamic_spending(self, date, events):
        """Simulates their dynamic lifestyle, including travel."""
        if date.day == 1 and date.month in [1, 4, 7, 10]:
            if random.random() < 0.5:
                self.is_traveling = True
                self.travel_start_day = random.randint(5, 15)
                self.travel_duration = random.randint(7, 14)
                
                travel_cost = random.uniform(20000, 80000) * (1.5 if self.economic_class in ['Upper_Middle', 'High'] else 1)
                txn = self.log_transaction("DEBIT", "Travel Booking (Flights/Hotels)", travel_cost, date, channel="Card")
                if txn: events.append(txn)
        
        if self.is_traveling:
            if date.day >= self.travel_start_day and date.day < self.travel_start_day + self.travel_duration:
                if random.random() < 0.9:
                    spend = random.uniform(1000, 5000)
                    txn = self.log_transaction("DEBIT", "Travel/Forex Card Spend", spend, date, channel="Card")
                    if txn: events.append(txn)
            else:
                self.is_traveling = False
        else:
            if random.random() < 0.3:
                spend = random.uniform(1500, 6000)
                txn = self.log_transaction("DEBIT", "E-commerce/Dining", spend, date, channel="Card")
                if txn: events.append(txn)

    def _handle_social_p2p_transfers(self, date, events, context):
        """✅ UPDATED: Handles social and personal P2P transfers with realistic channels."""
        if (self.contacts and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 2000):
            
            recipient = random.choice(self.contacts)
            
            # Tech professionals typically send moderate to high amounts
            base_amount = random.uniform(1000, 5000)
            
            # Increase amounts based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                base_amount *= random.uniform(1.2, 2.0)
            
            # Adjust based on financial personality
            if self.financial_personality == 'Over_Spender':
                base_amount *= random.uniform(1.3, 1.8)
            elif self.financial_personality == 'Saver':
                base_amount *= random.uniform(0.7, 1.0)
            
            # ✅ NEW: Select realistic channel based on amount
            if base_amount > 50000:
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

    def _handle_professional_transfers(self, date, events, context):
        """✅ UPDATED: Handles professional and freelance-related transfers with realistic channels."""
        if (self.professional_network and 
            random.random() < self.professional_transfer_chance and
            self.balance > 5000):
            
            recipient = random.choice(self.professional_network)
            
            # Professional transfer amounts are typically higher
            amount = random.uniform(2000, 10000)
            
            # Higher amounts for high economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                amount *= random.uniform(1.5, 2.5)
            
            # ✅ NEW: Select realistic channel based on amount
            if amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])  # Very high amounts use secure channels
            elif amount > 50000:
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

    def _handle_family_support(self, date, events, context):
        """✅ UPDATED: Handles family support transfers with realistic channels."""
        if (self.family_dependents and 
            random.random() < self.family_support_chance and
            self.balance > 10000):
            
            recipient = random.choice(self.family_dependents)
            
            # Family support amounts based on salary
            support_amount = self.salary_amount * random.uniform(0.05, 0.15)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                support_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Select realistic channel based on amount
            if support_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(support_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_bonus_sharing(self, date, events, context):
        """✅ UPDATED: Handles increased transfers after receiving bonus with realistic channels."""
        if (self.has_received_bonus_this_year and 
            not self.has_shared_bonus_this_year and
            date.month == self.annual_bonus_month and
            date.day >= self.salary_day + 3 and  # Few days after bonus
            random.random() < self.bonus_sharing_chance):
            
            # Share bonus with family or close contacts
            recipients = []
            if self.family_dependents:
                recipients.extend(self.family_dependents[:2])  # Max 2 family members
            if self.contacts and len(recipients) < 2:
                recipients.extend(random.sample(self.contacts, min(2, len(self.contacts))))
            
            if recipients:
                for recipient in recipients[:2]:  # Limit to 2 recipients
                    # Bonus sharing is typically generous
                    bonus_share = self.salary_amount * random.uniform(0.3, 0.8)
                    
                    # ✅ NEW: Select appropriate channel for large bonus shares
                    if bonus_share > 100000:
                        channel = random.choice(['NEFT', 'RTGS'])
                    elif bonus_share > 50000:
                        channel = random.choice(['IMPS', 'NEFT'])
                    else:
                        channel = RealisticP2PStructure.select_realistic_channel()
                    
                    context.get('p2p_transfers', []).append({
                        'sender': self, 
                        'recipient': recipient, 
                        'amount': round(bonus_share, 2), 
                        'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                        'channel': channel  # ✅ Realistic channel
                    })
                
                self.has_shared_bonus_this_year = True

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling with realistic channels."""
        events = []
        self._handle_income(date, events)
        self._handle_fixed_debits(date, events)
        self._handle_dynamic_spending(date, events)
        self._handle_social_p2p_transfers(date, events, context)     # ✅ Updated with realistic channels
        self._handle_professional_transfers(date, events, context)   # ✅ Updated with realistic channels
        self._handle_family_support(date, events, context)          # ✅ Updated with realistic channels
        self._handle_bonus_sharing(date, events, context)           # ✅ Updated with realistic channels
        self._handle_daily_living_expenses(date, events)
        return events
