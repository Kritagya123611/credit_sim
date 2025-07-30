import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure  # ✅ NEW: Import for realistic P2P handling

class SalariedProfessional(BaseAgent):
    """
    A multi-dimensional profile for a Salaried Professional.
    Behavior is modified by economic_class and financial_personality.
    Updated with realistic P2P transaction handling.
    """
    def __init__(self, economic_class='Middle', financial_personality='Rational_Investor'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Salaried Professional"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "40000-80000"
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
            "income_type": "Salary",
            "avg_monthly_income_range": modified_income_range, 
            "income_pattern": "Fixed_Date",
            "savings_retention_rate": "High" if financial_personality == "Saver" else "Medium",
            "has_investment_activity": len(personality_config['investment_types']) > 0,
            "investment_types": personality_config['investment_types'],
            "has_loan_emi": True if random.random() < class_config['loan_propensity'] else False,
            "loan_emi_payment_status": "ALWAYS_ON_TIME", 
            "has_insurance_payments": True,
            "insurance_types": ["Health", "Life"], 
            "utility_payment_status": "ALWAYS_ON_TIME",
            "mobile_plan_type": "Postpaid", 
            "device_consistency_score": round(random.uniform(0.90, 0.98), 2),
            "ip_consistency_score": round(random.uniform(0.85, 0.95), 2), 
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Mobile_Banking"], 
            "login_pattern": "Structured_Daytime",
            "ecommerce_activity_level": "Medium", 
            "ecommerce_avg_ticket_size": "Medium",
        }
        super().__init__(**profile_attributes)

        self.salary_day = random.randint(1, 5)
        min_sal_mod, max_sal_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.salary_amount = random.uniform(min_sal_mod, max_sal_mod)
        self.emi_percentage = 0.25
        self.investment_percentage = 0.15 * personality_config['invest_chance_mod']
        self.insurance_percentage = 0.05
        self.utility_bill_percentage = 0.05
        self.ecommerce_spend_chance = 0.15 * personality_config['spend_chance_mod']
        self.weekday_spend_chance = 0.50 * personality_config['spend_chance_mod']
        self.weekend_spend_chance = 0.70 * personality_config['spend_chance_mod']
        self.annual_bonus_month = 3
        self.has_received_bonus_this_year = False
        
        # ✅ Enhanced P2P attributes - Salaried professionals have family and social networks
        self.dependents = []  # To be populated by simulation engine
        self.professional_network = []  # Professional colleagues and contacts
        self.social_contacts = []  # Friends and social contacts
        
        self.family_support_chance = 0.80  # High chance of family support
        self.professional_transfer_chance = 0.12  # Professional transfers
        self.social_transfer_chance = 0.15  # Social transfers
        self.bonus_sharing_chance = 0.25  # Sharing bonus with family
        
        # Track bonus sharing to avoid duplicate transfers
        self.has_shared_bonus_this_year = False
        
        self.balance = random.uniform(self.salary_amount * 0.2, self.salary_amount * 0.5)

    def _handle_monthly_credits(self, date, events):
        """Handles salary and annual bonus credits."""
        if date.day == self.salary_day:
            txn = self.log_transaction("CREDIT", "Salary Deposit", self.salary_amount, date, channel="Bank Transfer")
            if txn: events.append(txn)
            if date.month == 1: 
                self.has_received_bonus_this_year = False
                self.has_shared_bonus_this_year = False
                
        if date.month == self.annual_bonus_month and date.day == self.salary_day and not self.has_received_bonus_this_year:
            bonus_amount = self.salary_amount * random.uniform(1.5, 3.0)
            txn = self.log_transaction("CREDIT", "Annual Bonus", bonus_amount, date, channel="Bank Transfer")
            if txn: events.append(txn)
            self.has_received_bonus_this_year = True

    def _handle_family_support_transfers(self, date, events, context):
        """✅ UPDATED: Handles regular family support transfers with realistic channels."""
        if (date.day == self.salary_day and 
            self.dependents and 
            random.random() < self.family_support_chance):
            
            recipient = self.dependents[0]  # Primary dependent
            allowance = getattr(recipient, 'monthly_allowance', self.salary_amount * 0.2)
            
            # ✅ NEW: Select realistic channel based on amount
            if allowance > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(allowance, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_professional_network_transfers(self, date, events, context):
        """✅ NEW: Handles transfers within professional network."""
        if (self.professional_network and 
            random.random() < self.professional_transfer_chance and
            self.balance > 5000):
            
            colleague = random.choice(self.professional_network)
            
            # Professional transfers (shared expenses, office collections, etc.)
            transfer_amount = random.uniform(1000, 4000)
            
            # Adjust based on economic class
            if self.economic_class in ['Upper_Middle', 'High']:
                transfer_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': colleague, 
                'amount': round(transfer_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_social_network_transfers(self, date, events, context):
        """✅ NEW: Handles transfers within social network."""
        if (self.social_contacts and 
            random.random() < self.social_transfer_chance and
            self.balance > 3000):
            
            friend = random.choice(self.social_contacts)
            
            # Social transfers (shared outings, gifts, mutual support)
            social_amount = random.uniform(500, 3000)
            
            # Higher amounts on weekends (more social activities)
            if date.weekday() >= 5:  # Weekend
                social_amount *= random.uniform(1.2, 1.8)
            
            # Adjust based on financial personality
            if self.financial_personality == 'Over_Spender':
                social_amount *= random.uniform(1.3, 1.7)
            elif self.financial_personality == 'Saver':
                social_amount *= random.uniform(0.7, 1.0)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': friend, 
                'amount': round(social_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_bonus_sharing(self, date, events, context):
        """✅ NEW: Handles bonus sharing with family after annual bonus."""
        if (self.has_received_bonus_this_year and 
            not self.has_shared_bonus_this_year and
            date.month == self.annual_bonus_month and
            date.day >= self.salary_day + 2 and  # Few days after bonus
            random.random() < self.bonus_sharing_chance):
            
            # Share bonus with dependents or family
            recipients = []
            if self.dependents:
                recipients.extend(self.dependents[:2])  # Max 2 dependents
            
            if recipients:
                for recipient in recipients:
                    # Bonus sharing is typically generous
                    bonus_share = self.salary_amount * random.uniform(0.3, 0.8)
                    
                    # ✅ NEW: Select appropriate channel for larger bonus shares
                    if bonus_share > 50000:
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

    def _handle_recurring_debits(self, date, events):
        """Handles fixed monthly recurring payments."""
        if self.has_loan_emi and date.day == 10:
            emi_amount = self.salary_amount * self.emi_percentage
            txn = self.log_transaction("DEBIT", "Loan EMI Payment", emi_amount, date, channel="Auto_Debit")
            if txn: events.append(txn)
            
        if self.has_insurance_payments and date.day == 15:
            insurance_total = self.salary_amount * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "Insurance Premium", insurance_total, date, channel="Auto_Debit")
            if txn: events.append(txn)
            
        if self.has_investment_activity and date.day == 20:
            invest_amt = self.salary_amount * self.investment_percentage
            txn = self.log_transaction("DEBIT", "SIP Investment", invest_amt, date, channel="Auto_Debit")
            if txn: events.append(txn)
            
        if date.day == 25:
            bill = self.salary_amount * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", bill, date, channel="Netbanking")
            if txn: events.append(txn)

    def _handle_daily_spending(self, date, events):
        """Handles daily spending patterns."""
        ecommerce_chance = self.ecommerce_spend_chance * (2.5 if self.has_received_bonus_this_year else 1)
        if random.random() < ecommerce_chance:
            ecommerce_amt = random.uniform(1000, 5000)
            txn = self.log_transaction("DEBIT", "E-commerce Purchase", ecommerce_amt, date, channel="Card")
            if txn: events.append(txn)
            
        is_weekend = date.weekday() >= 5
        if is_weekend:
            if random.random() < self.weekend_spend_chance:
                spend = random.uniform(500, 2500)
                txn = self.log_transaction("DEBIT", "Weekend Entertainment/Dining", spend, date, channel="Card")
                if txn: events.append(txn)
        else:
            if random.random() < self.weekday_spend_chance:
                spend_type = random.choice(["Transport", "Groceries", "Lunch"])
                spend_amount = random.uniform(150, 800)
                txn = self.log_transaction("DEBIT", f"UPI Spend - {spend_type}", spend_amount, date, channel="UPI")
                if txn: events.append(txn)

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling with realistic channels."""
        events = []
        self._handle_monthly_credits(date, events)
        self._handle_family_support_transfers(date, events, context)    # ✅ Updated with realistic channels
        self._handle_professional_network_transfers(date, events, context)  # ✅ NEW: Professional network transfers
        self._handle_social_network_transfers(date, events, context)    # ✅ NEW: Social network transfers
        self._handle_bonus_sharing(date, events, context)               # ✅ NEW: Bonus sharing
        self._handle_recurring_debits(date, events)
        self._handle_daily_spending(date, events)
        self._handle_daily_living_expenses(date, events)
        return events
