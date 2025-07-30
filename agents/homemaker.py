import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure  # ✅ NEW: Import for realistic P2P handling

class Homemaker(BaseAgent):
    """
    A multi-dimensional profile for a Homemaker.
    Behavior is modified by the household's economic_class and financial_personality.
    Updated with realistic P2P transaction handling.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Homemaker"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

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
            "device_consistency_score": round(random.uniform(0.60, 0.80), 2),
            "ip_consistency_score": 0.98,
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["UPI", "Mobile_Banking"],
            "login_pattern": "Infrequent",
            "ecommerce_activity_level": "Medium",
            "ecommerce_avg_ticket_size": "Medium",
        }
        
        super().__init__(**profile_attributes)

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_allowance = random.uniform(min_mod, max_mod)

        self.loan_emi_amount = self.monthly_allowance * 0.30
        self.insurance_premium = self.monthly_allowance * 0.10
        self.utility_bill_amount = self.monthly_allowance * 0.15
        self.weekly_grocery_day = 5
        self.school_fee_months = [1, 4, 7, 10]
        self.occasional_spend_chance = 0.08 * personality_config['spend_chance_mod']
        self.shared_device_id = None 

        # ✅ Enhanced P2P attributes - Homemakers have social networks and family connections
        self.social_circle = []  # To be populated by simulation engine
        self.extended_family = []  # Extended family members
        self.children_contacts = []  # School/education related contacts
        
        self.p2p_transfer_chance = 0.18 * personality_config.get('spend_chance_mod', 1.0)
        self.social_support_chance = 0.12
        self.family_help_chance = 0.08
        self.children_expense_chance = 0.15
        
        # Special occasions when homemakers are more active in transfers
        self.festival_months = [3, 10, 11]  # Festival seasons
        self.school_activity_months = [6, 12]  # School event seasons

        self.balance = random.uniform(self.monthly_allowance * 0.05, self.monthly_allowance * 0.2)

    def _handle_monthly_income_and_fixed_costs(self, date, events):
        """Handles the monthly allowance and fixed, recurring household payments."""
        if date.day == 1:
            txn = self.log_transaction("CREDIT", "Family Support Transfer", self.monthly_allowance, date, channel="P2P")
            if txn: events.append(txn)

        if self.has_loan_emi and date.day == 10:
            txn = self.log_transaction("DEBIT", "Home/Car Loan EMI (Co-payment)", self.loan_emi_amount, date, channel="Auto_Debit")
            if txn: events.append(txn)
            
        if date.day == 15:
            txn = self.log_transaction("DEBIT", "Utility Bill Payment (Gas/DTH)", self.utility_bill_amount, date, channel="UPI")
            if txn: events.append(txn)

        if self.has_insurance_payments and date.day == 20:
            txn = self.log_transaction("DEBIT", "Child Education Plan Premium", self.insurance_premium, date, channel="Auto_Debit")
            if txn: events.append(txn)

    def _handle_household_spending(self, date, events):
        """Simulates structured and occasional spending for the household."""
        if date.weekday() == self.weekly_grocery_day:
            grocery_amount = self.monthly_allowance * random.uniform(0.1, 0.15)
            txn = self.log_transaction("DEBIT", "Weekly Groceries", grocery_amount, date, channel="UPI")
            if txn: events.append(txn)

        if date.month in self.school_fee_months and date.day == 5:
            fee_amount = self.monthly_allowance * random.uniform(0.5, 1.5)
            txn = self.log_transaction("DEBIT", "School Fees", fee_amount, date, channel="Netbanking")
            if txn: events.append(txn)

        if random.random() < self.occasional_spend_chance:
            amount = self.monthly_allowance * random.uniform(0.05, 0.1)
            category = random.choice(["Kids Clothing", "Home Goods", "Online Pharmacy"])
            txn = self.log_transaction("DEBIT", f"E-commerce - {category}", amount, date, channel="Card")
            if txn: events.append(txn)

    def _handle_social_p2p_transfers(self, date, events, context):
        """✅ UPDATED: Handles social circle and community transfers with realistic channels."""
        if (self.social_circle and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 500):
            
            recipient = random.choice(self.social_circle)
            
            # Homemakers typically send moderate amounts for social activities
            base_amount = random.uniform(300, 2000)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                base_amount *= random.uniform(1.5, 2.5)
            elif self.economic_class in ['Lower', 'Lower_Middle']:
                base_amount *= random.uniform(0.6, 1.0)
            
            # Increase during festival months
            if date.month in self.festival_months:
                base_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(base_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_family_support_transfers(self, date, events, context):
        """✅ UPDATED: Handles extended family support transfers with realistic channels."""
        if (self.extended_family and 
            random.random() < self.family_help_chance and
            self.balance > 1000):
            
            recipient = random.choice(self.extended_family)
            
            # Family support amounts based on household allowance
            support_amount = self.monthly_allowance * random.uniform(0.1, 0.25)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                support_amount *= random.uniform(1.2, 1.8)
            
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

    def _handle_children_related_transfers(self, date, events, context):
        """✅ UPDATED: Handles children and education-related transfers with realistic channels."""
        if (self.children_contacts and 
            random.random() < self.children_expense_chance and
            self.balance > 500):
            
            recipient = random.choice(self.children_contacts)
            
            # Children-related expenses are typically smaller but frequent
            base_amount = random.uniform(200, 1500)
            
            # Higher amounts during school activity months
            if date.month in self.school_activity_months:
                base_amount *= random.uniform(1.5, 2.5)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                base_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(base_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_community_support(self, date, events, context):
        """✅ UPDATED: Handles community and social support activities with realistic channels."""
        if (self.social_circle and 
            random.random() < self.social_support_chance and
            self.balance > 800):
            
            recipient = random.choice(self.social_circle)
            
            # Community support amounts
            support_amount = random.uniform(500, 1500)
            
            # Adjust based on economic class and personality
            if self.economic_class in ['High', 'Upper_Middle']:
                support_amount *= random.uniform(1.4, 2.2)
            
            if self.financial_personality == 'Saver':
                support_amount *= random.uniform(0.7, 1.0)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(support_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling with realistic channels."""
        events = []
        self._handle_monthly_income_and_fixed_costs(date, events)
        self._handle_household_spending(date, events)
        self._handle_social_p2p_transfers(date, events, context)        # ✅ Updated with realistic channels
        self._handle_family_support_transfers(date, events, context)    # ✅ Updated with realistic channels
        self._handle_children_related_transfers(date, events, context)  # ✅ Updated with realistic channels
        self._handle_community_support(date, events, context)          # ✅ Updated with realistic channels
        self._handle_daily_living_expenses(date, events)
        return events
