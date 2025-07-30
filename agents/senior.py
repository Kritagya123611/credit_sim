import random
from datetime import datetime
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure  # ✅ NEW: Import for realistic P2P handling

class SeniorCitizen(BaseAgent):
    """
    A multi-dimensional profile for a Retired Senior Citizen.
    Behavior is modified by economic_class and financial_personality.
    Updated with realistic P2P transaction handling.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Saver'):
        
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Retired Senior Citizen"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "10000-30000"
        min_income, max_income = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_income * income_multiplier)}-{int(max_income * income_multiplier)}"

        profile_attributes = {
            "archetype_name": archetype_name, 
            "risk_profile": risk_profile_category, 
            "risk_score": risk_score,
            "economic_class": economic_class, 
            "financial_personality": financial_personality,
            "employment_status": "Not_Applicable", 
            "employment_verification": "Pensioner_ID_Verified",
            "income_type": "Pension, Rent", 
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Fixed_Date", 
            "savings_retention_rate": "High",
            "has_investment_activity": True, 
            "investment_types": ["FD", "Senior_Citizen_Schemes"],
            "has_loan_emi": False, 
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": True, 
            "insurance_types": ["Health"],
            "utility_payment_status": "ALWAYS_ON_TIME", 
            "mobile_plan_type": "Basic_Postpaid",
            "device_consistency_score": 0.99, 
            "ip_consistency_score": 0.99, 
            "sim_churn_rate": "Low",
            "primary_digital_channels": ["Netbanking", "Branch"], 
            "login_pattern": "Infrequent",
            "ecommerce_activity_level": "None", 
            "ecommerce_avg_ticket_size": "N/A",
        }
        
        super().__init__(**profile_attributes)

        self.pension_day = 1
        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.monthly_income = random.uniform(min_mod, max_mod)
        
        self.insurance_percentage = 0.15
        self.utility_bill_percentage = 0.08
        
        self.weekly_grocery_day = 4
        self.monthly_pharmacy_day = 10

        self.large_event_month = random.randint(1, 12)
        self.has_done_large_event_this_year = False
        
        # ✅ Enhanced P2P attributes - Senior citizens have family networks
        self.family_members = []  # To be populated by simulation engine
        self.grandchildren = []  # Grandchildren for special gifts
        self.children_network = []  # Adult children for regular support
        
        self.p2p_transfer_chance = 0.08 * personality_config.get('spend_chance_mod', 1.0)
        self.grandchildren_gift_chance = 0.12  # Higher chance for grandchildren gifts
        self.emergency_family_support_chance = 0.05  # Emergency family support
        
        # Special occasions when seniors are more likely to send money
        self.festival_months = [3, 10, 11]  # Holi, Diwali, etc.
        self.birthday_months = random.sample(range(1, 13), k=random.randint(2, 4))  # Family birthdays

        self.balance = random.uniform(self.monthly_income * 2.0, self.monthly_income * 5.0)

    def _handle_monthly_events(self, date, events):
        """Handles fixed monthly income and debits."""
        if date.day == self.pension_day:
            txn = self.log_transaction("CREDIT", "Pension/Rent Deposit", self.monthly_income, date, channel="Bank Transfer")
            if txn: events.append(txn)
            if date.month == 1: self.has_done_large_event_this_year = False

        if self.has_insurance_payments and date.day == 5:
            insurance_amt = self.monthly_income * self.insurance_percentage
            txn = self.log_transaction("DEBIT", "Health Insurance Premium", insurance_amt, date, channel="Auto_Debit")
            if txn: events.append(txn)

        if date.day == self.monthly_pharmacy_day:
            pharma_spend = self.monthly_income * 0.05
            txn = self.log_transaction("DEBIT", "Pharmacy/Medicines", pharma_spend, date, channel="Card")
            if txn: events.append(txn)

        if date.day == 20:
            bill = self.monthly_income * self.utility_bill_percentage
            txn = self.log_transaction("DEBIT", "Utility Bill Payment", bill, date, channel="Netbanking")
            if txn: events.append(txn)

    def _handle_weekly_events(self, date, events):
        """Handles structured weekly spending, like for groceries."""
        if date.weekday() == self.weekly_grocery_day:
            grocery_spend = self.monthly_income * 0.08
            txn = self.log_transaction("DEBIT", "Weekly Groceries/Essentials", grocery_spend, date, channel="Card")
            if txn: events.append(txn)

    def _handle_annual_events(self, date, events):
        """Simulates a major, infrequent financial planning event."""
        if self.has_investment_activity and date.month == self.large_event_month and date.day == 25 and not self.has_done_large_event_this_year:
            fd_amount = self.balance * random.uniform(0.3, 0.5)
            if fd_amount > (10000 * (1 if self.economic_class in ['Lower', 'Lower_Middle'] else 2)):
                txn = self.log_transaction("DEBIT", "New Fixed Deposit Creation", fd_amount, date, channel="Netbanking")
                if txn: events.append(txn)
                self.has_done_large_event_this_year = True

    def _handle_regular_family_transfers(self, date, events, context):
        """✅ UPDATED: Handles regular family support transfers with realistic channels."""
        if (self.family_members and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 5000):
            
            recipient = random.choice(self.family_members)
            
            # Senior citizens send moderate to higher amounts to family
            base_amount = random.uniform(1000, 5000)
            
            # Adjust amount based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                base_amount *= random.uniform(1.2, 2.0)
            elif self.economic_class in ['Lower', 'Lower_Middle']:
                base_amount *= random.uniform(0.5, 0.8)
            
            # ✅ NEW: Select realistic channel based on amount
            if base_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])  # Seniors might use traditional channels for large amounts
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(base_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_grandchildren_gifts(self, date, events, context):
        """✅ NEW: Handles special gifts to grandchildren."""
        if (self.grandchildren and 
            random.random() < self.grandchildren_gift_chance and
            self.balance > 3000):
            
            grandchild = random.choice(self.grandchildren)
            
            # Grandchildren gifts are typically generous but smaller than family support
            gift_amount = random.uniform(500, 2500)
            
            # Higher amounts during birthdays and festivals
            if date.month in self.birthday_months or date.month in self.festival_months:
                gift_amount *= random.uniform(1.5, 2.5)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                gift_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': grandchild, 
                'amount': round(gift_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_festival_and_special_occasions(self, date, events, context):
        """✅ UPDATED: Handles festival and special occasion transfers with realistic channels."""
        # Increased P2P activity during festival months
        if (date.month in self.festival_months and 
            date.day <= 5 and
            self.family_members and 
            random.random() < (self.p2p_transfer_chance * 3) and
            self.balance > 8000):
            
            recipient = random.choice(self.family_members)
            festival_amount = self.monthly_income * random.uniform(0.15, 0.30)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                festival_amount *= random.uniform(1.4, 2.2)
            
            # ✅ NEW: Select realistic channel based on amount
            if festival_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(festival_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_emergency_family_support(self, date, events, context):
        """✅ NEW: Handles emergency family support transfers."""
        if (self.children_network and 
            random.random() < self.emergency_family_support_chance and
            self.balance > 15000):  # Need significant balance for emergency support
            
            child = random.choice(self.children_network)
            
            # Emergency support from seniors can be substantial
            emergency_amount = self.balance * random.uniform(0.2, 0.4)
            
            # Adjust based on economic class
            if self.economic_class in ['High', 'Upper_Middle']:
                emergency_amount *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Select appropriate channel for emergency amounts
            if emergency_amount > 100000:
                channel = random.choice(['NEFT', 'RTGS'])  # Very large emergency amounts
            elif emergency_amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])
            else:
                channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': child, 
                'amount': round(emergency_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling with realistic channels."""
        events = []
        self._handle_monthly_events(date, events)
        self._handle_weekly_events(date, events)
        self._handle_annual_events(date, events)
        self._handle_regular_family_transfers(date, events, context)      # ✅ Updated with realistic channels
        self._handle_grandchildren_gifts(date, events, context)           # ✅ NEW: Grandchildren gifts
        self._handle_festival_and_special_occasions(date, events, context)  # ✅ Updated with realistic channels
        self._handle_emergency_family_support(date, events, context)      # ✅ NEW: Emergency family support
        self._handle_daily_living_expenses(date, events, daily_spend_chance=0.1)
        return events
