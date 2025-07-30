import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure  # ✅ NEW: Import for realistic P2P handling

class Student(BaseAgent):
    """
    A multi-dimensional profile for a Student.
    Behavior is modified by economic_class and financial_personality.
    Updated with realistic P2P transaction handling.
    """
    def __init__(self, economic_class='Lower_Middle', financial_personality='Over_Spender'):
        class_config = ECONOMIC_CLASSES[economic_class]
        personality_config = FINANCIAL_PERSONALITIES[financial_personality]
        income_multiplier = random.uniform(*class_config['multiplier'])
        archetype_name = "Student"

        base_risk = ARCHETYPE_BASE_RISK[archetype_name]
        class_mod = class_config['risk_mod']
        pers_mod = personality_config['risk_mod']
        final_score = base_risk * class_mod * pers_mod
        risk_score = round(np.clip(final_score, 0.01, 0.99), 4)
        risk_profile_category = get_risk_profile_from_score(risk_score)

        base_income_range = "3000-10000"
        min_allowance, max_allowance = map(int, base_income_range.split('-'))
        modified_income_range = f"{int(min_allowance * income_multiplier)}-{int(max_allowance * income_multiplier)}"
        
        profile_attributes = {
            "archetype_name": archetype_name, 
            "risk_profile": risk_profile_category, 
            "risk_score": risk_score,
            "economic_class": economic_class, 
            "financial_personality": financial_personality,
            "employment_status": "Not_Applicable", 
            "employment_verification": "Not_Applicable",
            "income_type": "Allowance", 
            "avg_monthly_income_range": modified_income_range,
            "income_pattern": "Irregular", 
            "savings_retention_rate": "Very_Low",
            "has_investment_activity": False, 
            "investment_types": [],
            "has_loan_emi": False, 
            "loan_emi_payment_status": "N/A",
            "has_insurance_payments": False, 
            "insurance_types": [],
            "utility_payment_status": "N/A", 
            "mobile_plan_type": "Prepaid",
            "device_consistency_score": round(random.uniform(0.70, 0.85), 2),
            "ip_consistency_score": round(random.uniform(0.40, 0.60), 2),
            "sim_churn_rate": "Medium", 
            "primary_digital_channels": ["UPI", "Wallets"],
            "login_pattern": "Late_Night_Activity", 
            "ecommerce_activity_level": "High",
            "ecommerce_avg_ticket_size": "Low",
        }
        
        super().__init__(**profile_attributes)

        min_mod, max_mod = map(int, self.avg_monthly_income_range.split('-'))
        self.allowance_amount = random.uniform(min_mod, max_mod)
        self.allowance_days = sorted(random.sample(range(2, 28), random.randint(1, 2)))

        self.daily_spend_chance = 0.75 * personality_config['spend_chance_mod']
        self.recharge_chance = 0.08
        self.bnpl_chance = 0.15 if financial_personality == 'Over_Spender' else 0.05
        
        # ✅ Enhanced P2P attributes - Students have active social networks
        self.contacts = []  # To be populated by simulation engine
        self.study_group = []  # Study group members for academic expenses
        self.hostel_friends = []  # Hostel/roommate network for shared expenses
        
        self.p2p_transfer_chance = 0.25 * personality_config['spend_chance_mod']
        self.study_group_transfer_chance = 0.15  # Academic-related transfers
        self.hostel_sharing_chance = 0.18  # Shared hostel expenses
        self.emergency_help_chance = 0.08  # Emergency peer support
        
        # Special events when students are more active in transfers
        self.exam_months = [4, 10, 11]  # Exam seasons
        self.festival_months = [3, 10, 11]  # Festival seasons

        self.bnpl_repayments = {}
        self.balance = random.uniform(100, 500)

    def _handle_income(self, date, events):
        """Handles irregular allowance/family support income."""
        if date.day in self.allowance_days:
            txn = self.log_transaction("CREDIT", "Allowance/Family Support", self.allowance_amount, date, channel="P2P")
            if txn: events.append(txn)

    def _handle_spending(self, date, events):
        """Handles regular student spending patterns."""
        # Handle BNPL repayments
        if date.date() in self.bnpl_repayments:
            amount_due = self.bnpl_repayments.pop(date.date())
            txn = self.log_transaction("DEBIT", "BNPL Repayment", amount_due, date, channel="UPI")
            if txn: events.append(txn)

        # Handle mobile recharge
        if random.random() < self.recharge_chance:
            recharge_amount = random.choice([99, 149, 199, 239])
            txn = self.log_transaction("DEBIT", "Prepaid Mobile Recharge", recharge_amount, date, channel="UPI")
            if txn: events.append(txn)

        # Handle regular spending
        if random.random() < self.daily_spend_chance:
            spend_category = random.choice([
                "Food_Delivery", "Cab_Service", "OTT_Subscription", "Groceries", "Gaming_Purchase", "Stationery"
            ])
            
            if spend_category == "Food_Delivery" and random.random() < self.bnpl_chance:
                # BNPL transaction - no immediate debit
                spend_amount = random.uniform(100, 600)
                repayment_date = date.date() + timedelta(days=15)
                self.bnpl_repayments[repayment_date] = self.bnpl_repayments.get(repayment_date, 0) + spend_amount
                # Log as a BNPL transaction (optional - could track differently)
                txn = self.log_transaction("DEBIT", f"BNPL - {spend_category}", 0, date, channel="BNPL")
                if txn: events.append(txn)
            else:
                spend_amount = random.uniform(100, 600)
                txn = self.log_transaction("DEBIT", f"UPI Spend - {spend_category}", spend_amount, date, channel="UPI")
                if txn: events.append(txn)

    def _handle_peer_group_transfers(self, date, events, context):
        """✅ UPDATED: Handles general peer group transfers with realistic channels."""
        if (self.contacts and 
            random.random() < self.p2p_transfer_chance and
            self.balance > 200):
            
            recipient = random.choice(self.contacts)
            # Students typically send smaller amounts
            amount = random.uniform(100, 800)
            
            # Adjust based on economic class
            if self.economic_class in ['Upper_Middle', 'High']:
                amount *= random.uniform(1.2, 1.8)
            elif self.economic_class in ['Lower']:
                amount *= random.uniform(0.6, 0.9)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': recipient, 
                'amount': round(amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_study_group_transfers(self, date, events, context):
        """✅ NEW: Handles academic-related transfers to study group."""
        if (self.study_group and 
            random.random() < self.study_group_transfer_chance and
            self.balance > 150):
            
            study_mate = random.choice(self.study_group)
            
            # Academic expenses are typically small but frequent
            academic_amount = random.uniform(50, 400)
            
            # Higher amounts during exam months
            if date.month in self.exam_months:
                academic_amount *= random.uniform(1.3, 2.0)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': study_mate, 
                'amount': round(academic_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_hostel_sharing_transfers(self, date, events, context):
        """✅ NEW: Handles shared hostel/accommodation expenses."""
        if (self.hostel_friends and 
            random.random() < self.hostel_sharing_chance and
            self.balance > 200):
            
            hostel_mate = random.choice(self.hostel_friends)
            
            # Shared expenses (food, utilities, cleaning, etc.)
            shared_amount = random.uniform(150, 800)
            
            # Adjust based on economic class
            if self.economic_class in ['Upper_Middle', 'High']:
                shared_amount *= random.uniform(1.2, 1.6)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': hostel_mate, 
                'amount': round(shared_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def _handle_emergency_peer_support(self, date, events, context):
        """✅ NEW: Handles emergency support within student network."""
        if (self.contacts and 
            random.random() < self.emergency_help_chance and
            self.balance > 500):  # Need decent balance for emergency help
            
            friend = random.choice(self.contacts)
            
            # Emergency amounts for students
            emergency_amount = random.uniform(300, 1000)
            
            # Adjust based on economic class and personality
            if self.economic_class in ['Upper_Middle', 'High']:
                emergency_amount *= random.uniform(1.3, 2.0)
            
            if self.financial_personality == 'Over_Spender':
                emergency_amount *= random.uniform(1.2, 1.5)  # More generous
            
            # Can't give more than 40% of balance in emergency
            max_emergency = self.balance * 0.4
            final_amount = min(emergency_amount, max_emergency)
            
            if final_amount >= 100:  # Minimum for meaningful emergency help
                # ✅ NEW: Select realistic channel
                channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self, 
                    'recipient': friend, 
                    'amount': round(final_amount, 2), 
                    'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                    'channel': channel  # ✅ Realistic channel
                })

    def _handle_festival_transfers(self, date, events, context):
        """✅ NEW: Handles festival-related transfers."""
        if (date.month in self.festival_months and 
            date.day <= 5 and
            self.contacts and 
            random.random() < (self.p2p_transfer_chance * 2) and
            self.balance > 300):
            
            friend = random.choice(self.contacts)
            
            # Festival transfers are typically smaller gifts
            festival_amount = random.uniform(200, 800)
            
            # Adjust based on economic class
            if self.economic_class in ['Upper_Middle', 'High']:
                festival_amount *= random.uniform(1.2, 1.8)
            
            # ✅ NEW: Select realistic channel
            channel = RealisticP2PStructure.select_realistic_channel()
            
            context.get('p2p_transfers', []).append({
                'sender': self, 
                'recipient': friend, 
                'amount': round(festival_amount, 2), 
                'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                'channel': channel  # ✅ Realistic channel
            })

    def act(self, date: datetime, **context):
        """✅ Updated: Now includes comprehensive P2P transfer handling with realistic channels."""
        events = []
        self._handle_income(date, events)
        self._handle_spending(date, events)
        self._handle_peer_group_transfers(date, events, context)      # ✅ Updated with realistic channels
        self._handle_study_group_transfers(date, events, context)     # ✅ NEW: Academic transfers
        self._handle_hostel_sharing_transfers(date, events, context)  # ✅ NEW: Hostel sharing
        self._handle_emergency_peer_support(date, events, context)    # ✅ NEW: Emergency support
        self._handle_festival_transfers(date, events, context)        # ✅ NEW: Festival transfers
        self._handle_daily_living_expenses(date, events)
        return events
