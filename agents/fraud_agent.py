import random
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
import numpy as np
from config_pkg.p2p_structure import RealisticP2PStructure

# ✅ UPDATED IMPORTS: All legitimate agent classes for mimicking
from agents.salaried import SalariedProfessional
from agents.gig_worker import GigWorker
from agents.gov_employee import GovernmentEmployee
from agents.student import Student
from agents.laborer import DailyWageLaborer
from agents.small_business import SmallBusinessOwner
from agents.doctor import Doctor
from agents.it_employee import TechProfessional
from agents.police import PoliceOfficer
from agents.senior import SeniorCitizen
from agents.delivery_agent import DeliveryAgent
from agents.lawyer import Lawyer
from agents.migrant import MigrantWorker
from agents.content_creator import ContentCreator
from agents.homemaker import Homemaker

class FraudAgent(BaseAgent):
    """
    Enhanced fraud agent for Phase 2: Company salary source tracking
    Mimics legitimate agent profiles while exhibiting fraud patterns,
    forcing detection to rely on graph behavioral patterns rather than profile features.
    
    ✅ NEW FEATURES ADDED:
    - Enhanced behavioral diversity to prevent identical fraud patterns
    - Company relationship tracking for sophisticated fraud schemes
    - Improved device count and consistency variations
    - Seasonal and temporal fraud pattern variations
    """
    
    # ✅ ENHANCED: More sophisticated fraud configurations
    FRAUD_CONFIGS = {
        'ring': {
            'balance_range': (3000, 15000),  # More varied
            'transfer_probability': (0.25, 0.35),  # Range instead of fixed
            'channel_threshold': 50000,
            'daily_activity_chance': (0.1, 0.3),  # Varied activity
        },
        'bust_out': {
            'balance_range': (8000, 25000),  # More varied
            'building_days': (45, 120),  # Wider range
            'normal_activity_chance': (0.15, 0.25),  # Range
            'bust_transactions': (2, 8),  # More variation
            'escalation_phases': ['building', 'ramping', 'busting'],
        },
        'mule': {
            'balance_range': (0, 2000),  # Slightly higher range
            'cash_out_range': (25000, 60000),  # More varied
            'dormant_days': (7, 21),  # Waiting period variation
        }
    }
    
    def __init__(self, fraud_type: str, **fraud_params):
        # ✅ ENHANCED: Validate fraud type
        if fraud_type not in self.FRAUD_CONFIGS:
            raise ValueError(f"Invalid fraud type: {fraud_type}. Must be one of {list(self.FRAUD_CONFIGS.keys())}")
        
        # ✅ ENHANCED: More sophisticated agent mimicking
        mimic_class = fraud_params.get('mimic_agent_class', self._select_realistic_mimic_class())
        mimic_agent = mimic_class()
        profile_attributes = mimic_agent.to_dict()

        # ✅ ENHANCED: More realistic fraud profile variations
        profile_attributes.update({
            'archetype_name': f"Fraudulent Agent ({fraud_type})",
            'risk_profile': "Fraud",
            'risk_score': random.uniform(0.95, 0.99),  # Slight variation
            # ✅ UPDATED: More varied device consistency (prevents identical values)
            'device_consistency_score': round(random.uniform(0.05, 0.45), 3),
            'ip_consistency_score': round(random.uniform(0.05, 0.50), 3),
            'sim_churn_rate': random.choice(["High", "Very_High"]),  # Some variation
            # ✅ NEW: Heterogeneous graph attributes for fraud detection
            'industry_sector': "Fraud_Network",
            'company_size': "Fraudulent_Entity",
        })

        super().__init__(**profile_attributes)
        
        # ✅ ENHANCED: Initialize fraud-specific attributes
        self.fraud_type = fraud_type
        self.is_active = True
        
        # ✅ NEW: Enhanced fraud tracking for heterogeneous graphs
        self.fraud_network_connections = []  # Track other fraud agents
        self.compromised_devices = []  # Track device relationships
        self.suspicious_merchants = []  # Track merchant relationships
        self.shell_companies = []  # Track fake company relationships
        
        # ✅ NEW: Initialize required attributes if not inherited from BaseAgent
        if not hasattr(self, 'frequent_merchants'):
            self.frequent_merchants = []
        if not hasattr(self, 'device_fingerprints'):
            self.device_fingerprints = []
        if not hasattr(self, 'ip_clusters'):
            self.ip_clusters = []
        if not hasattr(self, 'relationship_start_dates'):
            self.relationship_start_dates = {}
        if not hasattr(self, 'employer_companies'):
            self.employer_companies = []
        if not hasattr(self, 'current_employer_id'):
            self.current_employer_id = None
        if not hasattr(self, 'behavioral_change_points'):
            self.behavioral_change_points = []
        
        # ✅ ENHANCED: Type-specific initialization with more variation
        config = self.FRAUD_CONFIGS[fraud_type]
        
        if fraud_type == 'ring':
            self._init_ring_fraud(fraud_params, config)
        elif fraud_type == 'bust_out':
            self._init_bust_out_fraud(fraud_params, config)
        elif fraud_type == 'mule':
            self._init_mule_fraud(fraud_params, config)

    # ✅ FIX: Add all missing methods
    def add_frequent_merchant(self, merchant_id, first_transaction_date=None):
        """Add merchant to frequent merchants list with optional transaction date tracking"""
        if merchant_id not in self.frequent_merchants:
            self.frequent_merchants.append(merchant_id)
        
        # Track relationship start date if available
        if first_transaction_date:
            self.relationship_start_dates[f'merchant_{merchant_id}'] = first_transaction_date

    def add_device_fingerprint(self, device_id):
        """Add device fingerprint for fraud tracking"""
        if device_id not in self.device_fingerprints:
            self.device_fingerprints.append(device_id)

    def add_ip_cluster(self, ip_cluster_id):
        """Add IP cluster for fraud tracking"""
        if ip_cluster_id not in self.ip_clusters:
            self.ip_clusters.append(ip_cluster_id)

    def assign_employer(self, company_id, employment_start_date=None):
        """Assign employer company for salary source tracking"""
        # Set as current employer
        self.current_employer_id = company_id
        
        # Add to employer list if not already present
        if company_id not in self.employer_companies:
            self.employer_companies.append(company_id)
        
        # Track employment start date
        if employment_start_date:
            self.relationship_start_dates[f'employer_{company_id}'] = employment_start_date
        else:
            self.relationship_start_dates[f'employer_{company_id}'] = datetime.now().date()

    def log_salary_transaction(self, amount, date, company_id=None):
        """Log salary transaction from employer company"""
        if not company_id and hasattr(self, 'current_employer_id'):
            company_id = self.current_employer_id
        
        if company_id:
            # Use realistic channel for salary transactions
            channel = RealisticP2PStructure.select_realistic_channel(
                amount=amount, 
                transaction_context='salary'
            )
            
            description = f"Salary Credit from {company_id[:8]}"
            
            txn = self.log_transaction("CREDIT", description, amount, date, channel=channel)
            
            if txn:
                # Add salary-specific metadata
                txn['company_id'] = company_id
                txn['transaction_category'] = 'salary'
                txn['company_type'] = 'employer'
                
            return txn
        
        return None

    def get_employment_tenure_months(self):
        """Calculate employment tenure in months"""
        if not self.current_employer_id:
            return 0
        
        start_date_key = f'employer_{self.current_employer_id}'
        if start_date_key not in self.relationship_start_dates:
            return 0
        
        start_date = self.relationship_start_dates[start_date_key]
        current_date = datetime.now().date()
        
        # Calculate tenure in months
        months = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
        return max(0, months)

    def get_heterogeneous_connections_summary(self):
        """Get summary of heterogeneous graph connections"""
        return {
            'employer_count': len(self.employer_companies),
            'merchant_count': len(self.frequent_merchants),
            'device_count': len(self.device_fingerprints),
            'current_employer': self.current_employer_id,
            'employment_tenure_months': self.get_employment_tenure_months()
        }

    def get_realistic_device_count(self):
        """✅ OVERRIDE: Fraudsters typically have more devices for obfuscation"""
        device_options = [2, 3, 4, 5, 6]
        weights = [0.1, 0.2, 0.3, 0.3, 0.1]  # Most have 3-5 devices
        return random.choices(device_options, weights=weights)[0]

    def _select_realistic_mimic_class(self):
        """✅ NEW: Select realistic agent class to mimic based on fraud patterns"""
        # Fraudsters typically mimic lower-income, less monitored agent types
        mimic_options = [
            (GigWorker, 0.25),
            (Student, 0.20), 
            (DailyWageLaborer, 0.18),
            (DeliveryAgent, 0.12),
            (MigrantWorker, 0.10),
            (SalariedProfessional, 0.08),
            (Homemaker, 0.07)
        ]
        agents, weights = zip(*mimic_options)
        return random.choices(agents, weights=weights)[0]

    def _init_ring_fraud(self, params, config):
        """✅ ENHANCED: Initialize ring fraud with company relationships"""
        self.ring_id = params.get('ring_id', f"ring_{random.randint(1000, 9999)}")
        shared_footprint = params.get('shared_footprint', {})
        
        # ✅ ENHANCED: Device/IP tracking with more variation
        self.device_id = shared_footprint.get('device_id', f"device_{random.randint(10000, 99999)}")
        self.ip_cluster_id = shared_footprint.get('ip_address', f"ip_cluster_{random.randint(100, 999)}")
        
        # Add to heterogeneous connections
        self.add_device_fingerprint(self.device_id)
        self.add_ip_cluster(self.ip_cluster_id)
        
        # ✅ NEW: Create shell companies for sophisticated fraud
        shell_company_id = f"shell_company_{self.ring_id}_{random.randint(100, 999)}"
        self.shell_companies.append(shell_company_id)
        self.assign_employer(shell_company_id, datetime.now().date() - timedelta(days=random.randint(30, 180)))
        
        # ✅ ENHANCED: More varied balance ranges
        self.balance = random.uniform(*config['balance_range'])
        self.daily_activity_chance = random.uniform(*config['daily_activity_chance'])
        self.transfer_probability = random.uniform(*config['transfer_probability'])

    def _init_bust_out_fraud(self, params, config):
        """✅ ENHANCED: Initialize bust-out fraud with company salary simulation"""
        self.behavior_state = 'building_credit'
        self.creation_date = params.get('creation_date', datetime.now().date())
        self.bust_out_day_threshold = random.randint(*config['building_days'])
        self.balance = random.uniform(*config['balance_range'])
        self.normal_activity_chance = random.uniform(*config['normal_activity_chance'])
        
        # ✅ NEW: Create fake employment for credit building
        fake_company_id = f"fake_employer_{random.randint(10000, 99999)}"
        self.shell_companies.append(fake_company_id)
        self.assign_employer(fake_company_id, self.creation_date)
        
        # ✅ ENHANCED: Track behavioral phases
        self.escalation_phases = config['escalation_phases']
        self.current_phase_index = 0
        
        # ✅ NEW: Track behavioral change points for temporal analysis
        self.behavioral_change_points.append({
            'date': self.creation_date,
            'event': 'fraud_account_creation',
            'state': 'building_credit',
            'phase': self.escalation_phases[0]
        })

    def _init_mule_fraud(self, params, config):
        """✅ ENHANCED: Initialize money mule with dormancy period"""
        self.cash_out_threshold = random.uniform(*config['cash_out_range'])
        self.balance = random.uniform(*config['balance_range'])
        self.dormant_days = random.randint(*config['dormant_days'])
        self.activation_date = datetime.now().date() + timedelta(days=self.dormant_days)
        
        # ✅ FIXED: Mules often have suspicious merchant relationships
        for _ in range(random.randint(2, 5)):
            suspicious_merchant_id = f"suspicious_merchant_{random.randint(1000, 9999)}"
            self.suspicious_merchants.append(suspicious_merchant_id)
            # ✅ NOW WORKS: Method exists
            self.add_frequent_merchant(suspicious_merchant_id)

    def _handle_fake_salary_payments(self, date, events):
        """✅ NEW: Generate fake salary payments for credit building"""
        if (self.shell_companies and 
            date.day in [28, 29, 30, 31] and  # End of month
            random.random() < 0.8):  # 80% chance of fake salary
            
            company_id = random.choice(self.shell_companies)
            
            # Fake salary amount based on mimicked agent type
            base_salary = random.uniform(15000, 45000)
            
            # Add some seasonal variation
            month_multiplier = {
                12: 1.2,  # December bonus
                3: 1.1,   # Year-end bonus
                6: 0.9,   # Mid-year low
            }.get(date.month, 1.0)
            
            salary_amount = base_salary * month_multiplier
            
            # ✅ NEW: Log as salary transaction from shell company
            txn = self.log_salary_transaction(
                amount=salary_amount,
                date=date,
                company_id=company_id
            )
            
            if txn:
                txn['transaction_category'] = 'fake_salary'
                txn['company_type'] = 'shell_company'
                events.append(txn)

    def act(self, date: datetime, **context):
        """✅ ENHANCED: More sophisticated fraud behavior with company salary tracking"""
        if not self.is_active:
            return []

        events = []
        
        # ✅ NEW: Handle fake salary payments for all fraud types
        self._handle_fake_salary_payments(date, events)
        
        # ✅ ENHANCED: Delegate to specific fraud type handlers with improvements
        if self.fraud_type == 'ring':
            events.extend(self._execute_ring_fraud(date, context))
        elif self.fraud_type == 'bust_out':
            events.extend(self._execute_bust_out_fraud(date, context))
        elif self.fraud_type == 'mule':
            events.extend(self._execute_mule_fraud(date, context))
        
        return events

    def _execute_ring_fraud(self, date, context):
        """✅ ENHANCED: Ring fraud with improved behavioral patterns"""
        events = []
        ring_members = context.get('ring_members', [])
        config = self.FRAUD_CONFIGS['ring']
        
        # ✅ NEW: Daily legitimate-looking activity
        if random.random() < self.daily_activity_chance:
            # Small legitimate-looking transactions
            amount = random.uniform(50, 500)
            desc = random.choice(["Mobile Recharge", "Grocery", "Transport"])
            txn = self.log_transaction("DEBIT", desc, amount, date, channel="UPI")
            if txn:
                events.append(txn)
        
        # ✅ ENHANCED: Ring transfers with improved logic
        if (len(ring_members) > 1 and 
            random.random() < self.transfer_probability):
            
            try:
                my_index = ring_members.index(self)
                recipient_index = (my_index + 1) % len(ring_members)
                recipient = ring_members[recipient_index]
                
                # ✅ ENHANCED: More sophisticated amount calculation
                base_amount = self.balance * random.uniform(0.08, 0.35)
                
                # Vary amounts based on day of week (higher on weekends)
                day_multiplier = 1.3 if date.weekday() >= 5 else 1.0
                amount = base_amount * day_multiplier
                
                # ✅ ENHANCED: Channel selection with more realism
                if amount > config['channel_threshold']:
                    channel = random.choice(['NEFT', 'RTGS'])
                elif amount > 10000:
                    channel = random.choice(['IMPS', 'NEFT'])
                else:
                    channel = RealisticP2PStructure.select_realistic_channel()
                
                context.get('p2p_transfers', []).append({
                    'sender': self,
                    'recipient': recipient,
                    'amount': amount,
                    'desc': 'Ring Transfer',
                    'channel': channel,
                    'transaction_category': 'fraud_ring_transfer',
                    'ring_id': self.ring_id
                })
                
            except (ValueError, IndexError):
                # Handle cases where agent not in ring or ring structure issues
                pass
        
        return events

    def _execute_bust_out_fraud(self, date, context):
        """✅ ENHANCED: Bust-out fraud with sophisticated phases"""
        events = []
        config = self.FRAUD_CONFIGS['bust_out']
        days_active = (date.date() - self.creation_date).days
        
        # ✅ ENHANCED: Multi-phase bust-out behavior
        if self.behavior_state == 'building_credit':
            if days_active < self.bust_out_day_threshold * 0.6:
                # Phase 1: Building credit with small transactions
                if random.random() < self.normal_activity_chance:
                    desc = random.choice(["Grocery Shopping", "Mobile Recharge", "Bill Payment"])
                    amount = random.uniform(100, 800)
                    txn = self.log_transaction("DEBIT", desc, amount, date, channel="UPI")
                    if txn:
                        events.append(txn)
                        
            elif days_active < self.bust_out_day_threshold:
                # Phase 2: Ramping up activity
                self.behavior_state = 'ramping'
                if random.random() < (self.normal_activity_chance * 1.5):
                    desc = random.choice(["Online Shopping", "Restaurant", "Entertainment"])
                    amount = random.uniform(500, 2000)
                    txn = self.log_transaction("DEBIT", desc, amount, date, channel="Card")
                    if txn:
                        events.append(txn)
            else:
                # Phase 3: Transition to busting
                self.behavior_state = 'busting_out'
                self.behavioral_change_points.append({
                    'date': date.date(),
                    'event': 'bust_out_initiation',
                    'state': 'busting_out',
                    'phase': 'active_bust'
                })

        if self.behavior_state == 'busting_out':
            # ✅ ENHANCED: Rapid depletion with varied patterns
            burst_count = random.randint(*config['bust_transactions'])
            for i in range(burst_count):
                if self.balance <= 100:  # Keep small buffer
                    break
                    
                # ✅ NEW: Progressive amount increases
                progress_multiplier = 1 + (i * 0.1)  # Escalating amounts
                base_amount = self.balance * random.uniform(0.15, 0.35)
                amount = min(base_amount * progress_multiplier, self.balance * 0.8)
                
                desc = random.choice([
                    "Large E-commerce Purchase", 
                    "ATM Withdrawal", 
                    "Cryptocurrency Purchase",
                    "Investment Transfer"
                ])
                
                channel = "Card" if "E-commerce" in desc else ("ATM" if "ATM" in desc else "Netbanking")
                
                # ✅ NEW: Track suspicious merchants for e-commerce purchases
                if "E-commerce" in desc:
                    suspicious_merchant_id = f"suspicious_ecommerce_{random.randint(1000, 9999)}"
                    self.suspicious_merchants.append(suspicious_merchant_id)
                    self.add_frequent_merchant(suspicious_merchant_id, date)
                
                txn = self.log_transaction("DEBIT", desc, amount, date, channel=channel)
                if txn:
                    events.append(txn)
            
            self.is_active = False
        
        return events

    def _execute_mule_fraud(self, date, context):
        """✅ ENHANCED: Money mule with dormancy and activation patterns"""
        events = []
        
        # ✅ NEW: Dormancy period - mules wait before activation
        if date.date() < self.activation_date:
            # Small legitimate transactions during dormancy
            if random.random() < 0.1:  # 10% chance of small activity
                amount = random.uniform(50, 200)
                desc = random.choice(["Mobile Recharge", "Small Purchase"])
                txn = self.log_transaction("DEBIT", desc, amount, date, channel="UPI")
                if txn:
                    events.append(txn)
            return events
        
        # ✅ ENHANCED: Cash out behavior with pattern variation
        if self.balance > self.cash_out_threshold:
            # ✅ NEW: Multiple cash-out transactions to avoid detection
            remaining_balance = self.balance
            cash_out_sessions = random.randint(1, 3)
            
            for session in range(cash_out_sessions):
                if remaining_balance <= 1000:
                    break
                    
                # ✅ NEW: Vary cash-out amounts and timing
                session_amount = remaining_balance * random.uniform(0.3, 0.8)
                session_amount = min(session_amount, 20000)  # ATM limits
                
                txn = self.log_transaction(
                    "DEBIT", "ATM Cash Withdrawal", session_amount, date, channel="ATM"
                )
                if txn:
                    events.append(txn)
                    remaining_balance -= session_amount
                
                # ✅ NEW: Track cash-out events
                self.behavioral_change_points.append({
                    'date': date.date(),
                    'event': 'mule_cash_out',
                    'amount': session_amount,
                    'session': session + 1
                })
                
            self.is_active = False
        
        return events

    def get_fraud_specific_features(self):
        """✅ ENHANCED: Comprehensive fraud-specific features"""
        base_features = self.get_heterogeneous_connections_summary()
        
        fraud_features = {
            'fraud_type': self.fraud_type,
            'fraud_network_size': len(self.fraud_network_connections),
            'compromised_device_count': len(self.compromised_devices),
            'suspicious_merchant_count': len(self.suspicious_merchants),
            'shell_company_count': len(self.shell_companies),
            'behavioral_change_count': len(self.behavioral_change_points),
            'is_active_fraud': self.is_active,
            'employment_tenure_months': self.get_employment_tenure_months(),
            'device_count': len(self.device_fingerprints),
            'total_company_relationships': len(self.shell_companies)
        }
        
        # ✅ ENHANCED: Type-specific features
        if self.fraud_type == 'ring':
            fraud_features.update({
                'ring_id': self.ring_id,
                'daily_activity_chance': self.daily_activity_chance,
                'transfer_probability': self.transfer_probability
            })
        elif self.fraud_type == 'bust_out':
            fraud_features.update({
                'behavior_state': self.behavior_state,
                'days_since_creation': (datetime.now().date() - self.creation_date).days,
                'bust_out_threshold': self.bust_out_day_threshold,
                'current_phase': self.escalation_phases[min(self.current_phase_index, len(self.escalation_phases)-1)]
            })
        elif self.fraud_type == 'mule':
            fraud_features.update({
                'cash_out_threshold': self.cash_out_threshold,
                'dormant_days': self.dormant_days,
                'days_until_activation': max(0, (self.activation_date - datetime.now().date()).days)
            })
        
        return {**base_features, **fraud_features}

    def __repr__(self):
        return f"<FraudAgent {self.agent_id[:6]} - {self.fraud_type}, Active: {self.is_active}, ₹{self.balance:.2f}>"
