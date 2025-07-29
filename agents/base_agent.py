import uuid
from faker import Faker
from datetime import datetime

# Initialize Faker once to be used by all agents
fake = Faker('en_IN')

class BaseAgent:
    """
    The parent agent class containing standardized fields,
    transaction logging, and common behavior across all agent archetypes.
    """

    STANDARDIZED_FIELDS = {
        # Core Profile
        "archetype_name": "Generic",             # str
        "risk_profile": "N/A",                   # str
        "risk_score": 0.0,                       # float
        "economic_class": "N/A",                 # str             
        "financial_personality": "N/A",          # str
        "employment_status": "N/A",              # str
        "employment_verification": "N/A",        # str
        "income_type": "N/A",                    # str
        "avg_monthly_income_range": "N/A",       # str
        "income_pattern": "N/A",                 # str

        # Financial Habits
        "savings_retention_rate": "N/A",         # str
        "has_investment_activity": False,        # bool
        "investment_types": [],                  # list[str]
        "has_loan_emi": False,                   # bool
        "loan_emi_payment_status": "N/A",        # str
        "has_insurance_payments": False,         # bool
        "insurance_types": [],                   # list[str]

        # Utility Payments
        "utility_payment_status": "N/A",         # str
        "mobile_plan_type": "N/A",               # str

        # Digital Footprint
        "device_consistency_score": 0.0,         # float
        "ip_consistency_score": 0.0,             # float
        "sim_churn_rate": "N/A",                 # str
        "primary_digital_channels": [],          # list[str]
        "login_pattern": "N/A",                  # str

        # E-commerce
        "ecommerce_activity_level": "N/A",       # str
        "ecommerce_avg_ticket_size": "N/A",      # str
    }

    def __init__(self, **kwargs):
        # Universal Identity Attributes
        self.agent_id = str(uuid.uuid4())
        self.name = fake.name()
        self.email = fake.email()
        self.account_no = fake.bban()
        self.balance = kwargs.get("starting_balance", 0.0)
        self.txn_log = []

        # Standardized Fields from schema
        for field, default in self.STANDARDIZED_FIELDS.items():
            setattr(self, field, kwargs.get(field, default))

    def __repr__(self):
        return f"<Agent {self.agent_id[:6]} - {self.archetype_name}, ₹{self.balance:.2f}>"

    def to_dict(self):
        """Converts the agent's profile attributes to a dictionary for CSV/JSON export."""
        profile_dict = self.__dict__.copy()
        profile_dict.pop('balance', None)
        profile_dict.pop('txn_log', None)
        return profile_dict

    def get_transaction_log(self):
        """Returns the list of all transactions performed by the agent."""
        return self.txn_log

    def log_transaction(self, txn_type, description, amount, date):
        """Logs a validated transaction and updates agent's balance."""
        if txn_type == "CREDIT":
            self.balance += amount
        elif txn_type == "DEBIT":
            if self.balance < amount:
                return None  # Reject transaction
            self.balance -= amount

        txn = {
            "agent_id": self.agent_id,
            "date": date.strftime("%Y-%m-%d"),
            "type": txn_type,
            "description": description,
            "amount": round(amount, 2),
            "balance_after_txn": round(self.balance, 2)
        }
        self.txn_log.append(txn)
        return txn

    def act(self, date: datetime):
        """Override this method in child classes to define agent behavior."""
        return []
