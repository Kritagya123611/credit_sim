# config.py
import numpy as np # Import numpy for clipping

# NEW: Assign a base risk score (0 to 1) for each archetype
ARCHETYPE_BASE_RISK = {
    "Salaried Professional": 0.15,
    "Gig Worker / Freelancer": 0.45,
    "Government Employee": 0.05,
    "Student": 0.65,
    "Daily Wage Laborer": 0.85,
    "Small Business Owner": 0.50,
    "Doctor / Healthcare Worker": 0.10,
    "Tech Professional / IT Employee": 0.12,
    "Police / Security Personnel": 0.08,
    "Retired Senior Citizen": 0.07,
    "Delivery Agent / Rider": 0.55,
    "Lawyer / Consultant": 0.25,
    "Migrant Worker": 0.90,
    "Content Creator / Influencer": 0.60,
    "Homemaker": 0.75,
}

# UPDATE: Add a 'risk_mod' to each class. <1 reduces risk, >1 increases it.
ECONOMIC_CLASSES = {
    "Lower":      {'multiplier': (0.6, 0.8), 'loan_propensity': 0.1, 'risk_mod': 1.20},
    "Lower_Middle": {'multiplier': (0.8, 1.0), 'loan_propensity': 0.2, 'risk_mod': 1.10},
    "Middle":     {'multiplier': (1.0, 1.2), 'loan_propensity': 0.4, 'risk_mod': 1.00},
    "Upper_Middle": {'multiplier': (1.2, 2.0), 'loan_propensity': 0.6, 'risk_mod': 0.85},
    "High":       {'multiplier': (2.0, 4.0), 'loan_propensity': 0.3, 'risk_mod': 0.70}
}

# UPDATE: Add a 'risk_mod' to each personality.
FINANCIAL_PERSONALITIES = {
    "Saver": {
        'spend_chance_mod': 0.7, 
        'invest_chance_mod': 1.2,
        'investment_types': ["FD", "SIP", "LIC"],
        'risk_mod': 0.85 
    },
    "Over_Spender": {
        'spend_chance_mod': 1.5,
        'invest_chance_mod': 0.1,
        'investment_types': [],
        'risk_mod': 1.25
    },
    "Rational_Investor": {
        'spend_chance_mod': 0.9,
        'invest_chance_mod': 1.5,
        'investment_types': ["Stocks", "Mutual_Funds", "SIP"],
        'risk_mod': 0.95
    },
    "Risk_Addict": {
        'spend_chance_mod': 1.2,
        'invest_chance_mod': 1.8,
        'investment_types': ["Crypto", "Stocks"],
        'risk_mod': 1.30
    }
}

# NEW: A helper function to convert the final score back to a category
def get_risk_profile_from_score(score):
    if score < 0.10: return "Very_Low"
    if score < 0.25: return "Low"
    if score < 0.60: return "Medium"
    if score < 0.80: return "High"
    return "Very_High"