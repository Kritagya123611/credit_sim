# simulation_engine.py

import os
import random
import pandas as pd
import numpy as np
import uuid
from datetime import date, timedelta, datetime
from faker import Faker

# Step 1: Import configurations and all agent classes
from config import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES
from agents import (
    SalariedProfessional, GigWorker, GovernmentEmployee, Student, DailyWageLaborer,
    SmallBusinessOwner, Doctor, TechProfessional, PoliceOfficer, SeniorCitizen,
    DeliveryAgent, Lawyer, MigrantWorker, ContentCreator, Homemaker, FraudAgent
)

print("All agent classes and configurations imported successfully.")
fake = Faker('en_IN')

# Step 2: Define Simulation Parameters
TOTAL_POPULATION = 1000
SIMULATION_START_DATE = date(2024, 1, 1)
SIMULATION_END_DATE = date(2024, 6, 30) #<-- Set to 6 months per your request

# Step 3: Define Population & Behavior Distributions
POPULATION_MIX = {
    SalariedProfessional: 0.12, TechProfessional: 0.04, SmallBusinessOwner: 0.18,
    GovernmentEmployee: 0.06, PoliceOfficer: 0.03, Doctor: 0.01, Lawyer: 0.01,
    GigWorker: 0.04, DeliveryAgent: 0.03, ContentCreator: 0.01, Student: 0.08,
    Homemaker: 0.11, DailyWageLaborer: 0.12, MigrantWorker: 0.04, SeniorCitizen: 0.04,
}
FRAUD_POPULATION_PERCENTAGE = 0.04
FRAUD_MIX = {'ring': 0.50, 'bust_out': 0.25, 'mule': 0.25}

# --- Archetype-Specific Behavioral Distributions ---
BEHAVIORAL_DISTRIBUTIONS = {
    SalariedProfessional: {"class": [0.05, 0.40, 0.40, 0.15, 0.00], "personality": [0.40, 0.20, 0.40, 0.00]},
    GigWorker:            {"class": [0.40, 0.50, 0.10, 0.00, 0.00], "personality": [0.30, 0.50, 0.10, 0.10]},
    GovernmentEmployee:   {"class": [0.00, 0.45, 0.50, 0.05, 0.00], "personality": [0.70, 0.15, 0.15, 0.00]},
    Student:              {"class": [0.30, 0.50, 0.20, 0.00, 0.00], "personality": [0.20, 0.60, 0.05, 0.15]},
    DailyWageLaborer:     {"class": [0.95, 0.05, 0.00, 0.00, 0.00], "personality": [0.80, 0.20, 0.00, 0.00]},
    SmallBusinessOwner:   {"class": [0.20, 0.40, 0.30, 0.08, 0.02], "personality": [0.30, 0.20, 0.40, 0.10]},
    Doctor:               {"class": [0.00, 0.05, 0.35, 0.40, 0.20], "personality": [0.20, 0.10, 0.65, 0.05]},
    TechProfessional:     {"class": [0.00, 0.10, 0.40, 0.40, 0.10], "personality": [0.20, 0.20, 0.40, 0.20]},
    PoliceOfficer:        {"class": [0.10, 0.60, 0.30, 0.00, 0.00], "personality": [0.60, 0.30, 0.10, 0.00]},
    SeniorCitizen:        {"class": [0.20, 0.50, 0.25, 0.05, 0.00], "personality": [0.80, 0.15, 0.05, 0.00]},
    DeliveryAgent:        {"class": [0.50, 0.45, 0.05, 0.00, 0.00], "personality": [0.40, 0.50, 0.05, 0.05]},
    Lawyer:               {"class": [0.00, 0.15, 0.40, 0.35, 0.10], "personality": [0.20, 0.15, 0.60, 0.05]},
    MigrantWorker:        {"class": [0.90, 0.10, 0.00, 0.00, 0.00], "personality": [0.90, 0.10, 0.00, 0.00]},
    ContentCreator:       {"class": [0.30, 0.40, 0.20, 0.08, 0.02], "personality": [0.10, 0.40, 0.20, 0.30]},
    Homemaker:            {"class": [0.20, 0.40, 0.30, 0.10, 0.00], "personality": [0.50, 0.40, 0.10, 0.00]},
}

# Step 4: Create the Agent Population
print(f"Creating a population of {TOTAL_POPULATION} agents...")
agent_population = []
legit_population = []
num_legitimate = int(TOTAL_POPULATION * (1 - FRAUD_POPULATION_PERCENTAGE))
num_fraud = TOTAL_POPULATION - num_legitimate

# --- Create Legitimate Population with new dimensions ---
legit_archetypes = list(POPULATION_MIX.keys())
legit_probabilities = list(POPULATION_MIX.values())
legit_probabilities /= np.sum(legit_probabilities)

for _ in range(num_legitimate):
    ChosenAgentClass = np.random.choice(legit_archetypes, p=legit_probabilities)
    
    # --- UPDATED LOGIC ---
    # Look up the specific distributions for the chosen archetype
    class_dist = BEHAVIORAL_DISTRIBUTIONS[ChosenAgentClass]["class"]
    pers_dist = BEHAVIORAL_DISTRIBUTIONS[ChosenAgentClass]["personality"]
    
    # Choose dimensions based on the archetype-specific distributions
    chosen_class = np.random.choice(list(ECONOMIC_CLASSES.keys()), p=class_dist)
    chosen_personality = np.random.choice(list(FINANCIAL_PERSONALITIES.keys()), p=pers_dist)
    
    agent = ChosenAgentClass(
        economic_class=chosen_class,
        financial_personality=chosen_personality
    )
    legit_population.append(agent)
    agent_population.append(agent)

# --- Create Fraud Population ---
fraud_rings = {}
mule_agents = []
ring_fraudsters = []
for _ in range(num_fraud):
    fraud_type = np.random.choice(list(FRAUD_MIX.keys()), p=list(FRAUD_MIX.values()))
    
    if fraud_type == 'ring':
        num_rings = max(1, int(num_fraud * FRAUD_MIX['ring'] / 4))
        ring_id = f"ring_{random.randint(1, num_rings)}"
        
        if ring_id not in fraud_rings:
            shared_footprint = {'device_id': str(uuid.uuid4()), 'ip_address': fake.ipv4()}
            fraud_rings[ring_id] = {'members': [], 'footprint': shared_footprint, 'mules': []}
        
        agent = FraudAgent(fraud_type='ring', ring_id=ring_id, shared_footprint=fraud_rings[ring_id]['footprint'])
        fraud_rings[ring_id]['members'].append(agent)
        ring_fraudsters.append(agent)
        agent_population.append(agent)

    elif fraud_type == 'bust_out':
        agent_population.append(FraudAgent(fraud_type='bust_out', creation_date=SIMULATION_START_DATE))

    elif fraud_type == 'mule':
        agent = FraudAgent(fraud_type='mule')
        mule_agents.append(agent)
        agent_population.append(agent)

# Step 4.5: Pre-Link Agents for Realistic Interactions
print("Pre-linking agents for realistic social and economic interactions...")
for ring_id, ring_data in fraud_rings.items():
    if mule_agents:
        assigned_mules = random.sample(mule_agents, min(random.randint(1, 2), len(mule_agents)))
        ring_data['mules'].extend(assigned_mules)
        print(f"Linked {ring_id} to {len(assigned_mules)} dedicated mule(s).")

print(f"Agent population created and linked: {len(agent_population)} total agents.")

# Step 5: Run the Simulation
print(f"Running simulation from {SIMULATION_START_DATE} to {SIMULATION_END_DATE}...")
delta = SIMULATION_END_DATE - SIMULATION_START_DATE
all_transactions = []

for i in range(delta.days + 1):
    current_date = SIMULATION_START_DATE + timedelta(days=i)
    current_datetime = datetime.combine(current_date, datetime.min.time())
    p2p_transfers_today = []
    
    if random.random() < 0.25:
        for ring_data in fraud_rings.values():
            if ring_data['members'] and ring_data['mules']:
                sender = random.choice(ring_data['members'])
                recipient = random.choice(ring_data['mules'])
                amount = sender.balance * random.uniform(0.05, 0.15)
                p2p_transfers_today.append({'sender': sender, 'recipient': recipient, 'amount': amount, 'desc': 'P2P to Mule'})

    for agent in agent_population:
        if isinstance(agent, FraudAgent):
            context = {'p2p_transfers': p2p_transfers_today}
            if agent.fraud_type == 'ring':
                context['ring_members'] = fraud_rings[agent.ring_id]['members']
            daily_events = agent.act(current_datetime, **context)
        else:
            daily_events = agent.act(current_datetime)
        
        if daily_events:
            all_transactions.extend(daily_events)

    for p2p in p2p_transfers_today:
        sender = p2p.get('sender')
        recipient = p2p['recipient']
        amount = p2p['amount']
        desc = p2p.get('desc', 'P2P Transfer')
        sender_id = p2p.get('sender_id', sender.agent_id if sender else 'EXTERNAL')
        
        if sender:
            debit_txn = sender.log_transaction("DEBIT", f"P2P to {recipient.agent_id[:6]}", amount, current_date)
            if debit_txn:
                all_transactions.append(debit_txn)
                credit_txn = recipient.log_transaction("CREDIT", f"{desc} from {sender_id[:6]}", amount, current_date)
                if credit_txn: all_transactions.append(credit_txn)
        else:
             credit_txn = recipient.log_transaction("CREDIT", f"{desc} from {sender_id[:6]}", amount, current_date)
             if credit_txn: all_transactions.append(credit_txn)

print("Simulation complete.")

# Step 6: Harvest and Export Data
print("Harvesting data for CSV export...")
agent_profile_data = [agent.to_dict() for agent in agent_population]
agent_df = pd.DataFrame(agent_profile_data)
transactions_df = pd.DataFrame(all_transactions)

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

agent_df.to_csv(os.path.join(output_dir, "agents.csv"), index=False)
transactions_df.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)

print("\nSuccessfully generated two files:")
print(f"1. {os.path.join(output_dir, 'agents.csv')} ({len(agent_df)} rows)")
print(f"2. {os.path.join(output_dir, 'transactions.csv')} ({len(transactions_df)} rows)")