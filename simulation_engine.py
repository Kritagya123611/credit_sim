# simulation_engine.py

import os
import random
import pandas as pd
import numpy as np
import uuid
from datetime import date, timedelta, datetime
from faker import Faker

# Step 1: Import all agent classes
# (This assumes you have an __init__.py file in your 'agents' folder)
from agents import (
    SalariedProfessional, GigWorker, GovernmentEmployee, Student, DailyWageLaborer,
    SmallBusinessOwner, Doctor, TechProfessional, PoliceOfficer, SeniorCitizen,
    DeliveryAgent, Lawyer, MigrantWorker, ContentCreator, Homemaker, FraudAgent
)

print("All agent classes imported successfully.")
fake = Faker('en_IN')

# Step 2: Define Simulation Parameters
TOTAL_POPULATION = 1000
SIMULATION_START_DATE = date(2024, 1, 1)
SIMULATION_END_DATE = date(2024, 12, 31)

# Step 3: Define Population Mix
LEGITIMATE_POPULATION_MIX = {
    SalariedProfessional: 0.12, TechProfessional: 0.04, SmallBusinessOwner: 0.18,
    GovernmentEmployee: 0.06, PoliceOfficer: 0.03, Doctor: 0.01, Lawyer: 0.01,
    GigWorker: 0.04, DeliveryAgent: 0.03, ContentCreator: 0.01, Student: 0.08,
    Homemaker: 0.11, DailyWageLaborer: 0.12, MigrantWorker: 0.04, SeniorCitizen: 0.04,
}
total_prob = sum(LEGITIMATE_POPULATION_MIX.values())
for k in LEGITIMATE_POPULATION_MIX: LEGITIMATE_POPULATION_MIX[k] /= total_prob

FRAUD_POPULATION_PERCENTAGE = 0.04
FRAUD_MIX = {'ring': 0.50, 'bust_out': 0.25, 'mule': 0.25}

# Step 4: Create the Agent Population
print(f"Creating a population of {TOTAL_POPULATION} agents...")
agent_population = []
legit_population = []
num_legitimate = int(TOTAL_POPULATION * (1 - FRAUD_POPULATION_PERCENTAGE))
num_fraud = TOTAL_POPULATION - num_legitimate

legit_archetypes = list(LEGITIMATE_POPULATION_MIX.keys())
legit_probabilities = list(LEGITIMATE_POPULATION_MIX.values())
for _ in range(num_legitimate):
    agent = np.random.choice(legit_archetypes, p=legit_probabilities)()
    legit_population.append(agent)
    agent_population.append(agent)

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
    p2p_transfers_today = []
    
    # --- Orchestrate Targeted Fraud Transactions ---
    if random.random() < 0.25:
        for ring_data in fraud_rings.values():
            if ring_data['members'] and ring_data['mules']:
                sender = random.choice(ring_data['members'])
                recipient = random.choice(ring_data['mules'])
                amount = sender.balance * random.uniform(0.05, 0.15)
                # Queue the transfer; it will be processed after all agents act
                p2p_transfers_today.append({'sender': sender, 'recipient': recipient, 'amount': amount, 'desc': 'P2P to Mule'})

    # --- Let Each Agent Act ---
    for agent in agent_population:
        current_datetime = datetime.combine(current_date, datetime.min.time())
        
        # CORRECTED BLOCK: Logic to handle context-aware vs simple 'act' calls
        if isinstance(agent, FraudAgent):
            context = {'p2p_transfers': p2p_transfers_today}
            if agent.fraud_type == 'ring':
                context['ring_members'] = fraud_rings[agent.ring_id]['members']
            
            daily_events = agent.act(current_datetime, **context)
        else:
            daily_events = agent.act(current_datetime)
        
        if daily_events:
            all_transactions.extend(daily_events)

    # --- Process the P2P transfers for the day ---
    for p2p in p2p_transfers_today:
        sender = p2p.get('sender')
        recipient = p2p['recipient']
        amount = p2p['amount']
        desc = p2p.get('desc', 'P2P Transfer')
        sender_id = p2p.get('sender_id', sender.agent_id if sender else 'EXTERNAL')
        
        # CORRECTED BLOCK: Ensure sender is debited for orchestrated transfers
        if sender:
            debit_txn = sender.log_transaction("DEBIT", f"P2P to {recipient.agent_id[:6]}", amount, current_date)
            if debit_txn:
                all_transactions.append(debit_txn)
                # Only credit the recipient if the sender had enough money
                credit_txn = recipient.log_transaction("CREDIT", f"{desc} from {sender_id[:6]}", amount, current_date)
                if credit_txn:
                    all_transactions.append(credit_txn)
        else:
            # This handles cases where the sender is not an agent (e.g., from a ring's act method)
             credit_txn = recipient.log_transaction("CREDIT", f"{desc} from {sender_id[:6]}", amount, current_date)
             if credit_txn:
                all_transactions.append(credit_txn)

print("Simulation complete.")

# Step 6: Harvest and Export Data
print("Harvesting data for CSV export...")
agent_profile_data = [agent.to_dict() for agent in agent_population]
agent_df = pd.DataFrame(agent_profile_data)
transactions_df = pd.DataFrame(all_transactions)

# --- Ensure output directory exists ---
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

agent_df.to_csv(os.path.join(output_dir, "agents.csv"), index=False)
transactions_df.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)

print("\nSuccessfully generated two files:")
print(f"1. {os.path.join(output_dir, 'agents.csv')} ({len(agent_df)} rows)")
print(f"2. {os.path.join(output_dir, 'transactions.csv')} ({len(transactions_df)} rows)")