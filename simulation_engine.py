import os
import random
import pandas as pd
import numpy as np
import uuid
from datetime import date, timedelta, datetime
from faker import Faker

# Step 1: Import configurations and all agent classes
from config_pkg import ECONOMIC_CLASSES, FINANCIAL_PERSONALITIES, ARCHETYPE_BASE_RISK, get_risk_profile_from_score
from config_pkg.p2p_structure import RealisticP2PStructure  # ✅ Import for realistic P2P handling
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
SIMULATION_END_DATE = date(2024, 6, 30)

# Step 3: Define Population & Behavior Distributions  
POPULATION_MIX = {
    SalariedProfessional: 0.12, TechProfessional: 0.04, SmallBusinessOwner: 0.18,
    GovernmentEmployee: 0.06, PoliceOfficer: 0.03, Doctor: 0.01, Lawyer: 0.01,
    GigWorker: 0.04, DeliveryAgent: 0.03, ContentCreator: 0.01, Student: 0.08,
    Homemaker: 0.11, DailyWageLaborer: 0.12, MigrantWorker: 0.04, SeniorCitizen: 0.04,
}
FRAUD_POPULATION_PERCENTAGE = 0.04
FRAUD_MIX = {'ring': 0.50, 'bust_out': 0.25, 'mule': 0.25}

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

legit_archetypes = list(POPULATION_MIX.keys())
legit_probabilities = list(POPULATION_MIX.values())
legit_probabilities /= np.sum(legit_probabilities)

for _ in range(num_legitimate):
    ChosenAgentClass = np.random.choice(legit_archetypes, p=legit_probabilities)
    class_dist = BEHAVIORAL_DISTRIBUTIONS[ChosenAgentClass]["class"]
    pers_dist = BEHAVIORAL_DISTRIBUTIONS[ChosenAgentClass]["personality"]
    chosen_class = np.random.choice(list(ECONOMIC_CLASSES.keys()), p=class_dist)
    chosen_personality = np.random.choice(list(FINANCIAL_PERSONALITIES.keys()), p=pers_dist)
    agent = ChosenAgentClass(economic_class=chosen_class, financial_personality=chosen_personality)
    legit_population.append(agent)
    agent_population.append(agent)

# --- Create Fraud Population ---
fraud_rings = {}
mule_agents = []

# ✅ NEW: Weighted fraud agent mimicking distribution
mimic_choices = [
    (GigWorker, 0.20),          # High vulnerability - gig economy workers
    (Student, 0.18),            # High vulnerability - limited income, tech-savvy
    (DailyWageLaborer, 0.15),   # High vulnerability - economic desperation
    (DeliveryAgent, 0.12),      # High vulnerability - access to networks
    (MigrantWorker, 0.10),      # High vulnerability - limited banking knowledge
    (SalariedProfessional, 0.08), # Medium vulnerability - stable but targeted
    (Homemaker, 0.07),          # Medium vulnerability - limited financial literacy
    (ContentCreator, 0.05),     # Medium vulnerability - irregular income
    (SmallBusinessOwner, 0.05)  # Lower vulnerability - business experience
]

agents, weights = zip(*mimic_choices)

for _ in range(num_fraud):
    fraud_type = np.random.choice(list(FRAUD_MIX.keys()), p=list(FRAUD_MIX.values()))
    mimic_class = random.choices(agents, weights=weights)[0]  # ✅ Weighted selection
    
    if fraud_type == 'ring':
        num_rings = max(1, int(num_fraud * FRAUD_MIX['ring'] / 4))
        ring_id = f"ring_{random.randint(1, num_rings)}"
        if ring_id not in fraud_rings:
            shared_footprint = {'device_id': str(uuid.uuid4()), 'ip_address': fake.ipv4()}
            fraud_rings[ring_id] = {'members': [], 'footprint': shared_footprint, 'mules': []}
        agent = FraudAgent(fraud_type='ring', ring_id=ring_id, shared_footprint=fraud_rings[ring_id]['footprint'], mimic_agent_class=mimic_class)
        fraud_rings[ring_id]['members'].append(agent)
    elif fraud_type == 'bust_out':
        agent = FraudAgent(fraud_type='bust_out', creation_date=SIMULATION_START_DATE, mimic_agent_class=mimic_class)
    elif fraud_type == 'mule':
        agent = FraudAgent(fraud_type='mule', mimic_agent_class=mimic_class)
        mule_agents.append(agent)
    agent_population.append(agent)

# Step 4.5: Pre-Link Agents for Realistic Interactions
print("Pre-linking agents for realistic social and economic interactions...")
for ring_id, ring_data in fraud_rings.items():
    if mule_agents:
        assigned_mules = random.sample(mule_agents, min(random.randint(1, 2), len(mule_agents)))
        ring_data['mules'].extend(assigned_mules)
        print(f"Linked {ring_id} to {len(assigned_mules)} dedicated mule(s).")

# ✅ COMPREHENSIVE AGENT PRE-LINKING FOR ALL P2P BEHAVIORS
for agent in agent_population:
    
    # --- EXISTING LINKAGES (MAINTAINED) ---
    if isinstance(agent, ContentCreator):
        # Main collaborators
        possible_collaborators = [p for p in legit_population if p.agent_id != agent.agent_id]
        if len(possible_collaborators) > 3:
            agent.collaborators = random.sample(possible_collaborators, k=random.randint(1, 3))
        
        # ✅ Additional networks for ContentCreator
        possible_creators = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, ContentCreator)]
        if len(possible_creators) > 1:
            agent.creator_network = random.sample(possible_creators, k=random.randint(1, min(3, len(possible_creators))))
        
        possible_freelancers = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (GigWorker, TechProfessional))]
        if len(possible_freelancers) > 2:
            agent.freelancer_network = random.sample(possible_freelancers, k=random.randint(1, 2))
        
        possible_brands = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, SmallBusinessOwner)]
        if len(possible_brands) > 1:
            agent.brand_contacts = random.sample(possible_brands, k=random.randint(0, 2))

    if isinstance(agent, Doctor):
        possible_providers = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Lawyer, TechProfessional, SmallBusinessOwner))]
        if len(possible_providers) > 2:
            agent.service_providers = random.sample(possible_providers, k=random.randint(1, 2))
            
    if isinstance(agent, GigWorker):
        possible_contacts = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (GigWorker, Student, DailyWageLaborer))]
        if len(possible_contacts) > 5:
            agent.contacts = random.sample(possible_contacts, k=random.randint(2, 5))

    if isinstance(agent, Student):
        possible_contacts = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Student, GigWorker))]
        if len(possible_contacts) > 3:
            agent.contacts = random.sample(possible_contacts, k=random.randint(2, 4))

    if isinstance(agent, SmallBusinessOwner):
        possible_employees = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (DailyWageLaborer, GigWorker))]
        num_to_hire = min(len(possible_employees), agent.num_employees)
        if num_to_hire > 0:
            agent.employees = random.sample(possible_employees, k=num_to_hire)

    if isinstance(agent, SalariedProfessional):
        possible_dependents = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Homemaker, Student, SeniorCitizen))]
        if possible_dependents:
            agent.dependents = [random.choice(possible_dependents)]

    if isinstance(agent, (GovernmentEmployee, PoliceOfficer)):
        possible_recipients = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Homemaker, SeniorCitizen))]
        if possible_recipients:
            agent.family_member_recipient = random.choice(possible_recipients)

    # ✅ NEW COMPREHENSIVE P2P LINKAGES
    
    # --- SENIOR CITIZEN LINKAGES ---
    if isinstance(agent, SeniorCitizen):
        possible_family = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (SalariedProfessional, Homemaker, Student, TechProfessional))]
        if len(possible_family) > 1:
            agent.family_members = random.sample(possible_family, k=random.randint(1, 2))
    
    # --- MIGRANT WORKER LINKAGES ---
    if isinstance(agent, MigrantWorker):
        possible_family = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Homemaker, SeniorCitizen, Student))]
        if len(possible_family) > 1:
            agent.family_back_home = random.sample(possible_family, k=random.randint(1, 3))
    
    # --- LAWYER LINKAGES ---
    if isinstance(agent, Lawyer):
        possible_network = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Doctor, TechProfessional, SmallBusinessOwner, Lawyer))]
        if len(possible_network) > 2:
            agent.professional_network = random.sample(possible_network, k=random.randint(1, 3))
        
        # Junior associate assignment
        possible_juniors = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Student, GigWorker))]
        if possible_juniors:
            agent.junior_associate = random.choice(possible_juniors)
    
    # --- DAILY WAGE LABORER LINKAGES ---
    if isinstance(agent, DailyWageLaborer):
        possible_workers = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (DailyWageLaborer, MigrantWorker, DeliveryAgent))]
        if len(possible_workers) > 3:
            agent.worker_network = random.sample(possible_workers, k=random.randint(2, 4))
        
        # Family recipient for remittances
        possible_family = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Homemaker, SeniorCitizen))]
        if possible_family:
            agent.family_recipient = random.choice(possible_family)
    
    # --- TECH PROFESSIONAL LINKAGES ---
    if isinstance(agent, TechProfessional):
        # Professional network
        possible_professionals = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (TechProfessional, ContentCreator, SmallBusinessOwner))]
        if len(possible_professionals) > 2:
            agent.professional_network = random.sample(possible_professionals, k=random.randint(1, 3))
        
        # Social contacts
        possible_contacts = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (TechProfessional, SalariedProfessional, Doctor))]
        if len(possible_contacts) > 3:
            agent.contacts = random.sample(possible_contacts, k=random.randint(2, 4))
        
        # Family dependents
        possible_dependents = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Homemaker, SeniorCitizen, Student))]
        if len(possible_dependents) > 1:
            agent.family_dependents = random.sample(possible_dependents, k=random.randint(0, 2))
    
    # --- HOMEMAKER LINKAGES ---
    if isinstance(agent, Homemaker):
        # Social circle
        possible_circle = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Homemaker, SalariedProfessional, SeniorCitizen))]
        if len(possible_circle) > 2:
            agent.social_circle = random.sample(possible_circle, k=random.randint(1, 3))
        
        # Extended family
        possible_extended = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (SeniorCitizen, SalariedProfessional))]
        if len(possible_extended) > 1:
            agent.extended_family = random.sample(possible_extended, k=random.randint(0, 2))
        
        # Children contacts (education related)
        possible_children_contacts = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Student, TechProfessional))]
        if len(possible_children_contacts) > 1:
            agent.children_contacts = random.sample(possible_children_contacts, k=random.randint(0, 2))
    
    # --- DELIVERY AGENT LINKAGES ---
    if isinstance(agent, DeliveryAgent):
        # Fellow delivery agents
        possible_agents = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (DeliveryAgent, GigWorker, DailyWageLaborer))]
        if len(possible_agents) > 3:
            agent.fellow_agents = random.sample(possible_agents, k=random.randint(2, 4))
        
        # Family back home
        possible_family = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (Homemaker, SeniorCitizen, Student))]
        if len(possible_family) > 1:
            agent.family_back_home = random.sample(possible_family, k=random.randint(1, 2))
        
        # Peer network (broader gig workers)
        possible_peers = [p for p in legit_population if p.agent_id != agent.agent_id and isinstance(p, (GigWorker, DailyWageLaborer))]
        if len(possible_peers) > 2:
            agent.peer_network = random.sample(possible_peers, k=random.randint(1, 3))

print(f"Agent population created and linked: {len(agent_population)} total agents.")

# Step 5: Run the Simulation
print(f"Running simulation from {SIMULATION_START_DATE} to {SIMULATION_END_DATE}...")
delta = SIMULATION_END_DATE - SIMULATION_START_DATE
all_transactions = []

for i in range(delta.days + 1):
    current_date = SIMULATION_START_DATE + timedelta(days=i)
    current_datetime = datetime.combine(current_date, datetime.min.time())
    p2p_transfers_today = []
    
    # ✅ UPDATED: Inter-agent Fraud Ring Transfers with standardized descriptions
    if random.random() < 0.25:
        for ring_data in fraud_rings.values():
            if ring_data['members'] and ring_data['mules']:
                sender = random.choice(ring_data['members'])
                recipient = random.choice(ring_data['mules'])
                amount = sender.balance * random.uniform(0.05, 0.15)
                if amount > 0:
                    p2p_transfers_today.append({
                        'sender': sender, 
                        'recipient': recipient, 
                        'amount': amount, 
                        'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                        'channel': 'UPI'  # ✅ Realistic channel
                    })

    # ✅ UPDATED: Inter-agent UPI Transfers with realistic channel selection
    if random.random() < 0.15 and len(legit_population) > 1:
        for _ in range(random.randint(1, 5)):
            sender, recipient = random.sample(legit_population, 2)
            amount = random.uniform(100, 2500)
            
            # ✅ NEW: Select realistic channel based on amount
            if amount > 50000:
                channel = random.choice(['IMPS', 'NEFT'])  # Higher amounts use traditional channels
            else:
                channel = RealisticP2PStructure.select_realistic_channel()  # Market-based selection
            
            if sender.balance > amount:
                p2p_transfers_today.append({
                    'sender': sender, 
                    'recipient': recipient, 
                    'amount': amount, 
                    'desc': 'UPI P2P Transfer',  # ✅ Standardized description
                    'channel': channel  # ✅ Realistic channel
                })

    # --- AGENT ACTION LOOP ---
    for agent in agent_population:
        agent__context = {'p2p_transfers': p2p_transfers_today}
        if isinstance(agent, FraudAgent) and agent.fraud_type == 'ring':
            agent__context['ring_members'] = fraud_rings[agent.ring_id]['members']
        daily_events = agent.act(current_datetime, **agent__context)
        if daily_events:
            all_transactions.extend(daily_events)

    # ✅ UPDATED: Process all queued transfers with realistic bank-visible descriptions
    for p2p in p2p_transfers_today:
        sender = p2p.get('sender')
        recipient = p2p.get('recipient')
        amount = p2p.get('amount')
        channel = p2p.get('channel', 'UPI')
        
        if sender and recipient and amount > 0:
            # ✅ UPDATED: Let BaseAgent handle realistic formatting automatically
            debit_txn = sender.log_transaction(
                txn_type="DEBIT", 
                description="UPI P2P Transfer",  # Generic description (gets reformatted automatically)
                amount=amount, 
                date=current_datetime, 
                channel=channel, 
                recipient_id=recipient.agent_id  # This triggers realistic formatting
            )
            
            if debit_txn:
                all_transactions.append(debit_txn)
                credit_txn = recipient.log_transaction(
                    txn_type="CREDIT", 
                    description="UPI P2P Transfer",  # Generic description (gets reformatted automatically)
                    amount=amount, 
                    date=current_datetime, 
                    channel=channel,
                    recipient_id=sender.agent_id  # This triggers realistic formatting
                )
                
                if credit_txn: 
                    all_transactions.append(credit_txn)

print("Simulation complete.")

# ✅ NEW: Enhanced transaction analysis
print("\n=== TRANSACTION ANALYSIS ===")
p2p_transactions = [txn for txn in all_transactions if any(indicator in txn['description'] for indicator in ['UPI P2P', 'IMPS TXN', 'P2P TXN', 'NEFT CR', 'RTGS CR', 'WALLET TXN'])]
total_p2p = len(p2p_transactions)
total_transactions = len(all_transactions)

print(f"Total transactions: {total_transactions:,}")
print(f"P2P transactions: {total_p2p:,} ({total_p2p/total_transactions*100:.1f}%)")

# ✅ NEW: Channel breakdown
channel_counts = {}
for txn in p2p_transactions:
    channel = txn['channel']
    channel_counts[channel] = channel_counts.get(channel, 0) + 1

print("\nP2P Channel Distribution:")
for channel, count in sorted(channel_counts.items()):
    percentage = count / total_p2p * 100 if total_p2p > 0 else 0
    print(f"  {channel}: {count:,} ({percentage:.1f}%)")

# ✅ NEW: Fraud pattern analysis
fraud_agents = [agent for agent in agent_population if hasattr(agent, 'fraud_type')]
fraud_transactions = [txn for txn in all_transactions if any(agent.agent_id == txn['agent_id'] for agent in fraud_agents)]
print(f"\nFraud-related transactions: {len(fraud_transactions):,}")

# ✅ NEW: Analyze fraud rings
print(f"Fraud Rings Created: {len(fraud_rings)}")
for ring_id, ring_data in fraud_rings.items():
    members = len(ring_data['members'])
    mules = len(ring_data['mules'])
    print(f"  {ring_id}: {members} members, {mules} mules")

# ✅ NEW: Fraud mimicking analysis
print(f"\nFraud Agent Mimicking Distribution:")
mimic_counts = {}
for agent in fraud_agents:
    archetype = agent.archetype_name.split(' (')[0] if '(' in agent.archetype_name else agent.archetype_name
    # Get the actual mimicked class name
    mimicked_class = 'Unknown'
    if hasattr(agent, 'employment_status'):
        # Try to infer from employment status patterns
        employment_mapping = {
            'Gig_Work_Contractor': 'GigWorker/DeliveryAgent',
            'Student': 'Student', 
            'Manual_Labor': 'DailyWageLaborer',
            'Self-Employed': 'SmallBusinessOwner/ContentCreator',
            'Full_Time_Employee': 'SalariedProfessional',
            'Unemployed_Homemaker': 'Homemaker',
            'Migrant_Daily_Wage': 'MigrantWorker'
        }
        mimicked_class = employment_mapping.get(agent.employment_status, 'Unknown')
    
    mimic_counts[mimicked_class] = mimic_counts.get(mimicked_class, 0) + 1

for mimic_type, count in sorted(mimic_counts.items()):
    percentage = count / len(fraud_agents) * 100 if fraud_agents else 0
    print(f"  {mimic_type}: {count} ({percentage:.1f}%)")

# Step 6: Harvest and Export Data
print("\nHarvesting data for CSV export...")
agent_profile_data = [agent.to_dict() for agent in agent_population]
agent_df = pd.DataFrame(agent_profile_data)
transactions_df = pd.DataFrame(all_transactions)

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

agent_df.to_csv(os.path.join(output_dir, "agents.csv"), index=False)
if not transactions_df.empty:
    transactions_df.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
else:
    print("Warning: No transactions were generated.")

print("\nSuccessfully generated two files:")
print(f"1. {os.path.join(output_dir, 'agents.csv')} ({len(agent_df)} rows)")
print(f"2. {os.path.join(output_dir, 'transactions.csv')} ({len(transactions_df)} rows)")
