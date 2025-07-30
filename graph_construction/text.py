# From your graph_construction directory, run:
import pandas as pd

agents_df = pd.read_csv('../output/agents.csv')
fraud_distribution = agents_df['fraud_type'].value_counts()
print('Fraud type distribution:')
print(fraud_distribution)
