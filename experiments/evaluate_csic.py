import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error

# Load your output CSV and ground truth
output = pd.read_csv('risk_register.csv')
ground_truth = pd.read_csv('data/datasets/csic_database.csv')

# Map ground truth 'classification' to your threat labels
label_map = {
    'SQL Injection': 'sql_injection',
    'XSS': 'xss',
    'Normal': 'benign',
    # add others as needed
}
y_true = ground_truth['classification'].map(label_map).fillna('unknown')
y_pred = output['extracted_threat']

print(classification_report(y_true, y_pred))