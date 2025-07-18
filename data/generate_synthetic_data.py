"""
Advanced Synthetic Data Generator for Customer Churn Prediction
Enterprise-grade data generation for ML model training

This script generates realistic synthetic data with:
- Proper business logic and correlations
- Data quality issues that mirror real-world scenarios
- Canadian-specific patterns (provinces, postal codes, etc.)
- Realistic churn patterns based on customer behavior
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import uuid
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Initialize Faker for Canadian locale
fake = Faker('en_CA')

# Configuration
N_CUSTOMERS = 10000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 1, 1)
CHURN_RATE = 0.18  # 18% realistic churn rate

# Canadian provinces and their relative populations (for realistic distribution)
PROVINCES = {
    'ON': 0.39,  # Ontario
    'QC': 0.23,  # Quebec
    'BC': 0.13,  # British Columbia
    'AB': 0.12,  # Alberta
    'MB': 0.04,  # Manitoba
    'SK': 0.03,  # Saskatchewan
    'NS': 0.03,  # Nova Scotia
    'NB': 0.02,  # New Brunswick
    'NL': 0.01,  # Newfoundland and Labrador
    'PE': 0.00,  # Prince Edward Island
}

# Business logic parameters
SUBSCRIPTION_TYPES = ['Basic', 'Premium', 'Enterprise']
SUBSCRIPTION_PRICES = {'Basic': 29.99, 'Premium': 59.99, 'Enterprise': 149.99}
PAYMENT_METHODS = ['Credit Card', 'Bank Transfer', 'PayPal']
ISSUE_CATEGORIES = ['Billing', 'Technical', 'Account', 'Cancellation', 'General']
TRANSACTION_TYPES = ['Payment', 'Refund', 'Fee', 'Adjustment']

def generate_canadian_postal_code(province):
    """Generate realistic Canadian postal codes by province"""
    # First letter corresponds to province
    province_letters = {
        'ON': ['K', 'L', 'M', 'N', 'P'],
        'QC': ['G', 'H', 'J'],
        'BC': ['V'],
        'AB': ['T'],
        'MB': ['R'],
        'SK': ['S'],
        'NS': ['B'],
        'NB': ['E'],
        'NL': ['A'],
        'PE': ['C'],
    }
    
    first_letter = random.choice(province_letters[province])
    return f"{first_letter}{random.randint(0,9)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')} {random.randint(0,9)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(0,9)}"

def generate_customers(n_customers):
    """Generate realistic customer data with Canadian characteristics"""
    customers = []
    
    for i in range(n_customers):
        # Select province based on population distribution
        province = np.random.choice(list(PROVINCES.keys()), p=list(PROVINCES.values()))
        
        # Generate customer with realistic patterns
        subscription_type = np.random.choice(SUBSCRIPTION_TYPES, p=[0.6, 0.3, 0.1])  # Most choose Basic
        
        # Account creation date (customers joined over time)
        account_creation_date = fake.date_between(start_date=START_DATE, end_date=END_DATE - timedelta(days=30))
        
        # Generate correlated data
        customer = {
            'customer_id': f"CUST_{str(i+1).zfill(6)}",
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'email': fake.email(),
            'phone': fake.phone_number(),
            'date_of_birth': fake.date_of_birth(minimum_age=18, maximum_age=80),
            'province': province,
            'city': fake.city(),
            'postal_code': generate_canadian_postal_code(province),
            'account_creation_date': account_creation_date,
            'subscription_type': subscription_type,
            'monthly_charges': SUBSCRIPTION_PRICES[subscription_type] * random.uniform(0.95, 1.05),  # Small variations
            'total_charges': 0,  # Will be calculated later
            'contract_length': np.random.choice([12, 24, 36], p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(PAYMENT_METHODS, p=[0.7, 0.2, 0.1]),
            'paperless_billing': random.choice([True, False]),
            'auto_pay': random.choice([True, False]),
            'referral_code': fake.bothify(text='REF####') if random.random() < 0.3 else None,
            'last_login_date': fake.date_between(start_date=account_creation_date, end_date=END_DATE) if random.random() < 0.95 else None,
            'account_status': 'Active'
        }
        
        customers.append(customer)
    
    return pd.DataFrame(customers)

def generate_transactions(customers_df):
    """Generate realistic transaction data with business logic"""
    transactions = []
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        account_creation = customer['account_creation_date']
        monthly_charge = customer['monthly_charges']
        
        # Generate monthly payments based on account age
        current_date = account_creation
        total_charges = 0
        
        # Convert END_DATE to date object for comparison
        end_date = END_DATE.date() if isinstance(END_DATE, datetime) else END_DATE
        
        while current_date < end_date:
            # Monthly payment (with some realistic variations)
            if random.random() < 0.95:  # 95% payment success rate
                payment_amount = monthly_charge * random.uniform(0.98, 1.02)
                status = 'Completed'
            else:
                payment_amount = monthly_charge
                status = random.choice(['Failed', 'Pending'])
            
            transaction = {
                'transaction_id': f"TXN_{uuid.uuid4().hex[:12].upper()}",
                'customer_id': customer_id,
                'transaction_date': current_date,
                'transaction_type': 'Payment',
                'amount': payment_amount,
                'currency': 'CAD',
                'payment_method': customer['payment_method'],
                'status': status,
                'description': f"Monthly subscription - {customer['subscription_type']}",
                'merchant_category': 'Subscription Services',
                'is_recurring': True,
                'failure_reason': 'Insufficient funds' if status == 'Failed' else None,
                'processing_fee': 0.99 if customer['payment_method'] == 'Credit Card' else 0.00
            }
            
            transactions.append(transaction)
            
            if status == 'Completed':
                total_charges += payment_amount
            
            # Add occasional refunds (2% chance)
            if random.random() < 0.02:
                refund_transaction = {
                    'transaction_id': f"REF_{uuid.uuid4().hex[:12].upper()}",
                    'customer_id': customer_id,
                    'transaction_date': current_date + timedelta(days=random.randint(1, 10)),
                    'transaction_type': 'Refund',
                    'amount': -payment_amount,
                    'currency': 'CAD',
                    'payment_method': customer['payment_method'],
                    'status': 'Completed',
                    'description': 'Service refund',
                    'merchant_category': 'Refunds',
                    'is_recurring': False,
                    'failure_reason': None,
                    'processing_fee': 0.00
                }
                transactions.append(refund_transaction)
                total_charges -= payment_amount
            
            # Move to next month
            current_date += timedelta(days=30)
        
        # Update total charges in customers dataframe
        customers_df.loc[customers_df['customer_id'] == customer_id, 'total_charges'] = total_charges
    
    return pd.DataFrame(transactions)

def generate_support_tickets(customers_df):
    """Generate realistic support ticket data with patterns that correlate with churn"""
    tickets = []
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        account_creation = customer['account_creation_date']
        
        # Number of tickets varies by customer type (Enterprise customers get more support)
        if customer['subscription_type'] == 'Enterprise':
            avg_tickets = 4
        elif customer['subscription_type'] == 'Premium':
            avg_tickets = 2
        else:
            avg_tickets = 1
        
        # Generate tickets with realistic timing
        n_tickets = np.random.poisson(avg_tickets)
        
        for _ in range(n_tickets):
            # Convert END_DATE to date object for comparison
            end_date = END_DATE.date() if isinstance(END_DATE, datetime) else END_DATE
            created_date = fake.date_between(start_date=account_creation, end_date=end_date)
            
            # Issue category influences resolution time and satisfaction
            issue_category = np.random.choice(ISSUE_CATEGORIES, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            
            # Priority based on issue category
            if issue_category == 'Cancellation':
                priority = np.random.choice(['High', 'Critical'], p=[0.7, 0.3])
            elif issue_category == 'Billing':
                priority = np.random.choice(['Medium', 'High'], p=[0.8, 0.2])
            else:
                priority = np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.5, 0.1])
            
            # Resolution time based on priority
            if priority == 'Critical':
                resolution_time = random.randint(1, 4)
            elif priority == 'High':
                resolution_time = random.randint(2, 12)
            elif priority == 'Medium':
                resolution_time = random.randint(4, 24)
            else:
                resolution_time = random.randint(8, 72)
            
            # Resolution date
            resolution_date = created_date + timedelta(hours=resolution_time)
            # Convert END_DATE to date object for comparison
            end_date = END_DATE.date() if isinstance(END_DATE, datetime) else END_DATE
            if resolution_date > end_date:
                resolution_date = None
                status = 'Open'
            else:
                status = 'Resolved'
            
            # Satisfaction score (lower for billing/cancellation issues)
            if issue_category in ['Billing', 'Cancellation']:
                satisfaction_score = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.15, 0.05])
            else:
                satisfaction_score = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.25, 0.4, 0.2])
            
            ticket = {
                'ticket_id': f"TICK_{uuid.uuid4().hex[:12].upper()}",
                'customer_id': customer_id,
                'created_date': created_date,
                'issue_category': issue_category,
                'priority': priority,
                'status': status,
                'resolution_date': resolution_date,
                'satisfaction_score': satisfaction_score if status == 'Resolved' else None,
                'agent_id': f"AGENT_{random.randint(1, 50):03d}",
                'channel': np.random.choice(['Phone', 'Email', 'Chat', 'Social Media'], p=[0.5, 0.3, 0.15, 0.05]),
                'first_response_time_hours': random.randint(1, 8),
                'resolution_time_hours': resolution_time if status == 'Resolved' else None,
                'escalated': random.random() < 0.1,  # 10% escalation rate
                'description': f"{issue_category} issue reported by customer",
                'resolution_notes': f"Issue resolved via {np.random.choice(['phone', 'email', 'system update'])}" if status == 'Resolved' else None
            }
            
            tickets.append(ticket)
    
    return pd.DataFrame(tickets)

def generate_churn_labels(customers_df, transactions_df, support_tickets_df):
    """Generate realistic churn labels with business logic"""
    churn_labels = []
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        
        # Calculate churn probability based on customer behavior
        churn_probability = 0.1  # Base churn probability
        
        # Factors that increase churn probability
        customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id]
        customer_tickets = support_tickets_df[support_tickets_df['customer_id'] == customer_id]
        
        # Failed transactions increase churn risk
        failed_transactions = len(customer_transactions[customer_transactions['status'] == 'Failed'])
        churn_probability += failed_transactions * 0.05
        
        # Support tickets increase churn risk
        churn_probability += len(customer_tickets) * 0.02
        
        # Billing/cancellation tickets significantly increase churn risk
        billing_tickets = len(customer_tickets[customer_tickets['issue_category'] == 'Billing'])
        cancellation_tickets = len(customer_tickets[customer_tickets['issue_category'] == 'Cancellation'])
        churn_probability += billing_tickets * 0.1 + cancellation_tickets * 0.3
        
        # Low satisfaction scores increase churn risk
        satisfaction_scores = customer_tickets['satisfaction_score'].dropna()
        if len(satisfaction_scores) > 0:
            avg_satisfaction = satisfaction_scores.mean()
            if avg_satisfaction < 3:
                churn_probability += 0.15
        
        # Long-term customers less likely to churn
        # Convert END_DATE to date object for comparison
        end_date = END_DATE.date() if isinstance(END_DATE, datetime) else END_DATE
        account_age_months = (end_date - customer['account_creation_date']).days / 30
        if account_age_months > 12:
            churn_probability *= 0.8
        
        # High-value customers less likely to churn
        if customer['subscription_type'] == 'Enterprise':
            churn_probability *= 0.7
        
        # Determine if customer churned
        is_churned = random.random() < min(churn_probability, 0.6)  # Cap at 60%
        
        # If churned, determine when and why
        if is_churned:
            # Churn date is random between account creation and end date
            churn_date = fake.date_between(
                start_date=customer['account_creation_date'] + timedelta(days=30),
                end_date=end_date
            )
            
            # Churn reason based on customer behavior
            if cancellation_tickets > 0:
                churn_reason = 'Service Quality'
            elif billing_tickets > 0:
                churn_reason = 'Price'
            elif failed_transactions > 2:
                churn_reason = 'Payment Issues'
            else:
                churn_reason = np.random.choice(['Competition', 'Price', 'Service Quality', 'Technical Issues'])
            
            churn_type = 'Voluntary' if churn_reason != 'Payment Issues' else 'Involuntary'
        else:
            churn_date = None
            churn_reason = None
            churn_type = None
        
        # Calculate last transaction date
        last_transaction = customer_transactions['transaction_date'].max()
        days_since_last_transaction = (end_date - last_transaction).days if pd.notna(last_transaction) else None
        
        churn_label = {
            'customer_id': customer_id,
            'churn_date': churn_date,
            'is_churned': is_churned,
            'churn_reason': churn_reason,
            'churn_type': churn_type,
            'days_since_last_transaction': days_since_last_transaction,
            'last_transaction_date': last_transaction,
            'total_customer_value': customer['total_charges'],
            'customer_lifetime_months': account_age_months,
            'win_back_eligible': is_churned and churn_type == 'Voluntary',
            'prediction_date': end_date - timedelta(days=30),  # Prediction made 30 days ago
            'actual_churn_date': churn_date
        }
        
        churn_labels.append(churn_label)
    
    return pd.DataFrame(churn_labels)

def main():
    """Generate all synthetic data and save to CSV files"""
    print("Generating Customer Churn Prediction Dataset...")
    print("=" * 50)
    
    # Create output directory (relative to current script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'raw')
    
    # Alternative: check if we're in a different working directory
    if not os.path.exists(script_dir) or not os.access(script_dir, os.W_OK):
        # Fallback to current working directory + data/raw
        output_dir = os.path.join(os.getcwd(), 'data', 'raw')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Generate customers
    print("1. Generating customers data...")
    customers_df = generate_customers(N_CUSTOMERS)
    customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
    print(f"   ✓ Generated {len(customers_df)} customers")
    
    # Generate transactions
    print("2. Generating transactions data...")
    transactions_df = generate_transactions(customers_df)
    transactions_df.to_csv(f'{output_dir}/transactions.csv', index=False)
    print(f"   ✓ Generated {len(transactions_df)} transactions")
    
    # Generate support tickets
    print("3. Generating support tickets data...")
    support_tickets_df = generate_support_tickets(customers_df)
    support_tickets_df.to_csv(f'{output_dir}/support_tickets.csv', index=False)
    print(f"   ✓ Generated {len(support_tickets_df)} support tickets")
    
    # Generate churn labels
    print("4. Generating churn labels...")
    churn_labels_df = generate_churn_labels(customers_df, transactions_df, support_tickets_df)
    churn_labels_df.to_csv(f'{output_dir}/churn_labels.csv', index=False)
    churn_rate_actual = churn_labels_df['is_churned'].mean()
    print(f"   ✓ Generated churn labels with {churn_rate_actual:.1%} churn rate")
    
    # Save updated customers data (with calculated total_charges)
    customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
    
    # Generate summary statistics
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total Customers: {len(customers_df):,}")
    print(f"Total Transactions: {len(transactions_df):,}")
    print(f"Total Support Tickets: {len(support_tickets_df):,}")
    print(f"Churn Rate: {churn_rate_actual:.1%}")
    print(f"Date Range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    
    print("\nCustomer Distribution by Province:")
    province_counts = customers_df['province'].value_counts()
    for province, count in province_counts.items():
        print(f"   {province}: {count:,} ({count/len(customers_df):.1%})")
    
    print("\nSubscription Type Distribution:")
    subscription_counts = customers_df['subscription_type'].value_counts()
    for sub_type, count in subscription_counts.items():
        print(f"   {sub_type}: {count:,} ({count/len(customers_df):.1%})")
    
    print("\nTransaction Status Distribution:")
    transaction_status = transactions_df['status'].value_counts()
    for status, count in transaction_status.items():
        print(f"   {status}: {count:,} ({count/len(transactions_df):.1%})")
    
    print("\nSupport Ticket Category Distribution:")
    ticket_categories = support_tickets_df['issue_category'].value_counts()
    for category, count in ticket_categories.items():
        print(f"   {category}: {count:,} ({count/len(support_tickets_df):.1%})")
    
    print("\n✓ All data generated successfully!")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main()