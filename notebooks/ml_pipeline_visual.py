"""
Customer Churn Prediction ML Pipeline with Comprehensive Visualizations
Advanced ML engineering pipeline with business-ready visualizations

Features:
- Exploratory Data Analysis with business insights
- Feature engineering with domain expertise
- Multiple model comparison with visual evaluation
- ROC curves, precision-recall curves, and feature importance plots
- Business-focused visualizations (customer segments, churn patterns)
- Production-ready model serialization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

class ChurnPredictionPipelineVisual:
    """
    ML pipeline with comprehensive visualizations for business stakeholders
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.model_performance = {}
        self.figures = {}
        
    def load_data(self, data_path=None):
        """Load and prepare data from CSV files"""
        if data_path is None:
            data_path = '/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/data/raw'
        
        # Load all tables
        customers = pd.read_csv(f'{data_path}/customers.csv')
        transactions = pd.read_csv(f'{data_path}/transactions.csv')
        support_tickets = pd.read_csv(f'{data_path}/support_tickets.csv')
        churn_labels = pd.read_csv(f'{data_path}/churn_labels.csv')
        
        print("Data loaded successfully:")
        print(f"- Customers: {len(customers):,}")
        print(f"- Transactions: {len(transactions):,}")
        print(f"- Support Tickets: {len(support_tickets):,}")
        print(f"- Churn Labels: {len(churn_labels):,}")
        
        return customers, transactions, support_tickets, churn_labels
    
    def create_eda_visualizations(self, customers, transactions, support_tickets, churn_labels):
        """Create comprehensive EDA visualizations"""
        print("Creating EDA visualizations...")
        
        # Create figure directory
        import os
        os.makedirs('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures', exist_ok=True)
        
        # 1. Churn Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Churn rate by province
        churn_by_province = customers.merge(churn_labels, on='customer_id')
        churn_by_province = churn_by_province.groupby('province')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
        churn_by_province.columns = ['province', 'total_customers', 'churned_customers', 'churn_rate']
        
        axes[0,0].bar(churn_by_province['province'], churn_by_province['churn_rate'])
        axes[0,0].set_title('Churn Rate by Province')
        axes[0,0].set_ylabel('Churn Rate')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Churn rate by subscription type
        churn_by_subscription = customers.merge(churn_labels, on='customer_id')
        churn_by_subscription = churn_by_subscription.groupby('subscription_type')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
        churn_by_subscription.columns = ['subscription_type', 'total_customers', 'churned_customers', 'churn_rate']
        
        axes[0,1].bar(churn_by_subscription['subscription_type'], churn_by_subscription['churn_rate'])
        axes[0,1].set_title('Churn Rate by Subscription Type')
        axes[0,1].set_ylabel('Churn Rate')
        
        # Monthly charges distribution
        customer_churn = customers.merge(churn_labels, on='customer_id')
        axes[1,0].hist(customer_churn[customer_churn['is_churned']==False]['monthly_charges'], 
                      alpha=0.7, label='Retained', bins=30)
        axes[1,0].hist(customer_churn[customer_churn['is_churned']==True]['monthly_charges'], 
                      alpha=0.7, label='Churned', bins=30)
        axes[1,0].set_title('Monthly Charges Distribution')
        axes[1,0].set_xlabel('Monthly Charges ($)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # Support tickets vs churn
        support_churn = support_tickets.merge(churn_labels, on='customer_id')
        support_summary = support_churn.groupby(['customer_id', 'is_churned']).size().reset_index(name='ticket_count')
        
        churned_tickets = support_summary[support_summary['is_churned']==True]['ticket_count']
        retained_tickets = support_summary[support_summary['is_churned']==False]['ticket_count']
        
        axes[1,1].hist(retained_tickets, alpha=0.7, label='Retained', bins=20)
        axes[1,1].hist(churned_tickets, alpha=0.7, label='Churned', bins=20)
        axes[1,1].set_title('Support Tickets vs Churn')
        axes[1,1].set_xlabel('Number of Support Tickets')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Transaction Patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convert dates
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        
        # Monthly transaction volume
        monthly_transactions = transactions.groupby(transactions['transaction_date'].dt.to_period('M')).size()
        axes[0,0].plot(monthly_transactions.index.astype(str), monthly_transactions.values)
        axes[0,0].set_title('Monthly Transaction Volume')
        axes[0,0].set_ylabel('Number of Transactions')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Transaction failure rate by month
        monthly_failures = transactions.groupby(transactions['transaction_date'].dt.to_period('M'))['status'].apply(
            lambda x: (x == 'Failed').sum() / len(x)
        )
        axes[0,1].plot(monthly_failures.index.astype(str), monthly_failures.values)
        axes[0,1].set_title('Monthly Transaction Failure Rate')
        axes[0,1].set_ylabel('Failure Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Transaction amount distribution
        completed_transactions = transactions[transactions['status'] == 'Completed']
        axes[1,0].hist(completed_transactions['amount'], bins=50)
        axes[1,0].set_title('Transaction Amount Distribution')
        axes[1,0].set_xlabel('Amount ($)')
        axes[1,0].set_ylabel('Frequency')
        
        # Payment method distribution
        payment_counts = transactions['payment_method'].value_counts()
        axes[1,1].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Payment Method Distribution')
        
        plt.tight_layout()
        plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/transaction_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Support Ticket Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Support ticket categories
        category_counts = support_tickets['issue_category'].value_counts()
        axes[0,0].bar(category_counts.index, category_counts.values)
        axes[0,0].set_title('Support Ticket Categories')
        axes[0,0].set_ylabel('Number of Tickets')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Average satisfaction by category
        satisfaction_by_category = support_tickets.groupby('issue_category')['satisfaction_score'].mean().dropna()
        axes[0,1].bar(satisfaction_by_category.index, satisfaction_by_category.values)
        axes[0,1].set_title('Average Satisfaction by Category')
        axes[0,1].set_ylabel('Average Satisfaction Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Resolution time distribution
        axes[1,0].hist(support_tickets['resolution_time_hours'].dropna(), bins=30)
        axes[1,0].set_title('Resolution Time Distribution')
        axes[1,0].set_xlabel('Resolution Time (Hours)')
        axes[1,0].set_ylabel('Frequency')
        
        # Priority distribution
        priority_counts = support_tickets['priority'].value_counts()
        axes[1,1].pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Ticket Priority Distribution')
        
        plt.tight_layout()
        plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/support_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("EDA visualizations saved to figures/ directory")
    
    def engineer_features(self, customers, transactions, support_tickets, churn_labels):
        """Advanced feature engineering with business logic"""
        print("Engineering advanced features...")
        
        # Convert date columns
        customers['account_creation_date'] = pd.to_datetime(customers['account_creation_date'])
        customers['last_login_date'] = pd.to_datetime(customers['last_login_date'])
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        support_tickets['created_date'] = pd.to_datetime(support_tickets['created_date'])
        
        # Reference date
        reference_date = pd.to_datetime('2024-01-01')
        
        # Customer lifecycle features
        customers['account_age_days'] = (reference_date - customers['account_creation_date']).dt.days
        customers['days_since_last_login'] = (reference_date - customers['last_login_date']).dt.days
        customers['never_logged_in'] = customers['last_login_date'].isna().astype(int)
        customers['age'] = 2024 - pd.to_datetime(customers['date_of_birth']).dt.year
        
        # Transaction features
        transaction_features = transactions.groupby('customer_id').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean', 'std'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        transaction_features.columns = ['customer_id', 'total_transactions', 'total_amount', 
                                      'avg_amount', 'std_amount', 'first_transaction', 'last_transaction']
        
        # Failed transactions
        failed_transactions = transactions[transactions['status'] == 'Failed'].groupby('customer_id').size()
        transaction_features['failed_transactions'] = transaction_features['customer_id'].map(failed_transactions).fillna(0)
        transaction_features['failure_rate'] = transaction_features['failed_transactions'] / transaction_features['total_transactions']
        
        # Days since last transaction
        transaction_features['days_since_last_transaction'] = (reference_date - transaction_features['last_transaction']).dt.days
        
        # Support features
        support_features = support_tickets.groupby('customer_id').agg({
            'ticket_id': 'count',
            'satisfaction_score': ['mean', 'min'],
            'resolution_time_hours': 'mean'
        }).reset_index()
        
        support_features.columns = ['customer_id', 'total_tickets', 'avg_satisfaction', 
                                   'min_satisfaction', 'avg_resolution_time']
        
        # Support categories
        for category in ['Billing', 'Technical', 'Cancellation']:
            category_tickets = support_tickets[support_tickets['issue_category'] == category]
            category_counts = category_tickets.groupby('customer_id').size()
            support_features[f'{category.lower()}_tickets'] = support_features['customer_id'].map(category_counts).fillna(0)
        
        # Merge features
        features = customers.merge(transaction_features, on='customer_id', how='left')
        features = features.merge(support_features, on='customer_id', how='left')
        features = features.merge(churn_labels[['customer_id', 'is_churned']], on='customer_id', how='left')
        
        # Fill missing values
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features[numeric_columns] = features[numeric_columns].fillna(0)
        
        # Engineered features
        features['customer_value_score'] = (
            features['total_amount'] * 0.4 +
            features['total_transactions'] * 0.3 +
            features['account_age_days'] * 0.2 +
            np.where(features['days_since_last_transaction'] > 0, 
                    (365 - features['days_since_last_transaction']) * 0.1, 0)
        )
        
        features['risk_score'] = (
            features['failure_rate'] * 0.3 +
            features['cancellation_tickets'] * 0.4 +
            features['billing_tickets'] * 0.2 +
            np.where(features['avg_satisfaction'] > 0, (5 - features['avg_satisfaction']) * 0.1, 0)
        )
        
        print(f"Feature engineering completed. Final dataset shape: {features.shape}")
        
        return features
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        
        feature_columns = [
            'monthly_charges', 'total_charges', 'contract_length', 'account_age_days',
            'days_since_last_login', 'never_logged_in', 'age',
            'total_transactions', 'total_amount', 'avg_amount', 'std_amount',
            'failure_rate', 'days_since_last_transaction', 'failed_transactions',
            'total_tickets', 'avg_satisfaction', 'min_satisfaction',
            'avg_resolution_time', 'billing_tickets', 'technical_tickets', 'cancellation_tickets',
            'customer_value_score', 'risk_score',
            'subscription_type', 'province', 'payment_method', 'paperless_billing', 'auto_pay'
        ]
        
        # Handle categorical and boolean columns
        categorical_columns = ['subscription_type', 'province', 'payment_method']
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        bool_columns = ['paperless_billing', 'auto_pay']
        for col in bool_columns:
            df[col] = df[col].astype(int)
        
        X = df[feature_columns].copy()
        y = df['is_churned'].copy()
        
        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        self.feature_names = X_encoded.columns.tolist()
        
        return X_encoded, y
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        print("Training multiple models...")
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=4),
            'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=4, eval_metric='logloss')
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
        
        self.models = results
        return results
    
    def create_model_comparison_plots(self, results, y_test):
        """Create comprehensive model comparison visualizations"""
        print("Creating model comparison visualizations...")
        
        # 1. Performance Metrics Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        # Bar plot for each metric
        for i, metric in enumerate(metrics[:4]):
            ax = axes[i//2, i%2]
            values = [results[model][metric] for model in model_names]
            bars = ax.bar(model_names, values)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {result["auc_score"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
            plt.plot(recall, precision, label=f'{model_name} (F1 = {result["f1_score"]:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Confusion Matrices
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, (model_name, result) in enumerate(results.items()):
            ax = axes[i//2, i%2]
            cm = confusion_matrix(y_test, result['predictions'])
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Retained', 'Churned'],
                       yticklabels=['Retained', 'Churned'])
            ax.set_title(f'{model_name} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Model comparison visualizations saved to figures/ directory")
    
    def create_feature_importance_plot(self, model, model_name):
        """Create feature importance visualization"""
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = importance.head(20)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importances - {model_name}')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{top_features.iloc[i]["importance"]:.3f}', 
                        va='center', ha='left')
            
            plt.tight_layout()
            plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Feature importance plot saved for {model_name}")
            return importance
        
        return None
    
    def create_business_insights_plots(self, features):
        """Create business-focused visualizations"""
        print("Creating business insights visualizations...")
        
        # 1. Customer Segmentation
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Churn rate by customer value score quartiles
        features['value_quartile'] = pd.qcut(features['customer_value_score'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
        churn_by_value = features.groupby('value_quartile')['is_churned'].mean()
        
        axes[0,0].bar(churn_by_value.index, churn_by_value.values)
        axes[0,0].set_title('Churn Rate by Customer Value')
        axes[0,0].set_ylabel('Churn Rate')
        
        # Churn rate by risk score quartiles
        features['risk_quartile'] = pd.qcut(features['risk_score'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
        churn_by_risk = features.groupby('risk_quartile')['is_churned'].mean()
        
        axes[0,1].bar(churn_by_risk.index, churn_by_risk.values)
        axes[0,1].set_title('Churn Rate by Risk Score')
        axes[0,1].set_ylabel('Churn Rate')
        
        # Average monthly charges by churn status
        churn_charges = features.groupby('is_churned')['monthly_charges'].mean()
        axes[1,0].bar(['Retained', 'Churned'], churn_charges.values)
        axes[1,0].set_title('Average Monthly Charges by Churn Status')
        axes[1,0].set_ylabel('Monthly Charges ($)')
        
        # Support tickets vs churn
        support_churn = features.groupby('is_churned')['total_tickets'].mean()
        axes[1,1].bar(['Retained', 'Churned'], support_churn.values)
        axes[1,1].set_title('Average Support Tickets by Churn Status')
        axes[1,1].set_ylabel('Average Support Tickets')
        
        plt.tight_layout()
        plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/business_insights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        
        # Select key numerical features
        key_features = ['monthly_charges', 'total_amount', 'failure_rate', 'total_tickets',
                       'avg_satisfaction', 'customer_value_score', 'risk_score', 'is_churned']
        
        correlation_matrix = features[key_features].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Business insights visualizations saved to figures/ directory")
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline with visualizations"""
        print("Starting Customer Churn Prediction ML Pipeline with Visualizations")
        print("=" * 70)
        
        # Load data
        customers, transactions, support_tickets, churn_labels = self.load_data()
        
        # Create EDA visualizations
        self.create_eda_visualizations(customers, transactions, support_tickets, churn_labels)
        
        # Feature engineering
        features = self.engineer_features(customers, transactions, support_tickets, churn_labels)
        
        # Create business insights
        self.create_business_insights_plots(features)
        
        # Prepare features
        X, y = self.prepare_features(features)
        
        print(f"\nFinal feature set shape: {X.shape}")
        print(f"Target distribution: {y.value_counts(normalize=True)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test)
        
        # Create model comparison plots
        self.create_model_comparison_plots(results, y_test)
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
        # Feature importance
        feature_importance = self.create_feature_importance_plot(best_model, best_model_name)
        
        # Save model and metadata
        self.save_model(best_model, 'churn_prediction_model.pkl')
        joblib.dump(self.feature_names, '/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/models/feature_names.pkl')
        
        # Save preprocessing info
        preprocessing_info = {
            'categorical_columns': ['subscription_type', 'province', 'payment_method'],
            'boolean_columns': ['paperless_billing', 'auto_pay'],
            'feature_columns': [
                'monthly_charges', 'total_charges', 'contract_length', 'account_age_days',
                'days_since_last_login', 'never_logged_in', 'age',
                'total_transactions', 'total_amount', 'avg_amount', 'std_amount',
                'failure_rate', 'days_since_last_transaction', 'failed_transactions',
                'total_tickets', 'avg_satisfaction', 'min_satisfaction',
                'avg_resolution_time', 'billing_tickets', 'technical_tickets', 'cancellation_tickets',
                'customer_value_score', 'risk_score',
                'subscription_type', 'province', 'payment_method', 'paperless_billing', 'auto_pay'
            ]
        }
        
        joblib.dump(preprocessing_info, '/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/models/preprocessing_info.pkl')
        
        print("\n" + "=" * 70)
        print("ML Pipeline with Visualizations completed successfully!")
        print(f"Final model AUC: {results[best_model_name]['auc_score']:.4f}")
        print(f"Final model F1-Score: {results[best_model_name]['f1_score']:.4f}")
        print("\nGenerated visualizations:")
        print("- EDA Analysis: figures/eda_analysis.png")
        print("- Transaction Patterns: figures/transaction_patterns.png")
        print("- Support Analysis: figures/support_analysis.png")
        print("- Model Comparison: figures/model_comparison.png")
        print("- ROC Curves: figures/roc_curves.png")
        print("- Precision-Recall Curves: figures/precision_recall_curves.png")
        print("- Confusion Matrices: figures/confusion_matrices.png")
        print("- Feature Importance: figures/feature_importance.png")
        print("- Business Insights: figures/business_insights.png")
        print("- Correlation Heatmap: figures/correlation_heatmap.png")
        
        return best_model, results, feature_importance
    
    def save_model(self, model, filename):
        """Save trained model"""
        model_path = f'/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/models/{filename}'
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

def main():
    """Main function to run the visual ML pipeline"""
    pipeline = ChurnPredictionPipelineVisual()
    model, results, feature_importance = pipeline.run_complete_pipeline()
    
    return model, results, feature_importance

if __name__ == "__main__":
    model, results, feature_importance = main()