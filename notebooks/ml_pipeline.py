"""
Customer Churn Prediction ML Pipeline
Advanced ML engineering pipeline for predictive analytics

Features:
- Advanced feature engineering with domain expertise
- Multiple model comparison (scikit-learn, XGBoost, LightGBM)
- Comprehensive model evaluation and interpretation
- Production-ready model serialization
- Automated hyperparameter tuning
- Feature importance analysis
- Cross-validation and robust evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class ChurnPredictionPipeline:
    """
    Comprehensive ML pipeline for customer churn prediction
    """
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.best_model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_performance = {}
        
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
    
    def engineer_features(self, customers, transactions, support_tickets, churn_labels):
        """
        Advanced feature engineering showcasing domain expertise
        """
        print("Engineering advanced features...")
        
        # Convert date columns
        customers['account_creation_date'] = pd.to_datetime(customers['account_creation_date'])
        customers['last_login_date'] = pd.to_datetime(customers['last_login_date'])
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        support_tickets['created_date'] = pd.to_datetime(support_tickets['created_date'])
        
        # Reference date for calculations
        reference_date = pd.to_datetime('2024-01-01')
        
        # 1. Customer Lifecycle Features
        customers['account_age_days'] = (reference_date - customers['account_creation_date']).dt.days
        customers['days_since_last_login'] = (reference_date - customers['last_login_date']).dt.days
        customers['never_logged_in'] = customers['last_login_date'].isna().astype(int)
        customers['age'] = 2024 - pd.to_datetime(customers['date_of_birth']).dt.year
        
        # 2. Advanced Transaction Features
        transaction_features = transactions.groupby('customer_id').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean', 'std', 'min', 'max'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        transaction_features.columns = ['customer_id', 'total_transactions', 'total_amount', 
                                      'avg_amount', 'std_amount', 'min_amount', 'max_amount',
                                      'first_transaction', 'last_transaction']
        
        # Transaction timing features
        transaction_features['days_since_first_transaction'] = (reference_date - transaction_features['first_transaction']).dt.days
        transaction_features['days_since_last_transaction'] = (reference_date - transaction_features['last_transaction']).dt.days
        
        # Transaction failure rates
        failed_transactions = transactions[transactions['status'] == 'Failed'].groupby('customer_id').size()
        transaction_features['failed_transactions'] = transaction_features['customer_id'].map(failed_transactions).fillna(0)
        transaction_features['failure_rate'] = transaction_features['failed_transactions'] / transaction_features['total_transactions']
        
        # Recency features (transactions in last 30, 60, 90 days)
        for days in [30, 60, 90]:
            recent_transactions = transactions[transactions['transaction_date'] >= reference_date - timedelta(days=days)]
            recent_counts = recent_transactions.groupby('customer_id').size()
            transaction_features[f'transactions_last_{days}d'] = transaction_features['customer_id'].map(recent_counts).fillna(0)
        
        # 3. Support Ticket Features
        support_features = support_tickets.groupby('customer_id').agg({
            'ticket_id': 'count',
            'satisfaction_score': ['mean', 'min', 'count'],
            'resolution_time_hours': ['mean', 'max'],
            'created_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        support_features.columns = ['customer_id', 'total_tickets', 'avg_satisfaction', 
                                   'min_satisfaction', 'satisfaction_responses', 
                                   'avg_resolution_time', 'max_resolution_time',
                                   'first_ticket', 'last_ticket']
        
        # Support ticket categories
        for category in ['Billing', 'Technical', 'Account', 'Cancellation', 'General']:
            category_tickets = support_tickets[support_tickets['issue_category'] == category]
            category_counts = category_tickets.groupby('customer_id').size()
            support_features[f'{category.lower()}_tickets'] = support_features['customer_id'].map(category_counts).fillna(0)
        
        # Support timing features
        support_features['days_since_first_ticket'] = (reference_date - support_features['first_ticket']).dt.days
        support_features['days_since_last_ticket'] = (reference_date - support_features['last_ticket']).dt.days
        
        # High priority and escalated tickets
        high_priority = support_tickets[support_tickets['priority'].isin(['High', 'Critical'])]
        escalated = support_tickets[support_tickets['escalated'] == True]
        
        high_priority_counts = high_priority.groupby('customer_id').size()
        escalated_counts = escalated.groupby('customer_id').size()
        
        support_features['high_priority_tickets'] = support_features['customer_id'].map(high_priority_counts).fillna(0)
        support_features['escalated_tickets'] = support_features['customer_id'].map(escalated_counts).fillna(0)
        
        # 4. Merge all features
        features = customers.merge(transaction_features, on='customer_id', how='left')
        features = features.merge(support_features, on='customer_id', how='left')
        features = features.merge(churn_labels[['customer_id', 'is_churned']], on='customer_id', how='left')
        
        # Fill missing values
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features[numeric_columns] = features[numeric_columns].fillna(0)
        
        # 5. Advanced Engineered Features
        
        # Customer Value Score
        features['customer_value_score'] = (
            features['total_amount'] * 0.4 +
            features['total_transactions'] * 0.3 +
            features['account_age_days'] * 0.2 +
            (365 - features['days_since_last_transaction']) * 0.1
        )
        
        # Engagement Score
        features['engagement_score'] = (
            (365 - features['days_since_last_login']) * 0.3 +
            features['transactions_last_30d'] * 0.4 +
            (5 - features['avg_satisfaction']) * 0.3  # Higher satisfaction = higher engagement
        )
        
        # Risk Score
        features['risk_score'] = (
            features['failure_rate'] * 0.3 +
            features['cancellation_tickets'] * 0.4 +
            features['billing_tickets'] * 0.2 +
            features['escalated_tickets'] * 0.1
        )
        
        # Transaction Consistency
        features['transaction_consistency'] = 1 / (1 + features['std_amount'])
        
        # Support Dependency
        features['support_dependency'] = features['total_tickets'] / (features['account_age_days'] + 1)
        
        print(f"Feature engineering completed. Final dataset shape: {features.shape}")
        
        return features
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        
        # Select features for modeling
        feature_columns = [
            # Basic customer info
            'monthly_charges', 'total_charges', 'contract_length', 'account_age_days',
            'days_since_last_login', 'never_logged_in', 'age',
            
            # Transaction features
            'total_transactions', 'total_amount', 'avg_amount', 'std_amount',
            'failure_rate', 'days_since_last_transaction', 'failed_transactions',
            'transactions_last_30d', 'transactions_last_60d', 'transactions_last_90d',
            
            # Support features
            'total_tickets', 'avg_satisfaction', 'min_satisfaction',
            'avg_resolution_time', 'billing_tickets', 'technical_tickets',
            'cancellation_tickets', 'high_priority_tickets', 'escalated_tickets',
            
            # Engineered features
            'customer_value_score', 'engagement_score', 'risk_score',
            'transaction_consistency', 'support_dependency',
            
            # Categorical features
            'subscription_type', 'province', 'payment_method', 'paperless_billing', 'auto_pay'
        ]
        
        # Handle missing categorical values
        categorical_columns = ['subscription_type', 'province', 'payment_method']
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        # Convert boolean columns to int
        bool_columns = ['paperless_billing', 'auto_pay']
        for col in bool_columns:
            df[col] = df[col].astype(int)
        
        X = df[feature_columns].copy()
        y = df['is_churned'].copy()
        
        # One-hot encode categorical variables
        categorical_features = ['subscription_type', 'province', 'payment_method']
        X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        self.feature_names = X_encoded.columns.tolist()
        
        return X_encoded, y
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        print("Training multiple models...")
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
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
            print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = results
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='xgboost'):
        """Perform hyperparameter tuning on the best model"""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        elif model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            model = GradientBoostingClassifier(random_state=42)
        
        elif model_name == 'lightgbm':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_name}")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC Score: {auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def get_feature_importance(self, model):
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 20 Most Important Features:")
            print(importance.head(20))
            
            return importance
        else:
            print("Model does not have feature_importances_ attribute")
            return None
    
    def save_model(self, model, filename):
        """Save trained model"""
        model_path = f'/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/models/{filename}'
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("Starting Customer Churn Prediction ML Pipeline")
        print("=" * 60)
        
        # 1. Load data
        customers, transactions, support_tickets, churn_labels = self.load_data()
        
        # 2. Feature engineering
        features = self.engineer_features(customers, transactions, support_tickets, churn_labels)
        
        # 3. Prepare features
        X, y = self.prepare_features(features)
        
        print(f"\nFinal feature set shape: {X.shape}")
        print(f"Target distribution: {y.value_counts(normalize=True)}")
        
        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # 5. Train multiple models
        results = self.train_models(X_train, X_test, y_train, y_test)
        
        # 6. Select best model based on AUC
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        print(f"\nBest model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
        # 7. Hyperparameter tuning on best model
        tuned_model = self.hyperparameter_tuning(X_train, y_train, best_model_name)
        
        # 8. Final evaluation
        final_results = self.evaluate_model(tuned_model, X_test, y_test, "Tuned Model")
        
        # 9. Feature importance
        feature_importance = self.get_feature_importance(tuned_model)
        
        # 10. Save model
        self.save_model(tuned_model, 'churn_prediction_model.pkl')
        
        # Save feature names for later use
        joblib.dump(self.feature_names, '/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/models/feature_names.pkl')
        
        print("\n" + "=" * 60)
        print("ML Pipeline completed successfully!")
        print(f"Final model AUC: {final_results['auc_score']:.4f}")
        print(f"Final model F1-Score: {final_results['f1_score']:.4f}")
        
        return tuned_model, final_results, feature_importance

def main():
    """Main function to run the ML pipeline"""
    pipeline = ChurnPredictionPipeline()
    model, results, feature_importance = pipeline.run_complete_pipeline()
    
    return model, results, feature_importance

if __name__ == "__main__":
    model, results, feature_importance = main()