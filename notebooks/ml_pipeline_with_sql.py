"""
Customer Churn Prediction ML Pipeline with SQL Integration
Advanced ML pipeline using PostgreSQL for feature engineering
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path so we can import from data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database_setup import DatabaseManager

class ChurnPredictionPipeline:
    """ML Pipeline with SQL-based feature engineering"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_column = 'is_churned'
        
    def load_data_from_sql(self):
        """Load data from PostgreSQL using advanced SQL features"""
        print("Loading data from PostgreSQL...")
        
        # Test database connection
        if not self.db_manager.test_connection():
            raise Exception("Cannot connect to database. Please ensure PostgreSQL is running.")
        
        # Get ML features using SQL
        df = self.db_manager.get_ml_features()
        print(f"Loaded {len(df)} customers with {len(df.columns)} features")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for ML training"""
        print("Preprocessing data...")
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Separate features and target
        X = df.drop([self.target_column, 'customer_id'], axis=1)
        y = df[self.target_column]
        
        # Encode categorical variables
        categorical_columns = ['subscription_type', 'province', 'payment_method']
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("Training models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        
        # Train and evaluate models
        model_results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Store results
            model_results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Train Accuracy: {train_score:.3f}")
            print(f"  Test Accuracy: {test_score:.3f}")
            print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Select best model based on CV score
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['cv_score'])
        self.model = model_results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Cross-validation AUC: {model_results[best_model_name]['cv_score']:.3f}")
        
        return X_train, X_test, y_train, y_test, model_results
    
    def create_visualizations(self, X_train, X_test, y_train, y_test, model_results):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Create figures directory
        os.makedirs('figures', exist_ok=True)
        
        # 1. Model Comparison
        plt.figure(figsize=(12, 6))
        
        model_names = list(model_results.keys())
        train_scores = [model_results[name]['train_score'] for name in model_names]
        test_scores = [model_results[name]['test_score'] for name in model_names]
        cv_scores = [model_results[name]['cv_score'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.bar(x - width, train_scores, width, label='Train Score', alpha=0.8)
        plt.bar(x, test_scores, width, label='Test Score', alpha=0.8)
        plt.bar(x + width, cv_scores, width, label='CV AUC Score', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig('figures/sql_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            
            # Get feature importance
            importance = self.model.feature_importances_
            feature_names = self.feature_names
            
            # Sort by importance
            indices = np.argsort(importance)[::-1]
            
            # Plot top 20 features
            top_n = min(20, len(feature_names))
            plt.title('Feature Importance (Top 20)')
            plt.bar(range(top_n), importance[indices[:top_n]])
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45)
            plt.tight_layout()
            plt.savefig('figures/sql_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. ROC Curve
        plt.figure(figsize=(10, 8))
        
        for name, results in model_results.items():
            model = results['model']
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - SQL-based Features')
        plt.legend()
        plt.tight_layout()
        plt.savefig('figures/sql_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Confusion Matrix
        plt.figure(figsize=(8, 6))
        
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix - SQL-based Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('figures/sql_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to figures/ directory")
    
    def save_model(self):
        """Save the trained model and preprocessing components"""
        print("Saving model and preprocessing components...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model
        joblib.dump(self.model, 'models/churn_prediction_model_sql.pkl')
        
        # Save preprocessing components
        joblib.dump(self.scaler, 'models/scaler_sql.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders_sql.pkl')
        joblib.dump(self.feature_names, 'models/feature_names_sql.pkl')
        
        # Save preprocessing info
        preprocessing_info = {
            'feature_names': self.feature_names,
            'label_encoders': list(self.label_encoders.keys()),
            'target_column': self.target_column,
            'model_type': type(self.model).__name__,
            'features_count': len(self.feature_names)
        }
        joblib.dump(preprocessing_info, 'models/preprocessing_info_sql.pkl')
        
        print("Model saved successfully!")
    
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        print("="*60)
        print("Customer Churn Prediction Pipeline with SQL Integration")
        print("="*60)
        
        try:
            # Load data from SQL
            df = self.load_data_from_sql()
            
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            # Train models
            X_train, X_test, y_train, y_test, model_results = self.train_models(X, y)
            
            # Create visualizations
            self.create_visualizations(X_train, X_test, y_train, y_test, model_results)
            
            # Save model
            self.save_model()
            
            # Print final results
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"✓ Data loaded from PostgreSQL: {len(df)} customers")
            print(f"✓ Features engineered using SQL: {len(self.feature_names)} features")
            print(f"✓ Best model: {type(self.model).__name__}")
            print(f"✓ Model performance: {model_results[max(model_results.keys(), key=lambda x: model_results[x]['cv_score'])]['cv_score']:.3f} AUC")
            print(f"✓ Visualizations saved to figures/")
            print(f"✓ Model saved to models/")
            
            return True
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            return False

def main():
    """Main function to run the pipeline"""
    
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline()
    
    # Run pipeline
    success = pipeline.run_pipeline()
    
    if success:
        print("\n🎉 SQL-integrated ML pipeline completed successfully!")
        print("Your model is now trained using PostgreSQL feature engineering!")
    else:
        print("\n❌ Pipeline failed. Please check the errors above.")

if __name__ == "__main__":
    main()