"""
SHAP Explainer for Customer Churn Prediction
Provides model interpretability using SHAP values for individual predictions
"""

import shap
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class ChurnExplainer:
    """
    SHAP-based explainer for customer churn predictions
    """
    
    def __init__(self, model_path: str, feature_names_path: str, preprocessing_info_path: str):
        """
        Initialize the explainer with trained model and metadata
        
        Args:
            model_path: Path to the trained model
            feature_names_path: Path to feature names list
            preprocessing_info_path: Path to preprocessing information
        """
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)
        self.preprocessing_info = joblib.load(preprocessing_info_path)
        
        # Initialize SHAP explainer
        self.explainer = None
        self.shap_values = None
        self.setup_explainer()
        
    def setup_explainer(self):
        try:
            # Check if model is tree-based (RandomForest, XGBoost, etc.)
            if hasattr(self.model, 'estimators_') or hasattr(self.model, 'tree_'):
                self.explainer = shap.TreeExplainer(self.model)
                print("Using TreeExplainer for tree-based model")
            else:
                # For linear models (LogisticRegression, SVM, etc.)
                import sklearn.datasets
                data, _ = sklearn.datasets.make_classification(n_samples=100, n_features=len(self.feature_names), random_state=42)
                self.explainer = shap.LinearExplainer(self.model, data)
                print("Using LinearExplainer for linear model")
        except Exception as e:
            print(f"Error setting up explainer: {e}")
            # Fallback to KernelExplainer with zero background
            background = np.zeros((1, len(self.feature_names)))
            self.explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x)[:, 1], 
                background
            )
            print("Using KernelExplainer as fallback")
    
    def preprocess_customer_data(self, customer_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess customer data to match model input format
        
        Args:
            customer_data: Dictionary containing customer information
            
        Returns:
            Preprocessed DataFrame ready for model input
        """
        # Create DataFrame from customer data
        df = pd.DataFrame([customer_data])
        
        # Handle categorical columns
        categorical_columns = self.preprocessing_info['categorical_columns']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Handle boolean columns
        boolean_columns = self.preprocessing_info['boolean_columns']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Ensure all required columns are present
        feature_columns = self.preprocessing_info['feature_columns']
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select only the required columns
        df = df[feature_columns]
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        # Ensure all model features are present
        for feature in self.feature_names:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Reorder columns to match model training
        df_encoded = df_encoded[self.feature_names]
        
        return df_encoded
    
    def explain_prediction(self, customer_data: Dict[str, Any], 
                         top_n: int = 10) -> Dict[str, Any]:
        """
        Explain a single customer's churn prediction
        
        Args:
            customer_data: Dictionary containing customer information
            top_n: Number of top features to include in explanation
            
        Returns:
            Dictionary containing prediction and explanation
        """
        # Preprocess data
        processed_data = self.preprocess_customer_data(customer_data)
        
        # Make prediction
        prediction_proba = self.model.predict_proba(processed_data)[0]
        prediction = self.model.predict(processed_data)[0]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(processed_data.values)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For binary classification, use positive class (churn) SHAP values
            shap_values_churn = shap_values[1][0]  # Index 1 for churn class
        else:
            # For single array of SHAP values
            shap_values_churn = shap_values[0]
        
        # Ensure shap_values_churn is 1-dimensional
        if len(shap_values_churn.shape) > 1:
            shap_values_churn = shap_values_churn.flatten()
        
        # Ensure all arrays have the same length
        feature_names_len = len(self.feature_names)
        shap_values_len = len(shap_values_churn)
        feature_values_len = len(processed_data.iloc[0].values)
        
        # Debug information
        print(f"Feature names length: {feature_names_len}")
        print(f"SHAP values length: {shap_values_len}")
        print(f"Feature values length: {feature_values_len}")
        
        # Ensure consistent lengths by taking the minimum
        min_length = min(feature_names_len, shap_values_len, feature_values_len)
        
        # Create feature importance dataframe with consistent lengths
        feature_importance = pd.DataFrame({
            'feature': self.feature_names[:min_length],
            'shap_value': shap_values_churn[:min_length],
            'feature_value': processed_data.iloc[0].values[:min_length]
        })
        
        # Sort by absolute SHAP value
        feature_importance['abs_shap_value'] = np.abs(feature_importance['shap_value'])
        feature_importance = feature_importance.sort_values('abs_shap_value', ascending=False)
        
        # Get top features
        top_features = feature_importance.head(top_n)
        
        # Create explanation
        explanation = {
            'customer_id': customer_data.get('customer_id', 'Unknown'),
            'prediction': {
                'churn_probability': float(prediction_proba[1]),
                'will_churn': bool(prediction),
                'confidence': float(max(prediction_proba))
            },
            'top_factors': [],
            'summary': self._generate_summary(top_features, prediction_proba[1])
        }
        
        # Add top contributing factors
        for _, row in top_features.iterrows():
            factor = {
                'feature': row['feature'],
                'impact': 'Increases churn risk' if row['shap_value'] > 0 else 'Decreases churn risk',
                'shap_value': float(row['shap_value']),
                'feature_value': float(row['feature_value']),
                'importance_rank': int(row.name + 1)
            }
            explanation['top_factors'].append(factor)
        
        return explanation
    
    def _generate_summary(self, top_features: pd.DataFrame, churn_probability: float) -> str:
        """Generate a human-readable summary of the prediction"""
        
        risk_level = "High" if churn_probability > 0.7 else "Medium" if churn_probability > 0.3 else "Low"
        
        # Get top positive and negative factors
        positive_factors = top_features[top_features['shap_value'] > 0]
        negative_factors = top_features[top_features['shap_value'] < 0]
        
        summary = f"Customer has {risk_level.lower()} churn risk ({churn_probability:.1%} probability). "
        
        if len(positive_factors) > 0:
            top_risk_factor = positive_factors.iloc[0]['feature']
            summary += f"Main risk factor: {top_risk_factor.replace('_', ' ')}. "
        
        if len(negative_factors) > 0:
            top_retention_factor = negative_factors.iloc[0]['feature']
            summary += f"Main retention factor: {top_retention_factor.replace('_', ' ')}. "
        
        # Add actionable insights
        if churn_probability > 0.5:
            summary += "Recommend immediate intervention."
        elif churn_probability > 0.3:
            summary += "Consider proactive engagement."
        else:
            summary += "Customer appears stable."
        
        return summary
    
    def create_explanation_plot(self, customer_data: Dict[str, Any], 
                              save_path: Optional[str] = None) -> None:
        """
        Create a waterfall plot showing feature contributions
        
        Args:
            customer_data: Dictionary containing customer information
            save_path: Optional path to save the plot
        """
        # Preprocess data
        processed_data = self.preprocess_customer_data(customer_data)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(processed_data.values)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values_churn = shap_values[1][0]  # Index 1 for churn class
        else:
            shap_values_churn = shap_values[0]
        
        # Ensure shap_values_churn is 1-dimensional
        if len(shap_values_churn.shape) > 1:
            shap_values_churn = shap_values_churn.flatten()
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        
        # Get top 10 features by absolute SHAP value
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values_churn
        })
        
        feature_importance['abs_shap_value'] = np.abs(feature_importance['shap_value'])
        top_features = feature_importance.nlargest(10, 'abs_shap_value')
        
        # Create horizontal bar plot
        colors = ['red' if x > 0 else 'green' for x in top_features['shap_value']]
        
        plt.barh(range(len(top_features)), top_features['shap_value'], color=colors)
        plt.yticks(range(len(top_features)), 
                  [f.replace('_', ' ').title() for f in top_features['feature']])
        plt.xlabel('SHAP Value (Impact on Churn Probability)')
        plt.title(f'Churn Prediction Explanation for Customer {customer_data.get("customer_id", "Unknown")}')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend
        plt.text(0.02, 0.98, 'Red: Increases churn risk\nGreen: Decreases churn risk', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation plot saved to {save_path}")
        
        plt.show()
    
    def explain_batch(self, customer_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Explain predictions for a batch of customers
        
        Args:
            customer_data_list: List of customer data dictionaries
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for customer_data in customer_data_list:
            try:
                explanation = self.explain_prediction(customer_data)
                explanations.append(explanation)
            except Exception as e:
                print(f"Error explaining customer {customer_data.get('customer_id', 'Unknown')}: {e}")
                explanations.append({
                    'customer_id': customer_data.get('customer_id', 'Unknown'),
                    'error': str(e)
                })
        
        return explanations
    
    def get_global_feature_importance(self, sample_data: pd.DataFrame, 
                                    sample_size: int = 1000) -> pd.DataFrame:
        """
        Calculate global feature importance using SHAP values
        
        Args:
            sample_data: Sample of customer data
            sample_size: Number of samples to use for calculation
            
        Returns:
            DataFrame with global feature importance
        """
        # Sample data if needed
        if len(sample_data) > sample_size:
            sample_data = sample_data.sample(n=sample_size, random_state=42)
        
        # Calculate SHAP values for sample
        shap_values = self.explainer.shap_values(sample_data.values)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values_churn = shap_values[1]  # Index 1 for churn class
        else:
            shap_values_churn = shap_values
        
        # Ensure proper dimensionality for batch processing
        if len(shap_values_churn.shape) > 2:
            shap_values_churn = shap_values_churn.reshape(shap_values_churn.shape[0], -1)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values_churn), axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap_value': mean_abs_shap
        }).sort_values('mean_abs_shap_value', ascending=False)
        
        return importance_df
    
    def save_explainer(self, save_path: str):
        """Save the explainer for later use"""
        explainer_data = {
            'explainer': self.explainer,
            'feature_names': self.feature_names,
            'preprocessing_info': self.preprocessing_info
        }
        joblib.dump(explainer_data, save_path)
        print(f"Explainer saved to {save_path}")

def create_sample_explanation():
    """Create a sample explanation for testing"""
    
    # Sample customer data
    sample_customer = {
        'customer_id': 'CUST_001234',
        'monthly_charges': 75.50,
        'total_charges': 1200.00,
        'contract_length': 12,
        'account_age_days': 365,
        'days_since_last_login': 5,
        'never_logged_in': 0,
        'age': 35,
        'total_transactions': 24,
        'total_amount': 1800.00,
        'avg_amount': 75.00,
        'std_amount': 5.50,
        'failure_rate': 0.08,
        'days_since_last_transaction': 30,
        'failed_transactions': 2,
        'total_tickets': 3,
        'avg_satisfaction': 3.5,
        'min_satisfaction': 2.0,
        'avg_resolution_time': 24.5,
        'billing_tickets': 2,
        'technical_tickets': 1,
        'cancellation_tickets': 0,
        'customer_value_score': 1500.0,
        'risk_score': 0.3,
        'subscription_type': 'Premium',
        'province': 'ON',
        'payment_method': 'Credit Card',
        'paperless_billing': 1,
        'auto_pay': 1
    }
    
    # Initialize explainer
    model_path = 'models/churn_prediction_model.pkl'
    feature_names_path = 'models/feature_names.pkl'
    preprocessing_info_path = 'models/preprocessing_info.pkl'
    
    explainer = ChurnExplainer(model_path, feature_names_path, preprocessing_info_path)
    
    # Get explanation
    explanation = explainer.explain_prediction(sample_customer)
    
    print("Sample Customer Churn Explanation:")
    print("=" * 50)
    print(f"Customer ID: {explanation['customer_id']}")
    print(f"Churn Probability: {explanation['prediction']['churn_probability']:.2%}")
    print(f"Will Churn: {explanation['prediction']['will_churn']}")
    print(f"Summary: {explanation['summary']}")
    
    print("\nTop Contributing Factors:")
    for i, factor in enumerate(explanation['top_factors'][:5], 1):
        print(f"{i}. {factor['feature']}: {factor['impact']} (SHAP: {factor['shap_value']:.3f})")
    
    # Save explainer
    explainer.save_explainer('models/shap_explainer.pkl')
    
    return explainer, explanation

if __name__ == "__main__":
    explainer, explanation = create_sample_explanation()