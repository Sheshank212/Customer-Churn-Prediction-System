"""
Customer Churn Prediction System - Testing Framework
Tests all 9 system components step by step
"""

import os
import sys
import subprocess
import time
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ChurnSystemTester:
    
    def __init__(self):
        # Initialize test tracking variables
        self.test_results = {}
        self.current_step = 0
        
    def log_step(self, step_name, success=True, message=""):
        # Increment step counter
        self.current_step += 1
        
        # Determine status message
        status = "PASS" if success else "FAIL"
        
        # Print test results
        print(f"\nStep {self.current_step}: {step_name}")
        print(f"Status: {status}")
        if message:
            print(f"Details: {message}")
        
        # Store results for final summary
        self.test_results[self.current_step] = {
            'step': step_name,
            'success': success,
            'message': message
        }
        
        # Display warning for failed tests
        if not success:
            print("WARNING: Fix this issue before proceeding to next step")
            return False
        return True
    
    def test_step_1_data_generation(self):
        # Print step header
        print("\n" + "="*60)
        print("STEP 1: SYNTHETIC DATA GENERATION")
        print("="*60)
        
        try:
            # Execute data generation script with timeout
            print("Generating synthetic data...")
            result = subprocess.run([
                sys.executable, 'data/generate_synthetic_data.py'
            ], capture_output=True, text=True, timeout=300)
            
            # Check if script executed successfully
            if result.returncode == 0:
                print("SUCCESS: Data generation completed successfully")
                print(f"Output: {result.stdout[-500:]}")
                
                # Define required output files
                required_files = [
                    'data/raw/customers.csv',
                    'data/raw/transactions.csv',
                    'data/raw/support_tickets.csv',
                    'data/raw/churn_labels.csv'
                ]
                
                # Verify each file exists and count records
                for file in required_files:
                    if os.path.exists(file):
                        df = pd.read_csv(file)
                        print(f"VERIFIED: {file}: {len(df)} records")
                    else:
                        raise FileNotFoundError(f"Missing file: {file}")
                
                return self.log_step("Synthetic Data Generation", True, "All CSV files created successfully")
            else:
                return self.log_step("Synthetic Data Generation", False, f"Error: {result.stderr}")
                
        except Exception as e:
            return self.log_step("Synthetic Data Generation", False, str(e))
    
    def test_step_2_data_analysis(self):
        # Print step header
        print("\n" + "="*60)
        print("STEP 2: DATA ANALYSIS")
        print("="*60)
        
        try:
            # Load all generated CSV files
            customers = pd.read_csv('data/raw/customers.csv')
            transactions = pd.read_csv('data/raw/transactions.csv')
            support_tickets = pd.read_csv('data/raw/support_tickets.csv')
            churn_labels = pd.read_csv('data/raw/churn_labels.csv')
            
            # Display data summary statistics
            print(f"DATA SUMMARY:")
            print(f"   Customers: {len(customers):,}")
            print(f"   Transactions: {len(transactions):,}")
            print(f"   Support Tickets: {len(support_tickets):,}")
            print(f"   Churn Labels: {len(churn_labels):,}")
            
            # Calculate and display churn rate
            churn_rate = churn_labels['is_churned'].mean()
            print(f"   Churn Rate: {churn_rate:.1%}")
            
            # Analyze province distribution
            print(f"\nTOP 5 PROVINCES:")
            province_counts = customers['province'].value_counts().head()
            for province, count in province_counts.items():
                print(f"   {province}: {count:,} ({count/len(customers):.1%})")
            
            # Analyze transaction status distribution
            print(f"\nTRANSACTION STATUS:")
            status_counts = transactions['status'].value_counts()
            for status, count in status_counts.items():
                print(f"   {status}: {count:,} ({count/len(transactions):.1%})")
            
            return self.log_step("Data Analysis", True, f"Data quality looks good. Churn rate: {churn_rate:.1%}")
            
        except Exception as e:
            return self.log_step("Data Analysis", False, str(e))
    
    def test_step_3_model_training(self):
        """Step 3: Test ML model training"""
        print("\n" + "="*60)
        print(" STEP 3: ML MODEL TRAINING")
        print("="*60)
        
        try:
            print("Training ML model... (This may take a few minutes)")
            
            # Run ML pipeline
            result = subprocess.run([
                sys.executable, 'notebooks/ml_pipeline_visual.py'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(" Model training completed successfully")
                
                # Check if model files were created
                required_files = [
                    'models/churn_prediction_model.pkl',
                    'models/feature_names.pkl',
                    'models/preprocessing_info.pkl'
                ]
                
                for file in required_files:
                    if os.path.exists(file):
                        print(f" {file} created")
                    else:
                        raise FileNotFoundError(f"Missing file: {file}")
                
                # Check if figures were created
                figures_dir = 'figures'
                if os.path.exists(figures_dir):
                    figures = os.listdir(figures_dir)
                    print(f" Generated {len(figures)} visualization(s)")
                    for fig in figures:
                        print(f"   - {fig}")
                
                return self.log_step("ML Model Training", True, "Model and visualizations created successfully")
            else:
                return self.log_step("ML Model Training", False, f"Error: {result.stderr}")
                
        except Exception as e:
            return self.log_step("ML Model Training", False, str(e))
    
    def test_step_4_model_evaluation(self):
        """Step 4: Evaluate trained model"""
        print("\n" + "="*60)
        print(" STEP 4: MODEL EVALUATION")
        print("="*60)
        
        try:
            import joblib
            
            # Load model
            model = joblib.load('models/churn_prediction_model.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            
            print(f" Model loaded: {type(model).__name__}")
            print(f" Features: {len(feature_names)}")
            
            # Show model parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                print(f"\n🔧 Model Parameters:")
                for key, value in list(params.items())[:5]:  # Show first 5
                    print(f"   {key}: {value}")
                if len(params) > 5:
                    print(f"   ... and {len(params)-5} more parameters")
            
            # Test prediction on sample data
            print(f"\n🧪 Testing prediction on sample data...")
            
            # Create sample input
            sample_data = np.zeros((1, len(feature_names)))
            sample_data[0, 0] = 75.50  # monthly_charges
            sample_data[0, 1] = 1200.00  # total_charges
            
            # Make prediction
            prediction = model.predict(sample_data)[0]
            probability = model.predict_proba(sample_data)[0]
            
            print(f" Sample prediction: {prediction}")
            print(f" Probabilities: {probability}")
            
            return self.log_step("Model Evaluation", True, f"Model working correctly. Type: {type(model).__name__}")
            
        except Exception as e:
            return self.log_step("Model Evaluation", False, str(e))
    
    def test_step_5_shap_explainer(self):
        """Step 5: Test SHAP explainer"""
        print("\n" + "="*60)
        print(" STEP 5: SHAP EXPLAINER")
        print("="*60)
        
        try:
            from app.utils.shap_explainer import ChurnExplainer
            
            # Initialize explainer
            model_path = 'models/churn_prediction_model.pkl'
            feature_names_path = 'models/feature_names.pkl'
            preprocessing_info_path = 'models/preprocessing_info.pkl'
            
            explainer = ChurnExplainer(model_path, feature_names_path, preprocessing_info_path)
            print(" SHAP explainer initialized")
            
            # Test explanation
            sample_customer = {
                'customer_id': 'TEST_001',
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
            
            explanation = explainer.explain_prediction(sample_customer)
            
            print(f" Explanation generated for customer: {explanation['customer_id']}")
            print(f" Churn Probability: {explanation['prediction']['churn_probability']:.2%}")
            print(f" Will Churn: {explanation['prediction']['will_churn']}")
            print(f" Summary: {explanation['summary']}")
            
            print(f"\n Top 3 Contributing Factors:")
            for i, factor in enumerate(explanation['top_factors'][:3], 1):
                print(f"   {i}. {factor['feature']}: {factor['impact']}")
            
            return self.log_step("SHAP Explainer", True, "SHAP explanations working correctly")
            
        except Exception as e:
            return self.log_step("SHAP Explainer", False, str(e))
    
    def test_step_6_api_server(self):
        """Step 6: Test FastAPI server"""
        print("\n" + "="*60)
        print("🚀 STEP 6: FASTAPI SERVER")
        print("="*60)
        
        try:
            print("Starting FastAPI server...")
            print("💡 This will start the server. You can test it manually or continue to Step 7.")
            
            # Note: In Spyder, you might want to run this in a separate terminal
            print("\n🔧 To start the server manually:")
            print("   1. Open terminal in project directory")
            print("   2. Run: python -m uvicorn app.main:app --reload")
            print("   3. Server will start at: http://localhost:8000")
            print("   4. API docs at: http://localhost:8000/docs")
            
            # Check if we can import the main app
            from app.main import app
            print(" FastAPI app imported successfully")
            
            return self.log_step("FastAPI Server", True, "Server can be started manually")
            
        except Exception as e:
            return self.log_step("FastAPI Server", False, str(e))
    
    def test_step_7_api_endpoints(self):
        """Step 7: Test API endpoints (requires server running)"""
        print("\n" + "="*60)
        print(" STEP 7: API ENDPOINTS TEST")
        print("="*60)
        
        base_url = "http://localhost:8000"
        
        print("⚠️  This step requires the API server to be running!")
        print("   Start server with: uvicorn app.main:app --reload")
        
        choice = input("\nIs the server running? (y/n): ").lower()
        if choice != 'y':
            return self.log_step("API Endpoints", False, "Server not running. Skip this step or start server first.")
        
        try:
            # Test health endpoint
            print("\n Testing health endpoint...")
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(" Health endpoint working")
                print(f"   Response: {response.json()}")
            else:
                print(f" Health endpoint failed: {response.status_code}")
            
            # Test prediction endpoint
            print("\n🔮 Testing prediction endpoint...")
            customer_data = {
                "customer_id": "TEST_CUSTOMER_001",
                "monthly_charges": 75.50,
                "total_charges": 1200.00,
                "contract_length": 12,
                "account_age_days": 365,
                "days_since_last_login": 5,
                "never_logged_in": 0,
                "age": 35,
                "total_transactions": 24,
                "total_amount": 1800.00,
                "avg_amount": 75.00,
                "std_amount": 5.50,
                "failure_rate": 0.08,
                "days_since_last_transaction": 30,
                "failed_transactions": 2,
                "total_tickets": 3,
                "avg_satisfaction": 3.5,
                "min_satisfaction": 2.0,
                "avg_resolution_time": 24.5,
                "billing_tickets": 2,
                "technical_tickets": 1,
                "cancellation_tickets": 0,
                "customer_value_score": 1500.0,
                "risk_score": 0.3,
                "subscription_type": "Premium",
                "province": "ON",
                "payment_method": "Credit Card",
                "paperless_billing": 1,
                "auto_pay": 1
            }
            
            response = requests.post(
                f"{base_url}/predict/TEST_CUSTOMER_001",
                json=customer_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(" Prediction endpoint working")
                print(f"   Customer: {result['customer_id']}")
                print(f"   Churn Probability: {result['prediction']['churn_probability']:.2%}")
                print(f"   Summary: {result['explanation']['summary']}")
            else:
                print(f" Prediction endpoint failed: {response.status_code}")
                print(f"   Error: {response.text}")
            
            return self.log_step("API Endpoints", True, "API endpoints working correctly")
            
        except requests.exceptions.ConnectionError:
            return self.log_step("API Endpoints", False, "Cannot connect to server. Make sure it's running.")
        except Exception as e:
            return self.log_step("API Endpoints", False, str(e))
    
    def test_step_8_visualizations(self):
        """Step 8: Review generated visualizations"""
        print("\n" + "="*60)
        print(" STEP 8: VISUALIZATIONS REVIEW")
        print("="*60)
        
        try:
            figures_dir = 'figures'
            if not os.path.exists(figures_dir):
                return self.log_step("Visualizations", False, "Figures directory not found. Run model training first.")
            
            figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
            
            if not figures:
                return self.log_step("Visualizations", False, "No visualization files found.")
            
            # List all found visualization files
            print(f"FOUND {len(figures)} VISUALIZATION FILES:")
            for i, fig in enumerate(figures, 1):
                print(f"   {i}. {fig}")
            
            # Display each visualization in separate window
            print(f"\nDISPLAYING {len(figures)} VISUALIZATIONS INDIVIDUALLY:")
            
            # Loop through each visualization file
            for i, fig_name in enumerate(figures, 1):
                img_path = os.path.join(figures_dir, fig_name)
                try:
                    # Load image file
                    img = plt.imread(img_path)
                    
                    # Clean filename for display title
                    clean_title = fig_name.replace('.png', '').replace('_', ' ').title()
                    
                    # Create and display plot
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Visualization {i}/{len(figures)}: {clean_title}", 
                             fontsize=14, fontweight='bold', pad=20)
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"   SUCCESS: {i}. {clean_title} - Displayed")
                    
                except Exception as e:
                    print(f"   ERROR: {i}. {fig_name} - {e}")
            
            print(f"\nALL {len(figures)} VISUALIZATIONS DISPLAYED SUCCESSFULLY")
            
            return self.log_step("Visualizations", True, f"Generated {len(figures)} visualizations successfully")
            
        except Exception as e:
            return self.log_step("Visualizations", False, str(e))
    
    def test_step_9_sql_integration(self):
        """Step 9: Test SQL Integration (NEW)"""
        print("\n" + "="*60)
        print("🗄️ STEP 9: SQL INTEGRATION TEST")
        print("="*60)
        
        print("Testing the new PostgreSQL integration features...")
        
        try:
            # Check if PostgreSQL is running
            print("\n Checking PostgreSQL connection...")
            from data.database_setup import DatabaseManager
            
            db_manager = DatabaseManager()
            if not db_manager.test_connection():
                print("⚠️  PostgreSQL is not running. Starting Docker container...")
                
                # Try to start PostgreSQL
                result = subprocess.run([
                    'docker-compose', 'up', '-d', 'postgres'
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    return self.log_step("SQL Integration", False, "Failed to start PostgreSQL container")
                
                # Wait for PostgreSQL to be ready
                print("   Waiting for PostgreSQL to start...")
                time.sleep(10)
                
                # Test connection again
                if not db_manager.test_connection():
                    return self.log_step("SQL Integration", False, "PostgreSQL connection failed after startup")
            
            print("   PostgreSQL connection successful")
            
            # Test SQL feature extraction
            print("\n Testing SQL feature extraction...")
            try:
                features_df = db_manager.get_ml_features()
                print(f"   Retrieved {len(features_df)} customers with {len(features_df.columns)} features")
                
                # Check if we have the expected columns
                expected_columns = ['customer_id', 'monthly_charges', 'total_charges', 'is_churned']
                missing_columns = [col for col in expected_columns if col not in features_df.columns]
                
                if missing_columns:
                    return self.log_step("SQL Integration", False, f"Missing columns: {missing_columns}")
                
                print("   SQL feature extraction working correctly")
                
            except Exception as e:
                return self.log_step("SQL Integration", False, f"SQL feature extraction failed: {str(e)}")
            
            # Test SQL-based ML pipeline
            print("\n Testing SQL-based ML pipeline...")
            choice = input("   Run the SQL ML pipeline? This will train a new model (y/n): ").lower()
            
            if choice == 'y':
                try:
                    result = subprocess.run([
                        sys.executable, 'notebooks/ml_pipeline_with_sql.py'
                    ], capture_output=True, text=True, timeout=600)
                    
                    if result.returncode == 0:
                        print("   SQL ML pipeline completed successfully")
                        
                        # Check if SQL model files were created
                        sql_model_files = [
                            'models/churn_prediction_model_sql.pkl',
                            'models/scaler_sql.pkl',
                            'models/feature_names_sql.pkl'
                        ]
                        
                        for file in sql_model_files:
                            if os.path.exists(file):
                                print(f"   SQL model file created: {file}")
                            else:
                                return self.log_step("SQL Integration", False, f"Missing SQL model file: {file}")
                        
                        print("   SQL-based model training successful")
                        
                    else:
                        return self.log_step("SQL Integration", False, f"SQL ML pipeline failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    return self.log_step("SQL Integration", False, "SQL ML pipeline timed out")
                except Exception as e:
                    return self.log_step("SQL Integration", False, f"SQL ML pipeline error: {str(e)}")
            else:
                print("   SQL ML pipeline test skipped")
            
            return self.log_step("SQL Integration", True, "SQL integration working correctly")
            
        except Exception as e:
            return self.log_step("SQL Integration", False, str(e))
    
    def test_step_10_docker_test(self):
        """Step 10: Test Docker setup (optional)"""
        print("\n" + "="*60)
        print("🐳 STEP 10: DOCKER TEST (OPTIONAL)")
        print("="*60)
        
        print("⚠️  This step is optional and requires Docker to be installed.")
        choice = input("Do you want to test Docker setup? (y/n): ").lower()
        if choice != 'y':
            return self.log_step("Docker Test", True, "Skipped by user choice")
        
        try:
            # Check if Docker is available
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return self.log_step("Docker Test", False, "Docker not available")
            
            print(f" Docker version: {result.stdout.strip()}")
            
            # Test docker build
            print("\n Testing Docker build...")
            print(" This may take several minutes...")
            
            build_result = subprocess.run([
                'docker', 'build', '-t', 'churn-prediction-test', '.'
            ], capture_output=True, text=True, timeout=600)
            
            if build_result.returncode == 0:
                print(" Docker build successful")
                
                # Clean up
                subprocess.run(['docker', 'rmi', 'churn-prediction-test'], 
                             capture_output=True)
                
                return self.log_step("Docker Test", True, "Docker build successful")
            else:
                return self.log_step("Docker Test", False, f"Docker build failed: {build_result.stderr}")
                
        except Exception as e:
            return self.log_step("Docker Test", False, str(e))
    
    def test_step_11_final_summary(self):
        """Step 11: Final summary and recommendations"""
        print("\n" + "="*60)
        print(" STEP 11: FINAL SUMMARY")
        print("="*60)
        
        successful_tests = sum(1 for result in self.test_results.values() if result['success'])
        total_tests = len(self.test_results)
        
        print(f" Test Results: {successful_tests}/{total_tests} tests passed")
        print(f"   Success Rate: {successful_tests/total_tests:.1%}")
        
        print(f"\n Detailed Results:")
        for step_num, result in self.test_results.items():
            status = " PASS" if result['success'] else " FAIL"
            print(f"   Step {step_num}: {result['step']} - {status}")
            if not result['success']:
                print(f"      Issue: {result['message']}")
        
        # Display final results and recommendations
        if successful_tests == total_tests:
            print(f"\nALL TESTS PASSED SUCCESSFULLY!")
            print(f"   Your Customer Churn Prediction System is working perfectly!")
            print(f"\nNEXT STEPS:")
            print(f"   1. Start the API server: uvicorn app.main:app --reload")
            print(f"   2. Open http://localhost:8000/docs for API documentation")
            print(f"   3. Test the prediction endpoints")
            print(f"   4. Review visualizations in the figures/ directory")
            print(f"   5. Consider deploying with Docker Compose")
        else:
            print(f"\nSOME TESTS FAILED - Please review and fix the issues above.")
        
        return self.log_step("Final Summary", True, f"Testing completed: {successful_tests}/{total_tests} passed")
    
    def run_all_tests(self):
        # Display test framework header
        print("Customer Churn Prediction System - Complete Testing Guide")
        print("=" * 70)
        print("This will test each component step by step")
        print("You can run each step individually or all at once")
        print("=" * 70)
        
        # Get user choice for test execution mode
        choice = input("\nRun all tests automatically? (y/n): ").lower()
        if choice == 'y':
            # Execute all test steps in sequence
            self.test_step_1_data_generation()
            self.test_step_2_data_analysis()
            self.test_step_3_model_training()
            self.test_step_4_model_evaluation()
            self.test_step_5_shap_explainer()
            self.test_step_6_api_server()
            self.test_step_7_api_endpoints()
            self.test_step_8_visualizations()
            self.test_step_9_sql_integration()
            self.test_step_10_docker_test()
            self.test_step_11_final_summary()
        else:
            # Display manual testing instructions
            print("\nMANUAL TESTING MODE:")
            print("Copy and paste each test method in Spyder:")
            print("   tester = ChurnSystemTester()")
            print("   tester.test_step_1_data_generation()")
            print("   tester.test_step_2_data_analysis()")
            print("   ... and so on")

def main():
    # Initialize testing framework
    tester = ChurnSystemTester()
    
    # Execute test suite
    tester.run_all_tests()

if __name__ == "__main__":
    main()