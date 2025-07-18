"""
Test script for Customer Churn Prediction API
Demonstrates API functionality and performance
"""

import requests
import json
from datetime import datetime
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction():
    """Test prediction endpoint"""
    print("\nTesting prediction...")
    
    # Sample customer data
    customer_data = {
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
    
    try:
        customer_id = "CUST_TEST_001"
        response = requests.post(
            f"{BASE_URL}/predict/{customer_id}",
            json=customer_data
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Customer ID: {result['customer_id']}")
            print(f"Churn Probability: {result['prediction']['churn_probability']:.2%}")
            print(f"Will Churn: {result['prediction']['will_churn']}")
            print(f"Summary: {result['explanation']['summary']}")
            
            print("\nTop Contributing Factors:")
            for i, factor in enumerate(result['explanation']['top_factors'][:5], 1):
                print(f"{i}. {factor['feature']}: {factor['impact']}")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\nTesting batch prediction...")
    
    # Sample batch data
    batch_data = []
    for i in range(3):
        customer_data = {
            "customer_id": f"CUST_BATCH_{i+1:03d}",
            "monthly_charges": 50.0 + i * 25,
            "total_charges": 600.0 + i * 400,
            "contract_length": 12,
            "account_age_days": 300 + i * 100,
            "days_since_last_login": 5 + i * 10,
            "never_logged_in": 0,
            "age": 30 + i * 5,
            "total_transactions": 20 + i * 5,
            "total_amount": 1000.0 + i * 500,
            "avg_amount": 50.0 + i * 25,
            "std_amount": 5.0 + i * 2,
            "failure_rate": 0.05 + i * 0.03,
            "days_since_last_transaction": 20 + i * 10,
            "failed_transactions": 1 + i,
            "total_tickets": 2 + i,
            "avg_satisfaction": 4.0 - i * 0.5,
            "min_satisfaction": 3.0 - i * 0.5,
            "avg_resolution_time": 20.0 + i * 10,
            "billing_tickets": 1 + i,
            "technical_tickets": 1,
            "cancellation_tickets": 0,
            "customer_value_score": 1200.0 + i * 300,
            "risk_score": 0.2 + i * 0.1,
            "subscription_type": ["Basic", "Premium", "Enterprise"][i],
            "province": ["ON", "BC", "AB"][i],
            "payment_method": "Credit Card",
            "paperless_billing": 1,
            "auto_pay": 1
        }
        batch_data.append(customer_data)
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=batch_data
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Batch Size: {result['batch_size']}")
            print(f"Model Version: {result['model_version']}")
            
            print("\nBatch Results:")
            for i, res in enumerate(result['results'], 1):
                print(f"{i}. {res['customer_id']}: {res['prediction']['churn_probability']:.2%} churn probability")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_feedback():
    """Test feedback endpoint"""
    print("\nTesting feedback...")
    
    feedback_data = {
        "customer_id": "CUST_TEST_001",
        "predicted_churn": False,
        "actual_churn": True,
        "feedback_type": "incorrect_prediction",
        "confidence": 0.8,
        "notes": "Customer churned due to competitor offer"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/feedback",
            json=feedback_data
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("\nTesting metrics...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"Status Code: {response.status_code}")
        print(f"Metrics available: {len(response.text.splitlines())} lines")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def performance_test():
    """Basic performance test"""
    print("\nRunning performance test...")
    
    customer_data = {
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
    
    num_requests = 10
    total_time = 0
    successful_requests = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict/CUST_PERF_{i:03d}",
                json=customer_data
            )
            end_time = time.time()
            
            if response.status_code == 200:
                successful_requests += 1
                total_time += (end_time - start_time)
                
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    if successful_requests > 0:
        avg_response_time = total_time / successful_requests
        print(f"Successful requests: {successful_requests}/{num_requests}")
        print(f"Average response time: {avg_response_time:.3f} seconds")
        print(f"Requests per second: {1/avg_response_time:.2f}")
    else:
        print("No successful requests")

def main():
    """Run all tests"""
    print("Customer Churn Prediction API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Feedback", test_feedback),
        ("Metrics", test_metrics),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        performance_test()
    else:
        print("⚠️  Some tests failed. Check the API server.")

if __name__ == "__main__":
    main()