#!/usr/bin/env python3
"""
Prometheus Monitoring Explorer for Customer Churn Prediction System
Helps you start monitoring services and explore metrics
"""

import subprocess
import time
import requests
import json
import os

def check_service_health(url, service_name):
    """Check if a service is accessible"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ {service_name} is running at {url}")
            return True
        else:
            print(f"✗ {service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print(f"✗ {service_name} is not accessible at {url}")
        return False

def start_monitoring_stack():
    """Start the monitoring stack using Docker Compose"""
    print("Starting monitoring stack...")
    print("This will start: API, Prometheus, Grafana, PostgreSQL, and exporters")
    
    # Start services
    try:
        subprocess.run([
            'docker-compose', 'up', '-d', 
            'prometheus', 'grafana', 'node-exporter', 'cadvisor'
        ], check=True)
        print("✓ Monitoring services started successfully")
        
        # Wait for services to initialize
        print("Waiting for services to initialize...")
        time.sleep(10)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to start monitoring stack: {e}")
        return False

def check_all_services():
    """Check health of all monitoring services"""
    print("\n=== SERVICE HEALTH CHECK ===")
    
    services = {
        "FastAPI (Churn Prediction)": "http://localhost:8000/health",
        "Prometheus": "http://localhost:9090/-/healthy",
        "Grafana": "http://localhost:3000/api/health",
        "Node Exporter": "http://localhost:9100/metrics",
        "cAdvisor": "http://localhost:8080/containers/",
    }
    
    healthy_services = 0
    for service_name, url in services.items():
        if check_service_health(url, service_name):
            healthy_services += 1
    
    print(f"\n{healthy_services}/{len(services)} services are healthy")
    return healthy_services == len(services)

def show_prometheus_queries():
    """Show useful Prometheus queries for ML monitoring"""
    print("\n=== USEFUL PROMETHEUS QUERIES ===")
    
    queries = {
        "Total Predictions Made": "sum(predictions_total)",
        "Prediction Rate (per minute)": "rate(predictions_total[1m]) * 60",
        "Average Response Time": "avg(prediction_duration_seconds)",
        "95th Percentile Response Time": "histogram_quantile(0.95, prediction_duration_seconds_bucket)",
        "Churn Prediction Ratio": "sum(predictions_total{prediction=\"true\"}) / sum(predictions_total)",
        "API Error Rate": "rate(api_requests_total{status_code!~\"2..\"}[5m])",
        "Feedback Rate": "rate(feedback_total[5m])",
        "Memory Usage": "container_memory_usage_bytes{name=\"churn-prediction-api\"}",
        "CPU Usage": "rate(container_cpu_usage_seconds_total{name=\"churn-prediction-api\"}[5m])",
    }
    
    for description, query in queries.items():
        print(f"• {description}:")
        print(f"  {query}")
        print()

def generate_sample_metrics():
    """Generate sample metrics by making API calls"""
    print("\n=== GENERATING SAMPLE METRICS ===")
    
    # Check if API is running
    if not check_service_health("http://localhost:8000/health", "API"):
        print("API is not running. Start it first with: uvicorn app.main:app --reload")
        return
    
    # Sample customer data
    sample_customers = [
        {
            "monthly_charges": 89.99,
            "total_charges": 2700.00,
            "contract_length": 24,
            "account_age_days": 730,
            "days_since_last_login": 3,
            "never_logged_in": 0,
            "age": 42,
            "total_transactions": 48,
            "total_amount": 4320.00,
            "avg_amount": 90.00,
            "std_amount": 15.25,
            "failure_rate": 0.04,
            "days_since_last_transaction": 7,
            "failed_transactions": 2,
            "total_tickets": 5,
            "avg_satisfaction": 4.2,
            "min_satisfaction": 3.0,
            "avg_resolution_time": 18.5,
            "billing_tickets": 3,
            "technical_tickets": 2,
            "cancellation_tickets": 0,
            "customer_value_score": 2500.0,
            "risk_score": 0.25,
            "subscription_type": "Premium",
            "province": "ON",
            "payment_method": "Credit Card",
            "paperless_billing": 1,
            "auto_pay": 1
        },
        {
            "monthly_charges": 45.99,
            "total_charges": 500.00,
            "contract_length": 6,
            "account_age_days": 180,
            "days_since_last_login": 30,
            "never_logged_in": 0,
            "age": 28,
            "total_transactions": 8,
            "total_amount": 360.00,
            "avg_amount": 45.00,
            "std_amount": 8.50,
            "failure_rate": 0.25,
            "days_since_last_transaction": 45,
            "failed_transactions": 2,
            "total_tickets": 8,
            "avg_satisfaction": 2.1,
            "min_satisfaction": 1.0,
            "avg_resolution_time": 72.5,
            "billing_tickets": 5,
            "technical_tickets": 2,
            "cancellation_tickets": 1,
            "customer_value_score": 150.0,
            "risk_score": 0.85,
            "subscription_type": "Basic",
            "province": "QC",
            "payment_method": "Debit Card",
            "paperless_billing": 0,
            "auto_pay": 0
        }
    ]
    
    # Make prediction requests
    print("Making prediction requests to generate metrics...")
    for i, customer_data in enumerate(sample_customers):
        try:
            response = requests.post(
                f"http://localhost:8000/predict/SAMPLE_CUSTOMER_{i+1}",
                json=customer_data,
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Customer {i+1}: {result['prediction']['churn_probability']:.1%} churn probability")
            else:
                print(f"✗ Customer {i+1}: API returned {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"✗ Customer {i+1}: Request failed - {e}")
    
    # Submit sample feedback
    print("\nSubmitting sample feedback...")
    feedback_data = {
        "customer_id": "SAMPLE_CUSTOMER_1",
        "predicted_churn": False,
        "actual_churn": False,
        "feedback_type": "correct_prediction",
        "confidence": 0.95,
        "notes": "Prediction was accurate"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/feedback",
            json=feedback_data,
            timeout=10
        )
        if response.status_code == 200:
            print("✓ Feedback submitted successfully")
        else:
            print(f"✗ Feedback submission failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Feedback submission failed: {e}")

def main():
    """Main function to explore Prometheus monitoring"""
    print("🔍 Customer Churn Prediction System - Prometheus Monitoring Explorer")
    print("=" * 70)
    
    print("\n1. STARTING MONITORING STACK")
    print("-" * 30)
    
    if not start_monitoring_stack():
        print("Failed to start monitoring stack. Please check Docker and try again.")
        return
    
    print("\n2. CHECKING SERVICE HEALTH")
    print("-" * 30)
    
    if not check_all_services():
        print("Some services are not healthy. Please check logs with: docker-compose logs")
    
    print("\n3. ACCESS POINTS")
    print("-" * 30)
    print("• Prometheus Dashboard: http://localhost:9090")
    print("• Grafana Dashboard: http://localhost:3000 (admin/admin)")
    print("• API Documentation: http://localhost:8000/docs")
    print("• API Metrics: http://localhost:8000/metrics")
    
    show_prometheus_queries()
    
    print("\n4. GENERATE SAMPLE METRICS")
    print("-" * 30)
    generate_sample_metrics()
    
    print("\n5. NEXT STEPS")
    print("-" * 30)
    print("• Visit Prometheus at http://localhost:9090 and try the queries above")
    print("• Check Grafana at http://localhost:3000 for visual dashboards")
    print("• Use the API at http://localhost:8000/docs to generate more metrics")
    print("• Monitor alerts and system health in real-time")
    
    print("\n✓ Monitoring exploration complete!")

if __name__ == "__main__":
    main()