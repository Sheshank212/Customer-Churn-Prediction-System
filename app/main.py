"""
Customer Churn Prediction API
FastAPI application for serving ML predictions with SHAP explanations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import asyncio
from contextlib import asynccontextmanager
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import time

# Import custom modules
from app.utils.shap_explainer import ChurnExplainer
from app.api.models import (
    CustomerPredictionRequest, CustomerPredictionResponse,
    FeedbackRequest, FeedbackResponse, HealthResponse,
    PredictionResult, PredictionExplanation, FeatureImportance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REGISTRY = CollectorRegistry()
prediction_counter = Counter('predictions_total', 'Total predictions made', ['model_version', 'prediction'], registry=REGISTRY)
prediction_histogram = Histogram('prediction_duration_seconds', 'Prediction request duration', registry=REGISTRY)
feedback_counter = Counter('feedback_total', 'Total feedback received', ['feedback_type'], registry=REGISTRY)
api_request_counter = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status_code'], registry=REGISTRY)

# Global variables for model and explainer
model = None
explainer = None
model_version = "1.0.0"
model_metadata = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting Customer Churn Prediction API...")
    await load_models()
    logger.info("Models loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Customer Churn Prediction API...")

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready API for customer churn prediction with SHAP explanations",
    version=model_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Grafana
        "http://localhost:8080",  # Frontend
        "http://localhost:8000",  # API docs
        "https://your-domain.com"  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Dependency for request metrics
async def track_request_metrics(request: Request):
    """Track API request metrics"""
    start_time = time.time()
    
    yield
    
    duration = time.time() - start_time
    endpoint = request.url.path
    method = request.method
    # Note: status_code would be available in middleware
    api_request_counter.labels(endpoint=endpoint, method=method, status_code="200").inc()

def convert_api_model_to_explainer_format(customer_data: CustomerPredictionRequest, customer_id: str) -> Dict[str, Any]:
    """Convert API model to explainer format with default values for missing fields"""
    
    # Start with the provided data
    explainer_data = {
        'customer_id': customer_id,
        'monthly_charges': customer_data.monthly_charges,
        'total_charges': customer_data.total_charges,
        'contract_length': customer_data.contract_length,
        'account_age_days': customer_data.account_age_days,
        'days_since_last_login': customer_data.days_since_last_login or 0,
        'province': str(customer_data.province) if customer_data.province else 'ON',
        'payment_method': str(customer_data.payment_method) if customer_data.payment_method else 'Credit Card',
        'subscription_type': str(customer_data.subscription_type) if customer_data.subscription_type else 'Basic',
        'paperless_billing': int(customer_data.paperless_billing) if customer_data.paperless_billing is not None else 0,
        'auto_pay': int(customer_data.auto_pay) if customer_data.auto_pay is not None else 0,
    }
    
    # Add calculated/default fields that the explainer expects
    explainer_data.update({
        'never_logged_in': 1 if customer_data.days_since_last_login and customer_data.days_since_last_login > 365 else 0,
        'age': 35,  # Default age
        'total_transactions': customer_data.support_tickets or 0 * 10,  # Estimate based on support tickets
        'total_amount': customer_data.total_charges,
        'avg_amount': customer_data.monthly_charges,
        'std_amount': customer_data.monthly_charges * 0.1,  # 10% of monthly charges
        'failure_rate': min(0.1, (customer_data.transaction_failures or 0) / 10),  # Estimate failure rate
        'days_since_last_transaction': customer_data.days_since_last_login or 0,
        'failed_transactions': customer_data.transaction_failures or 0,
        'total_tickets': customer_data.support_tickets or 0,
        'avg_satisfaction': customer_data.satisfaction_score or 3.5,
        'min_satisfaction': (customer_data.satisfaction_score or 3.5) - 0.5,
        'avg_resolution_time': 24.0,  # Default resolution time
        'billing_tickets': max(0, (customer_data.support_tickets or 0) - 1),
        'technical_tickets': min(1, customer_data.support_tickets or 0),
        'cancellation_tickets': 0,
        'customer_value_score': customer_data.total_charges,
        'risk_score': min(1.0, (customer_data.transaction_failures or 0) / 10),
    })
    
    return explainer_data

async def load_models():
    """Load ML model and SHAP explainer"""
    global model, explainer, model_metadata
    
    try:
        # Load model
        model_path = 'models/churn_prediction_model.pkl'
        feature_names_path = 'models/feature_names.pkl'
        preprocessing_info_path = 'models/preprocessing_info.pkl'
        
        model = joblib.load(model_path)
        
        # Load explainer
        explainer = ChurnExplainer(model_path, feature_names_path, preprocessing_info_path)
        
        # Load metadata
        model_metadata = {
            'model_type': type(model).__name__,
            'model_version': model_version,
            'loaded_at': datetime.now().isoformat(),
            'feature_count': len(explainer.feature_names)
        }
        
        logger.info(f"Model loaded: {model_metadata}")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Churn Prediction API",
        "version": model_version,
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None and explainer is not None else "unhealthy",
        timestamp=datetime.now(),
        version=model_version,
        model_loaded=model is not None and explainer is not None
    )

@app.post("/predict/{customer_id}", response_model=CustomerPredictionResponse)
async def predict_churn(
    customer_id: str,
    customer_data: CustomerPredictionRequest,
    background_tasks: BackgroundTasks,
    request_tracker: Any = Depends(track_request_metrics)
):
    """
    Predict customer churn probability with SHAP explanations
    
    This endpoint:
    - Validates input data
    - Makes churn prediction
    - Provides SHAP-based explanations
    - Logs prediction for monitoring
    - Returns comprehensive response
    """
    
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with prediction_histogram.time():
            # Convert API model to explainer format
            customer_dict = convert_api_model_to_explainer_format(customer_data, customer_id)
            
            # Get prediction with explanation
            explanation = explainer.explain_prediction(customer_dict)
            
            # Create response compatible with api/models.py
            from app.api.models import PredictionResult, PredictionExplanation, RiskLevel
            
            # Determine risk level
            churn_prob = explanation['prediction']['churn_probability']
            if churn_prob > 0.7:
                risk_level = RiskLevel.VERY_HIGH
            elif churn_prob > 0.5:
                risk_level = RiskLevel.HIGH
            elif churn_prob > 0.3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Create prediction result
            prediction_result = PredictionResult(
                churn_probability=explanation['prediction']['churn_probability'],
                will_churn=explanation['prediction']['will_churn'],
                confidence=explanation['prediction']['confidence'],
                risk_level=risk_level
            )
            
            # Create feature importance list
            feature_importance = []
            for factor in explanation['top_factors']:
                feature_importance.append(FeatureImportance(
                    feature=factor['feature'],
                    importance=abs(factor['shap_value']),
                    impact=factor['impact'],
                    shap_value=factor['shap_value']
                ))
            
            # Create explanation
            prediction_explanation = PredictionExplanation(
                summary=explanation['summary'],
                top_factors=feature_importance,
                recommendation=None  # Could be enhanced based on risk level
            )
            
            # Create response
            response = CustomerPredictionResponse(
                customer_id=customer_id,
                prediction=prediction_result,
                explanation=prediction_explanation,
                model_version=model_version,
                timestamp=datetime.now()
            )
            
            # Update metrics
            prediction_counter.labels(
                model_version=model_version,
                prediction=str(explanation['prediction']['will_churn'])
            ).inc()
            
            # Log prediction (background task)
            background_tasks.add_task(
                log_prediction,
                customer_id,
                explanation['prediction'],
                explanation['summary']
            )
            
            return response
            
    except Exception as e:
        logger.error(f"Error making prediction for {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit feedback for model improvement
    
    This endpoint:
    - Accepts feedback about prediction accuracy
    - Stores feedback for model retraining
    - Updates monitoring metrics
    - Returns confirmation
    """
    
    try:
        # Validate feedback
        if feedback.predicted_churn == feedback.actual_churn:
            feedback_type = "correct_prediction"
        else:
            feedback_type = "incorrect_prediction"
        
        # Update metrics
        feedback_counter.labels(feedback_type=feedback_type).inc()
        
        # Store feedback (background task)
        background_tasks.add_task(store_feedback, feedback)
        
        return {
            "message": "Feedback received successfully",
            "customer_id": feedback.customer_id,
            "feedback_type": feedback_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    metrics_data = generate_latest(REGISTRY)
    if not metrics_data.endswith(b'# EOF\n'):
        metrics_data += b'# EOF\n'
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/model/info")
async def get_model_info():
    """Get model information and metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_metadata": model_metadata,
        "feature_count": len(explainer.feature_names),
        "feature_names": explainer.feature_names[:10],  # First 10 features
        "model_type": type(model).__name__
    }

@app.post("/batch_predict")
async def batch_predict(
    customers: List[CustomerPredictionRequest],
    background_tasks: BackgroundTasks
):
    """
    Batch prediction endpoint for multiple customers
    
    Useful for:
    - Bulk processing
    - Scheduled batch jobs
    - Analysis of customer segments
    """
    
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(customers) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        results = []
        
        for i, customer_data in enumerate(customers):
            # Generate customer ID for batch processing
            customer_id = f"BATCH_CUSTOMER_{i+1:03d}"
            customer_dict = convert_api_model_to_explainer_format(customer_data, customer_id)
            explanation = explainer.explain_prediction(customer_dict)
            
            result = {
                "customer_id": customer_dict['customer_id'],
                "prediction": explanation['prediction'],
                "summary": explanation['summary']
            }
            results.append(result)
            
            # Update metrics
            prediction_counter.labels(
                model_version=model_version,
                prediction=str(explanation['prediction']['will_churn'])
            ).inc()
        
        return {
            "results": results,
            "batch_size": len(customers),
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Background tasks

async def log_prediction(customer_id: str, prediction: Dict[str, Any], summary: str):
    """Log prediction for monitoring and analysis"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "churn_probability": prediction['churn_probability'],
            "will_churn": prediction['will_churn'],
            "confidence": prediction['confidence'],
            "summary": summary,
            "model_version": model_version
        }
        
        logger.info(f"Prediction logged: {log_entry}")
        
        # In production, you would store this in a database or data lake
        # For now, we'll just log it
        
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")

async def store_feedback(feedback: FeedbackRequest):
    """Store feedback for model improvement"""
    try:
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": feedback.customer_id,
            "predicted_churn": feedback.predicted_churn,
            "actual_churn": feedback.actual_churn,
            "feedback_type": str(feedback.feedback_type),
            "confidence_rating": feedback.confidence_rating,
            "comments": feedback.comments
        }
        
        logger.info(f"Feedback stored: {feedback_entry}")
        
        # In production, you would store this in a database
        # This data would be used for model retraining
        
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Customer Churn Prediction API",
        version=model_version,
        description="""
        ## Customer Churn Prediction API
        
        Production-ready API for predicting customer churn with explainable AI.
        
        ### Features
        - **Real-time predictions** with SHAP explanations
        - **Batch processing** for multiple customers
        - **Feedback system** for model improvement
        - **Comprehensive monitoring** with Prometheus metrics
        - **Production-ready** with proper error handling and logging
        
        ### Use Cases
        - Customer retention programs
        - Proactive customer engagement
        - Risk assessment and prioritization
        - Business intelligence and analytics
        
        ### Model Information
        - **Model Type**: Logistic Regression
        - **Features**: 37 engineered features
        - **Accuracy**: 77.1%
        - **AUC Score**: 0.745
        """,
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )