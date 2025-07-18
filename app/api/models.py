"""
Pydantic Models for Customer Churn Prediction API
Defines request/response schemas and validation rules
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class ProvinceName(str, Enum):
    """Valid Canadian provinces and territories"""
    ON = "ON"  # Ontario
    QC = "QC"  # Quebec
    BC = "BC"  # British Columbia
    AB = "AB"  # Alberta
    MB = "MB"  # Manitoba
    SK = "SK"  # Saskatchewan
    NS = "NS"  # Nova Scotia
    NB = "NB"  # New Brunswick
    NL = "NL"  # Newfoundland and Labrador
    PE = "PE"  # Prince Edward Island
    YT = "YT"  # Yukon
    NT = "NT"  # Northwest Territories
    NU = "NU"  # Nunavut


class PaymentMethod(str, Enum):
    """Valid payment methods"""
    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    BANK_TRANSFER = "Bank Transfer"
    CASH = "Cash"
    CHEQUE = "Cheque"


class SubscriptionType(str, Enum):
    """Valid subscription types"""
    BASIC = "Basic"
    STANDARD = "Standard"
    PREMIUM = "Premium"
    ENTERPRISE = "Enterprise"


class RiskLevel(str, Enum):
    """Risk level categories"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class FeedbackType(str, Enum):
    """Types of feedback"""
    INCORRECT_PREDICTION = "incorrect_prediction"
    CORRECT_PREDICTION = "correct_prediction"
    MODEL_IMPROVEMENT = "model_improvement"
    DATA_QUALITY = "data_quality"


class CustomerPredictionRequest(BaseModel):
    """Request model for single customer churn prediction"""
    
    # Required core features
    monthly_charges: float = Field(
        ..., 
        ge=0, 
        le=10000, 
        description="Monthly charges for the customer ($CAD)"
    )
    total_charges: float = Field(
        ..., 
        ge=0, 
        le=1000000, 
        description="Total charges accumulated ($CAD)"
    )
    contract_length: int = Field(
        ..., 
        ge=1, 
        le=60, 
        description="Contract length in months"
    )
    account_age_days: int = Field(
        ..., 
        ge=0, 
        le=3650, 
        description="Account age in days"
    )
    
    # Optional enhanced features
    province: Optional[ProvinceName] = Field(
        None, 
        description="Canadian province or territory"
    )
    payment_method: Optional[PaymentMethod] = Field(
        None, 
        description="Primary payment method"
    )
    subscription_type: Optional[SubscriptionType] = Field(
        None, 
        description="Subscription tier"
    )
    paperless_billing: Optional[bool] = Field(
        None, 
        description="Uses paperless billing"
    )
    auto_pay: Optional[bool] = Field(
        None, 
        description="Has automatic payment enabled"
    )
    
    # Support and transaction features
    support_tickets: Optional[int] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Number of support tickets"
    )
    satisfaction_score: Optional[float] = Field(
        None, 
        ge=1.0, 
        le=5.0, 
        description="Customer satisfaction score (1-5)"
    )
    transaction_failures: Optional[int] = Field(
        None, 
        ge=0, 
        le=50, 
        description="Number of failed transactions"
    )
    days_since_last_login: Optional[int] = Field(
        None, 
        ge=0, 
        le=365, 
        description="Days since last login"
    )
    
    @validator('total_charges')
    def total_charges_must_be_reasonable(cls, v, values):
        """Validate that total charges are reasonable compared to monthly charges"""
        if 'monthly_charges' in values and v < values['monthly_charges']:
            raise ValueError('Total charges should be at least monthly charges')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "monthly_charges": 75.50,
                "total_charges": 1200.00,
                "contract_length": 12,
                "account_age_days": 365,
                "province": "ON",
                "payment_method": "Credit Card",
                "subscription_type": "Standard",
                "paperless_billing": True,
                "auto_pay": False,
                "support_tickets": 2,
                "satisfaction_score": 4.2,
                "transaction_failures": 0,
                "days_since_last_login": 5
            }
        }


class PredictionResult(BaseModel):
    """Model prediction result"""
    
    churn_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Probability of customer churn (0-1)"
    )
    will_churn: bool = Field(
        ..., 
        description="Binary prediction: will customer churn?"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Model confidence in prediction (0-1)"
    )
    risk_level: RiskLevel = Field(
        ..., 
        description="Risk level category"
    )


class FeatureImportance(BaseModel):
    """Feature importance for explanation"""
    
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Feature importance score")
    impact: str = Field(..., description="Impact direction (Increases/Decreases churn risk)")
    shap_value: Optional[float] = Field(None, description="SHAP value for this feature")


class PredictionExplanation(BaseModel):
    """Explanation for the prediction"""
    
    summary: str = Field(..., description="Human-readable summary of prediction")
    top_factors: List[FeatureImportance] = Field(
        ..., 
        description="Top factors influencing the prediction"
    )
    recommendation: Optional[str] = Field(
        None, 
        description="Actionable recommendation"
    )


class CustomerPredictionResponse(BaseModel):
    """Response model for single customer prediction"""
    
    customer_id: str = Field(..., description="Customer identifier")
    prediction: PredictionResult = Field(..., description="Prediction result")
    explanation: PredictionExplanation = Field(..., description="Prediction explanation")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: Optional[str] = Field(None, description="Model version used")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001234",
                "prediction": {
                    "churn_probability": 0.23,
                    "will_churn": False,
                    "confidence": 0.77,
                    "risk_level": "Low"
                },
                "explanation": {
                    "summary": "Customer has low churn risk (23%). Main retention factors: high satisfaction score, long tenure.",
                    "top_factors": [
                        {
                            "feature": "satisfaction_score",
                            "importance": 0.15,
                            "impact": "Decreases churn risk",
                            "shap_value": -0.45
                        }
                    ],
                    "recommendation": "Continue providing excellent customer service"
                },
                "timestamp": "2024-01-15T10:30:00Z",
                "model_version": "1.0.0"
            }
        }


class BatchCustomerRequest(BaseModel):
    """Single customer in batch request"""
    
    customer_id: str = Field(..., description="Customer identifier")
    
    # Include all fields from CustomerPredictionRequest
    monthly_charges: float = Field(..., ge=0, le=10000)
    total_charges: float = Field(..., ge=0, le=1000000)
    contract_length: int = Field(..., ge=1, le=60)
    account_age_days: int = Field(..., ge=0, le=3650)
    
    province: Optional[ProvinceName] = None
    payment_method: Optional[PaymentMethod] = None
    subscription_type: Optional[SubscriptionType] = None
    paperless_billing: Optional[bool] = None
    auto_pay: Optional[bool] = None
    support_tickets: Optional[int] = Field(None, ge=0, le=100)
    satisfaction_score: Optional[float] = Field(None, ge=1.0, le=5.0)
    transaction_failures: Optional[int] = Field(None, ge=0, le=50)
    days_since_last_login: Optional[int] = Field(None, ge=0, le=365)


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    
    customers: List[BatchCustomerRequest] = Field(
        ..., 
        min_items=1, 
        max_items=1000, 
        description="List of customers for prediction"
    )
    
    @validator('customers')
    def validate_unique_customer_ids(cls, v):
        """Ensure customer IDs are unique"""
        customer_ids = [customer.customer_id for customer in v]
        if len(customer_ids) != len(set(customer_ids)):
            raise ValueError('Customer IDs must be unique')
        return v


class BatchPredictionSummary(BaseModel):
    """Summary statistics for batch prediction"""
    
    total_customers: int = Field(..., description="Total customers processed")
    high_risk_customers: int = Field(..., description="Number of high-risk customers")
    average_churn_probability: float = Field(..., description="Average churn probability")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    
    predictions: List[CustomerPredictionResponse] = Field(
        ..., 
        description="Individual predictions"
    )
    summary: BatchPredictionSummary = Field(..., description="Batch summary")
    timestamp: datetime = Field(..., description="Batch processing timestamp")


class FeedbackRequest(BaseModel):
    """Request model for prediction feedback"""
    
    customer_id: str = Field(..., description="Customer identifier")
    predicted_churn: bool = Field(..., description="What the model predicted")
    actual_churn: bool = Field(..., description="What actually happened")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    comments: Optional[str] = Field(
        None, 
        max_length=1000, 
        description="Additional comments"
    )
    confidence_rating: Optional[int] = Field(
        None, 
        ge=1, 
        le=5, 
        description="How confident are you in this feedback? (1-5)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001234",
                "predicted_churn": False,
                "actual_churn": True,
                "feedback_type": "incorrect_prediction",
                "comments": "Customer churned despite low prediction score",
                "confidence_rating": 5
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    
    status: str = Field(..., description="Feedback processing status")
    customer_id: str = Field(..., description="Customer identifier")
    feedback_id: str = Field(..., description="Unique feedback identifier")
    timestamp: datetime = Field(..., description="Feedback submission timestamp")
    message: str = Field(..., description="Response message")


class ModelInfo(BaseModel):
    """Model information response"""
    
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    algorithm: str = Field(..., description="ML algorithm used")
    last_trained: datetime = Field(..., description="Last training timestamp")
    training_data_size: int = Field(..., description="Training dataset size")
    features: List[str] = Field(..., description="List of feature names")
    performance: Dict[str, float] = Field(..., description="Model performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "Customer Churn Predictor",
                "version": "1.0.0",
                "algorithm": "Random Forest",
                "last_trained": "2024-01-01T00:00:00Z",
                "training_data_size": 10000,
                "features": ["monthly_charges", "total_charges", "contract_length"],
                "performance": {
                    "accuracy": 0.76,
                    "precision": 0.65,
                    "recall": 0.58,
                    "f1_score": 0.61,
                    "auc_score": 0.73
                }
            }
        }


class FeatureImportanceResponse(BaseModel):
    """Feature importance response"""
    
    features: List[FeatureImportance] = Field(
        ..., 
        description="List of features with importance scores"
    )
    model_version: str = Field(..., description="Model version")
    timestamp: datetime = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    database_connected: bool = Field(..., description="Database connection status")
    model_loaded: bool = Field(..., description="Model loading status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")