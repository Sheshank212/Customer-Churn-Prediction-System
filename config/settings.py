"""
Configuration Management for Customer Churn Prediction System
Centralized configuration with environment variable support
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="churn_prediction", env="DB_NAME")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="password", env="DB_PASSWORD")
    
    # Connection pool settings
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    # SSL settings
    ssl_mode: str = Field(default="prefer", env="DB_SSL_MODE")
    ssl_cert: Optional[str] = Field(default=None, env="DB_SSL_CERT")
    ssl_key: Optional[str] = Field(default=None, env="DB_SSL_KEY")
    ssl_ca: Optional[str] = Field(default=None, env="DB_SSL_CA")
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


class MLSettings(BaseSettings):
    """Machine Learning configuration"""
    
    # Model paths (relative to project root)
    model_path: str = Field(default="models/churn_prediction_model.pkl", env="ML_MODEL_PATH")
    feature_names_path: str = Field(default="models/feature_names.pkl", env="ML_FEATURE_NAMES_PATH")
    preprocessing_info_path: str = Field(default="models/preprocessing_info.pkl", env="ML_PREPROCESSING_INFO_PATH")
    shap_explainer_path: str = Field(default="models/shap_explainer.pkl", env="ML_SHAP_EXPLAINER_PATH")
    
    # Model configuration
    model_version: str = Field(default="1.0.0", env="ML_MODEL_VERSION")
    prediction_threshold: float = Field(default=0.5, env="ML_PREDICTION_THRESHOLD")
    confidence_threshold: float = Field(default=0.7, env="ML_CONFIDENCE_THRESHOLD")
    
    # Feature engineering
    enable_sql_features: bool = Field(default=True, env="ML_ENABLE_SQL_FEATURES")
    feature_cache_ttl: int = Field(default=3600, env="ML_FEATURE_CACHE_TTL")  # seconds
    
    # SHAP configuration
    enable_shap_explanations: bool = Field(default=True, env="ML_ENABLE_SHAP")
    shap_sample_size: int = Field(default=100, env="ML_SHAP_SAMPLE_SIZE")
    max_shap_features: int = Field(default=10, env="ML_MAX_SHAP_FEATURES")
    
    @validator('prediction_threshold', 'confidence_threshold')
    def validate_thresholds(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        return v
    
    class Config:
        env_prefix = "ML_"
        case_sensitive = False


class APISettings(BaseSettings):
    """API configuration"""
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    reload: bool = Field(default=False, env="API_RELOAD")
    
    # API metadata
    title: str = Field(default="Customer Churn Prediction API", env="API_TITLE")
    description: str = Field(default="ML-powered customer churn prediction with explanations", env="API_DESCRIPTION")
    version: str = Field(default="1.0.0", env="API_VERSION")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8080", 
            "http://localhost:8000"
        ],
        env="API_CORS_ORIGINS"
    )
    cors_credentials: bool = Field(default=True, env="API_CORS_CREDENTIALS")
    cors_methods: List[str] = Field(default=["GET", "POST", "OPTIONS"], env="API_CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="API_CORS_HEADERS")
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, env="API_ENABLE_RATE_LIMITING")
    rate_limit_requests_per_minute: int = Field(default=60, env="API_RATE_LIMIT_RPM")
    rate_limit_requests_per_hour: int = Field(default=1000, env="API_RATE_LIMIT_RPH")
    rate_limit_burst_capacity: int = Field(default=10, env="API_RATE_LIMIT_BURST")
    
    # Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    enable_api_key_auth: bool = Field(default=False, env="API_ENABLE_API_KEY_AUTH")
    valid_api_keys: List[str] = Field(default=[], env="API_VALID_KEYS")
    
    # Request limits
    max_batch_size: int = Field(default=1000, env="API_MAX_BATCH_SIZE")
    request_timeout_seconds: int = Field(default=30, env="API_REQUEST_TIMEOUT")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('cors_methods', pre=True)
    def parse_cors_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(',')]
        return v
    
    @validator('valid_api_keys', pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(',') if key.strip()]
        return v
    
    class Config:
        env_prefix = "API_"
        case_sensitive = False


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    
    # Log levels
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    root_log_level: str = Field(default="WARNING", env="ROOT_LOG_LEVEL")
    
    # Log output
    log_dir: str = Field(default="logs", env="LOG_DIR")
    enable_console_output: bool = Field(default=True, env="LOG_ENABLE_CONSOLE")
    enable_file_output: bool = Field(default=True, env="LOG_ENABLE_FILE")
    enable_json_format: bool = Field(default=True, env="LOG_ENABLE_JSON")
    
    # Log rotation
    max_log_file_size_mb: int = Field(default=50, env="LOG_MAX_FILE_SIZE_MB")
    max_log_backup_count: int = Field(default=10, env="LOG_MAX_BACKUP_COUNT")
    
    # Structured logging
    enable_correlation_ids: bool = Field(default=True, env="LOG_ENABLE_CORRELATION_IDS")
    log_sql_queries: bool = Field(default=False, env="LOG_SQL_QUERIES")
    log_predictions: bool = Field(default=True, env="LOG_PREDICTIONS")
    
    @validator('log_level', 'root_log_level')
    def validate_log_levels(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""
    
    # Prometheus metrics
    enable_prometheus: bool = Field(default=True, env="MONITORING_ENABLE_PROMETHEUS")
    prometheus_endpoint: str = Field(default="/metrics", env="MONITORING_PROMETHEUS_ENDPOINT")
    
    # Health checks
    health_check_endpoint: str = Field(default="/health", env="MONITORING_HEALTH_ENDPOINT")
    database_health_check: bool = Field(default=True, env="MONITORING_DB_HEALTH_CHECK")
    model_health_check: bool = Field(default=True, env="MONITORING_MODEL_HEALTH_CHECK")
    
    # Performance monitoring
    enable_performance_logging: bool = Field(default=True, env="MONITORING_ENABLE_PERFORMANCE")
    slow_query_threshold_ms: int = Field(default=1000, env="MONITORING_SLOW_QUERY_THRESHOLD_MS")
    slow_request_threshold_ms: int = Field(default=5000, env="MONITORING_SLOW_REQUEST_THRESHOLD_MS")
    
    # Business metrics
    enable_business_metrics: bool = Field(default=True, env="MONITORING_ENABLE_BUSINESS")
    track_prediction_accuracy: bool = Field(default=True, env="MONITORING_TRACK_ACCURACY")
    
    class Config:
        env_prefix = "MONITORING_"
        case_sensitive = False


class SecuritySettings(BaseSettings):
    """Security configuration"""
    
    # Encryption
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECURITY_SECRET_KEY")
    algorithm: str = Field(default="HS256", env="SECURITY_ALGORITHM")
    
    # Session management
    session_timeout_minutes: int = Field(default=60, env="SECURITY_SESSION_TIMEOUT")
    
    # Input validation
    max_request_size_mb: int = Field(default=10, env="SECURITY_MAX_REQUEST_SIZE_MB")
    enable_input_sanitization: bool = Field(default=True, env="SECURITY_ENABLE_INPUT_SANITIZATION")
    
    # IP filtering
    enable_ip_filtering: bool = Field(default=False, env="SECURITY_ENABLE_IP_FILTERING")
    allowed_ips: List[str] = Field(default=[], env="SECURITY_ALLOWED_IPS")
    blocked_ips: List[str] = Field(default=[], env="SECURITY_BLOCKED_IPS")
    
    @validator('allowed_ips', 'blocked_ips', pre=True)
    def parse_ip_lists(cls, v):
        if isinstance(v, str):
            return [ip.strip() for ip in v.split(',') if ip.strip()]
        return v
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class Settings(BaseSettings):
    """Main application settings"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v.lower()
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    def get_model_path(self, filename: str) -> Path:
        """Get absolute path for model file"""
        if filename.startswith('/'):
            return Path(filename)
        return self.models_dir / filename
    
    def get_data_path(self, filename: str) -> Path:
        """Get absolute path for data file"""
        if filename.startswith('/'):
            return Path(filename)
        return self.data_dir / filename
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# For backward compatibility
settings = get_settings()