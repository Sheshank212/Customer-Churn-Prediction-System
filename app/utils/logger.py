"""
Advanced Logging Configuration for Customer Churn Prediction System
Provides structured logging with correlation IDs, performance metrics, and security auditing
"""

import logging
import logging.handlers
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
import sys
import os
from pathlib import Path

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
client_ip_var: ContextVar[str] = ContextVar('client_ip', default='')


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record):
        record.request_id = request_id_var.get('')
        record.user_id = user_id_var.get('')
        record.client_ip = client_ip_var.get('')
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation IDs if present
        if hasattr(record, 'request_id') and record.request_id:
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id') and record.user_id:
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'client_ip') and record.client_ip:
            log_entry['client_ip'] = record.client_ip
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
    
    def log_request_timing(self, endpoint: str, method: str, 
                          duration: float, status_code: int = None):
        """Log API request timing"""
        extra_fields = {
            'metric_type': 'request_timing',
            'endpoint': endpoint,
            'http_method': method,
            'duration_ms': round(duration * 1000, 2),
            'status_code': status_code
        }
        
        log_record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.INFO,
            fn='', lno=0, msg=f"Request {method} {endpoint} completed in {duration*1000:.2f}ms",
            args=(), exc_info=None
        )
        log_record.extra_fields = extra_fields
        self.logger.handle(log_record)
    
    def log_ml_operation(self, operation: str, model_name: str, 
                        duration: float, input_size: int = None):
        """Log ML operation timing"""
        extra_fields = {
            'metric_type': 'ml_operation',
            'operation': operation,
            'model_name': model_name,
            'duration_ms': round(duration * 1000, 2),
            'input_size': input_size
        }
        
        log_record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.INFO,
            fn='', lno=0, msg=f"ML operation {operation} on {model_name} completed in {duration*1000:.2f}ms",
            args=(), exc_info=None
        )
        log_record.extra_fields = extra_fields
        self.logger.handle(log_record)
    
    def log_database_query(self, query_type: str, table: str, 
                          duration: float, rows_affected: int = None):
        """Log database query timing"""
        extra_fields = {
            'metric_type': 'database_query',
            'query_type': query_type,
            'table': table,
            'duration_ms': round(duration * 1000, 2),
            'rows_affected': rows_affected
        }
        
        log_record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.INFO,
            fn='', lno=0, msg=f"Database {query_type} on {table} completed in {duration*1000:.2f}ms",
            args=(), exc_info=None
        )
        log_record.extra_fields = extra_fields
        self.logger.handle(log_record)


class SecurityLogger:
    """Logger for security events and auditing"""
    
    def __init__(self, logger_name: str = "security"):
        self.logger = logging.getLogger(logger_name)
    
    def log_authentication_attempt(self, username: str, success: bool, 
                                 ip_address: str, user_agent: str = None):
        """Log authentication attempts"""
        extra_fields = {
            'event_type': 'authentication_attempt',
            'username': username,
            'success': success,
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        
        level = logging.INFO if success else logging.WARNING
        message = f"Authentication {'successful' if success else 'failed'} for {username} from {ip_address}"
        
        log_record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn='', lno=0, msg=message,
            args=(), exc_info=None
        )
        log_record.extra_fields = extra_fields
        self.logger.handle(log_record)
    
    def log_rate_limit_violation(self, ip_address: str, endpoint: str, 
                                attempts: int, time_window: str):
        """Log rate limit violations"""
        extra_fields = {
            'event_type': 'rate_limit_violation',
            'ip_address': ip_address,
            'endpoint': endpoint,
            'attempts': attempts,
            'time_window': time_window
        }
        
        log_record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.WARNING,
            fn='', lno=0, msg=f"Rate limit exceeded for {ip_address} on {endpoint}: {attempts} attempts in {time_window}",
            args=(), exc_info=None
        )
        log_record.extra_fields = extra_fields
        self.logger.handle(log_record)
    
    def log_suspicious_activity(self, activity_type: str, details: Dict[str, Any]):
        """Log suspicious activities"""
        extra_fields = {
            'event_type': 'suspicious_activity',
            'activity_type': activity_type,
            **details
        }
        
        log_record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.WARNING,
            fn='', lno=0, msg=f"Suspicious activity detected: {activity_type}",
            args=(), exc_info=None
        )
        log_record.extra_fields = extra_fields
        self.logger.handle(log_record)


class BusinessLogger:
    """Logger for business events and metrics"""
    
    def __init__(self, logger_name: str = "business"):
        self.logger = logging.getLogger(logger_name)
    
    def log_prediction_request(self, customer_id: str, prediction_result: Dict[str, Any]):
        """Log prediction requests for business analytics"""
        extra_fields = {
            'event_type': 'prediction_request',
            'customer_id': customer_id,
            'churn_probability': prediction_result.get('churn_probability'),
            'will_churn': prediction_result.get('will_churn'),
            'confidence': prediction_result.get('confidence'),
            'risk_level': prediction_result.get('risk_level')
        }
        
        log_record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.INFO,
            fn='', lno=0, msg=f"Churn prediction for customer {customer_id}: {prediction_result.get('churn_probability', 'N/A')}",
            args=(), exc_info=None
        )
        log_record.extra_fields = extra_fields
        self.logger.handle(log_record)
    
    def log_feedback_received(self, customer_id: str, predicted_churn: bool, 
                             actual_churn: bool, feedback_type: str):
        """Log feedback for model improvement tracking"""
        extra_fields = {
            'event_type': 'feedback_received',
            'customer_id': customer_id,
            'predicted_churn': predicted_churn,
            'actual_churn': actual_churn,
            'feedback_type': feedback_type,
            'prediction_accuracy': predicted_churn == actual_churn
        }
        
        log_record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.INFO,
            fn='', lno=0, msg=f"Feedback received for customer {customer_id}: prediction {'correct' if predicted_churn == actual_churn else 'incorrect'}",
            args=(), exc_info=None
        )
        log_record.extra_fields = extra_fields
        self.logger.handle(log_record)


def setup_logging(log_level: str = "INFO", log_dir: str = "logs", 
                 enable_json: bool = True, enable_console: bool = True) -> None:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_json: Enable JSON formatting for structured logs
        enable_console: Enable console output
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if enable_json:
        json_formatter = JSONFormatter()
    
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s [%(request_id)s]',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add correlation ID filter to all handlers
    correlation_filter = CorrelationIdFilter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(correlation_filter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_json:
        # Main application log (JSON format)
        app_handler = logging.handlers.RotatingFileHandler(
            log_path / "app.jsonl",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(json_formatter)
        app_handler.addFilter(correlation_filter)
        root_logger.addHandler(app_handler)
        
        # Error log (JSON format)
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "error.jsonl",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        error_handler.addFilter(correlation_filter)
        root_logger.addHandler(error_handler)
        
        # Performance log
        perf_logger = logging.getLogger("performance")
        perf_handler = logging.handlers.RotatingFileHandler(
            log_path / "performance.jsonl",
            maxBytes=25 * 1024 * 1024,  # 25MB
            backupCount=5
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(json_formatter)
        perf_handler.addFilter(correlation_filter)
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False
        
        # Security log
        sec_logger = logging.getLogger("security")
        sec_handler = logging.handlers.RotatingFileHandler(
            log_path / "security.jsonl",
            maxBytes=25 * 1024 * 1024,  # 25MB
            backupCount=10
        )
        sec_handler.setLevel(logging.INFO)
        sec_handler.setFormatter(json_formatter)
        sec_handler.addFilter(correlation_filter)
        sec_logger.addHandler(sec_handler)
        sec_logger.propagate = False
        
        # Business log
        biz_logger = logging.getLogger("business")
        biz_handler = logging.handlers.RotatingFileHandler(
            log_path / "business.jsonl",
            maxBytes=25 * 1024 * 1024,  # 25MB
            backupCount=10
        )
        biz_handler.setLevel(logging.INFO)
        biz_handler.setFormatter(json_formatter)
        biz_handler.addFilter(correlation_filter)
        biz_logger.addHandler(biz_handler)
        biz_logger.propagate = False
    
    # Configure third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


def set_request_context(request_id: str = None, user_id: str = None, 
                       client_ip: str = None) -> str:
    """
    Set request context for correlation logging
    
    Returns:
        The request ID (generated if not provided)
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if client_ip:
        client_ip_var.set(client_ip)
    
    return request_id


def clear_request_context():
    """Clear request context"""
    request_id_var.set('')
    user_id_var.set('')
    client_ip_var.set('')


class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, logger: PerformanceLogger, operation: str, 
                 category: str = "general", **kwargs):
        self.logger = logger
        self.operation = operation
        self.category = category
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            
            if self.category == "request":
                self.logger.log_request_timing(
                    self.operation, 
                    duration=duration,
                    **self.kwargs
                )
            elif self.category == "ml":
                self.logger.log_ml_operation(
                    self.operation,
                    duration=duration,
                    **self.kwargs
                )
            elif self.category == "database":
                self.logger.log_database_query(
                    self.operation,
                    duration=duration,
                    **self.kwargs
                )


# Initialize loggers
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()
business_logger = BusinessLogger()