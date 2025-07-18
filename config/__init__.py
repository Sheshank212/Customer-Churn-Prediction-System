"""
Configuration package for Customer Churn Prediction System
Centralized configuration management with environment variable support
"""

from .settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]