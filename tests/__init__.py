"""
Test package for Customer Churn Prediction System
Comprehensive test suite covering API, ML pipeline, database, and integration testing
"""

__version__ = "1.0.0"
__author__ = "Customer Churn Prediction System"

# Test configuration
import pytest
import os
import sys
from pathlib import Path

# Add project root to path for all tests
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test markers
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]

# Common test fixtures and utilities
@pytest.fixture(scope="session")
def project_root():
    """Project root directory"""
    return PROJECT_ROOT

@pytest.fixture(scope="session") 
def data_dir():
    """Data directory"""
    return PROJECT_ROOT / "data"

@pytest.fixture(scope="session")
def models_dir():
    """Models directory"""
    return PROJECT_ROOT / "models"

@pytest.fixture(scope="session")
def figures_dir():
    """Figures directory"""
    return PROJECT_ROOT / "figures"

# Test categories
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests"
    )
    config.addinivalue_line(
        "markers", "database: Database tests"
    )
    config.addinivalue_line(
        "markers", "api: API tests"
    )
    config.addinivalue_line(
        "markers", "ml: Machine learning tests"
    )