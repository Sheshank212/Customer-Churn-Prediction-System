"""
Prerequisites Check Script
Run this first to verify your system is ready for testing
"""

import sys
import subprocess
import pkg_resources
import os

# Install missing psycopg2-binary package
print(" Installing psycopg2-binary...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "psycopg2-binary"],
                       capture_output=True, text=True)

if result.returncode == 0:
    print(" psycopg2-binary installed successfully!")
else:
    print(f" Installation failed: {result.stderr}")
    print(" Alternative: pip install psycopg2-binary")
    
    
def check_python_version():
    """Check Python version"""
    print(" Python Version Check:")
    print(f"   Current version: {sys.version}")
    if sys.version_info >= (3, 9):
        print("   Python version OK")
        return True
    else:
        print("    Python 3.9+ required")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    print("\n Required Packages Check:")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'fastapi', 'uvicorn', 'pydantic', 'shap', 'xgboost', 'lightgbm',
        'joblib', 'faker', 'psycopg2-binary', 'prometheus-client'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"    {package}")
        except pkg_resources.DistributionNotFound:
            print(f"    {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    else:
        print("    All required packages installed")
        return True

def check_docker():
    """Check if Docker is available"""
    print("\n Docker Check:")
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"    Docker available: {result.stdout.strip()}")
            return True
        else:
            print("    Docker not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("    Docker not found")
        print("    Docker is optional for basic testing")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    print("\n Project Structure Check:")
    
    required_dirs = [
        'data', 'data/raw', 'data/sql', 'notebooks', 'app', 'app/utils',
        'models', 'figures', 'monitoring', 'monitoring/prometheus',
        'monitoring/grafana', '.github', '.github/workflows'
    ]
    
    required_files = [
        'requirements.txt', 'Dockerfile', 'docker-compose.yml',
        'data/generate_synthetic_data.py', 'data/sql/schema.sql',
        'notebooks/ml_pipeline_visual.py', 'app/main.py',
        'app/utils/shap_explainer.py', 'test_api.py'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"    {dir_path}/")
        else:
            print(f"    {dir_path}/ - MISSING")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"    {file_path}")
        else:
            print(f"    {file_path} - MISSING")
            all_good = False
    
    return all_good

def main():
    """Run all prerequisite checks"""
    print(" Customer Churn Prediction System - Prerequisites Check")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_required_packages(),
        check_docker(),
        check_project_structure()
    ]
    
    print("\n" + "=" * 60)
    if all(checks[:3]):  # Docker is optional
        print(" System ready for testing!")
        print("\n Next steps:")
        print("1. Run: python test_step_by_step.py")
        print("2. Follow the step-by-step testing guide")
    else:
        print("⚠️  Please fix the issues above before proceeding")

if __name__ == "__main__":
    main()