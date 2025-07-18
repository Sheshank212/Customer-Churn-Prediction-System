#!/usr/bin/env python3
"""
Complete SQL-Integrated Pipeline Setup
Sets up PostgreSQL database and runs ML pipeline with SQL feature engineering
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLPipelineSetup:
    """Complete setup for SQL-integrated ML pipeline"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.sql_dir = self.data_dir / "sql"
        self.raw_data_dir = self.data_dir / "raw"
        
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        logger.info("Checking prerequisites...")
        
        # Check if PostgreSQL is running
        try:
            result = subprocess.run(
                ['psql', '--version'], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                logger.info(f"✓ PostgreSQL found: {result.stdout.strip()}")
            else:
                logger.error("✗ PostgreSQL not found or not in PATH")
                return False
        except FileNotFoundError:
            logger.error("✗ PostgreSQL not found. Please install PostgreSQL first.")
            return False
        
        # Check if required files exist
        required_files = [
            self.sql_dir / "schema.sql",
            self.sql_dir / "preprocessing_queries.sql",
            self.data_dir / "generate_synthetic_data.py",
            self.data_dir / "database_setup.py",
            self.project_root / "notebooks" / "ml_pipeline_with_sql.py"
        ]
        
        for file_path in required_files:
            if file_path.exists():
                logger.info(f"✓ Found: {file_path}")
            else:
                logger.error(f"✗ Missing: {file_path}")
                return False
        
        return True
    
    def generate_synthetic_data(self):
        """Generate synthetic data if not already present"""
        logger.info("Generating synthetic data...")
        
        # Check if data already exists
        data_files = [
            self.raw_data_dir / "customers.csv",
            self.raw_data_dir / "transactions.csv",
            self.raw_data_dir / "support_tickets.csv",
            self.raw_data_dir / "churn_labels.csv"
        ]
        
        if all(f.exists() for f in data_files):
            logger.info("✓ Synthetic data already exists")
            return True
        
        # Generate synthetic data
        try:
            result = subprocess.run([
                sys.executable, 
                str(self.data_dir / "generate_synthetic_data.py")
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✓ Synthetic data generated successfully")
                return True
            else:
                logger.error(f"✗ Data generation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("✗ Data generation timed out")
            return False
        except Exception as e:
            logger.error(f"✗ Data generation failed: {str(e)}")
            return False
    
    def setup_database(self):
        """Setup PostgreSQL database"""
        logger.info("Setting up PostgreSQL database...")
        
        try:
            # Import and run database setup
            from data.database_setup import setup_database
            
            success = setup_database()
            if success:
                logger.info("✓ Database setup completed successfully")
                return True
            else:
                logger.error("✗ Database setup failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Database setup failed: {str(e)}")
            return False
    
    def run_sql_ml_pipeline(self):
        """Run the SQL-integrated ML pipeline"""
        logger.info("Running SQL-integrated ML pipeline...")
        
        try:
            # Run the SQL ML pipeline
            result = subprocess.run([
                sys.executable, 
                str(self.project_root / "notebooks" / "ml_pipeline_with_sql.py")
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✓ SQL ML pipeline completed successfully")
                logger.info("Pipeline output:")
                print(result.stdout)
                return True
            else:
                logger.error(f"✗ SQL ML pipeline failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("✗ SQL ML pipeline timed out")
            return False
        except Exception as e:
            logger.error(f"✗ SQL ML pipeline failed: {str(e)}")
            return False
    
    def verify_setup(self):
        """Verify that everything is set up correctly"""
        logger.info("Verifying setup...")
        
        # Check if model files were created
        model_files = [
            self.project_root / "models" / "churn_prediction_model_sql.pkl",
            self.project_root / "models" / "scaler_sql.pkl",
            self.project_root / "models" / "label_encoders_sql.pkl",
            self.project_root / "models" / "feature_names_sql.pkl"
        ]
        
        for model_file in model_files:
            if model_file.exists():
                logger.info(f"✓ Model file created: {model_file}")
            else:
                logger.warning(f"⚠ Model file missing: {model_file}")
        
        # Check if SQL visualizations were created
        sql_figures = [
            self.project_root / "figures" / "sql_model_comparison.png",
            self.project_root / "figures" / "sql_feature_importance.png",
            self.project_root / "figures" / "sql_roc_curves.png",
            self.project_root / "figures" / "sql_confusion_matrix.png"
        ]
        
        for figure in sql_figures:
            if figure.exists():
                logger.info(f"✓ SQL visualization created: {figure}")
            else:
                logger.warning(f"⚠ SQL visualization missing: {figure}")
        
        return True
    
    def run_complete_setup(self):
        """Run the complete setup process"""
        logger.info("="*60)
        logger.info("CUSTOMER CHURN PREDICTION - SQL INTEGRATION SETUP")
        logger.info("="*60)
        
        steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Generate Synthetic Data", self.generate_synthetic_data),
            ("Setup Database", self.setup_database),
            ("Run SQL ML Pipeline", self.run_sql_ml_pipeline),
            ("Verify Setup", self.verify_setup)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n--- {step_name} ---")
            try:
                success = step_func()
                if not success:
                    logger.error(f"❌ {step_name} failed!")
                    return False
                logger.info(f"✅ {step_name} completed successfully!")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with error: {str(e)}")
                return False
        
        logger.info("\n" + "="*60)
        logger.info("🎉 SQL INTEGRATION SETUP COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("Your Customer Churn Prediction System now uses:")
        logger.info("✓ PostgreSQL for data storage")
        logger.info("✓ Advanced SQL for feature engineering")
        logger.info("✓ Integrated ML pipeline with SQL features")
        logger.info("✓ SQL-based visualizations")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review the SQL visualizations in figures/")
        logger.info("2. Test the API with SQL-based predictions")
        logger.info("3. Explore the database features using SQL queries")
        logger.info("4. Update your documentation to highlight SQL integration")
        
        return True

def main():
    """Main function"""
    setup = SQLPipelineSetup()
    success = setup.run_complete_setup()
    
    if not success:
        logger.error("Setup failed. Please review the errors above.")
        sys.exit(1)
    
    logger.info("Setup completed successfully!")

if __name__ == "__main__":
    main()