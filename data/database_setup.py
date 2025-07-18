"""
Database Setup and Connection Utilities
Handles PostgreSQL connection and data loading for the Customer Churn Prediction System
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
import os
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles database connections and operations"""
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize database connection
        
        Args:
            connection_params: Dictionary with database connection parameters
                             If None, uses environment variables
        """
        if connection_params is None:
            connection_params = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'churn_prediction'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password')
            }
        
        self.connection_params = connection_params
        self.connection_string = (
            f"postgresql://{connection_params['user']}:{connection_params['password']}"
            f"@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}"
        )
        
    def get_connection(self):
        """Get raw psycopg2 connection"""
        return psycopg2.connect(**self.connection_params)
    
    def get_engine(self):
        """Get SQLAlchemy engine"""
        return create_engine(self.connection_string)
    
    def execute_sql_file(self, file_path: str):
        """Execute SQL file"""
        with open(file_path, 'r') as file:
            sql_content = file.read()
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(sql_content)
                    conn.commit()
                    logger.info(f"Successfully executed SQL file: {file_path}")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error executing SQL file {file_path}: {str(e)}")
                    raise
    
    def execute_query(self, query: str, params: Optional[tuple] = None):
        """Execute a query and return results"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
    
    def load_csv_to_table(self, csv_file_path: str, table_name: str, if_exists: str = 'append'):
        """Load CSV data into database table"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loading {len(df)} records from {csv_file_path} to {table_name}")
            
            # Clear existing data first
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"DELETE FROM {table_name}")
                    conn.commit()
                    logger.info(f"Cleared existing data from {table_name}")
            
            # Load to database
            engine = self.get_engine()
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)
            logger.info(f"Successfully loaded data to {table_name}")
            
        except Exception as e:
            logger.error(f"Error loading CSV to table {table_name}: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str):
        """Get table information"""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
        """
        return self.execute_query(query, (table_name,))
    
    def get_table_count(self, table_name: str):
        """Get table record count"""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_query(query)
        return result[0]['count']
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()[0]
                    logger.info(f"Database connection successful. PostgreSQL version: {version}")
                    return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def get_ml_features(self) -> pd.DataFrame:
        """
        Get ML features using advanced SQL preprocessing
        
        Returns:
            pd.DataFrame: Feature dataset for ML training
        """
        try:
            # Use the advanced SQL feature engineering query
            query = """
            SELECT 
                c.customer_id,
                c.monthly_charges,
                c.total_charges,
                c.contract_length,
                CURRENT_DATE - c.account_creation_date as account_age_days,
                CASE 
                    WHEN c.last_login_date IS NOT NULL 
                    THEN CURRENT_DATE - c.last_login_date 
                    ELSE 999 
                END as days_since_last_login,
                CASE WHEN c.last_login_date IS NULL THEN 1 ELSE 0 END as never_logged_in,
                EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.date_of_birth)) as age,
                
                -- Transaction features
                COALESCE(tm.total_transactions, 0) as total_transactions,
                COALESCE(tm.successful_payments, 0) as total_amount,
                COALESCE(tm.avg_payment_amount, 0) as avg_amount,
                COALESCE(tm.payment_amount_stddev, 0) as std_amount,
                COALESCE(tm.failure_rate, 0) as failure_rate,
                COALESCE(CURRENT_DATE - tm.last_transaction_date, 999) as days_since_last_transaction,
                COALESCE(tm.failed_transactions, 0) as failed_transactions,
                
                -- Support features
                COALESCE(sm.total_support_tickets, 0) as total_tickets,
                COALESCE(sm.avg_satisfaction_score, 3.0) as avg_satisfaction,
                COALESCE(sm.min_satisfaction_score, 3.0) as min_satisfaction,
                COALESCE(sm.avg_resolution_time_hours, 24.0) as avg_resolution_time,
                COALESCE(sm.billing_tickets, 0) as billing_tickets,
                COALESCE(sm.technical_tickets, 0) as technical_tickets,
                COALESCE(sm.cancellation_tickets, 0) as cancellation_tickets,
                
                -- Business metrics
                COALESCE(c.total_charges * 0.1, 0) as customer_value_score,
                COALESCE((tm.failure_rate * 0.3 + COALESCE(sm.escalation_rate, 0) * 0.7), 0.1) as risk_score,
                
                -- Categorical features
                c.subscription_type,
                c.province,
                c.payment_method,
                CASE WHEN c.paperless_billing THEN 1 ELSE 0 END as paperless_billing,
                CASE WHEN c.auto_pay THEN 1 ELSE 0 END as auto_pay,
                
                -- Target variable
                COALESCE(ch.is_churned, false) as is_churned
                
            FROM customers c
            LEFT JOIN (
                SELECT 
                    customer_id,
                    COUNT(*) as total_transactions,
                    SUM(CASE WHEN transaction_type = 'Payment' AND status = 'Completed' THEN amount ELSE 0 END) as successful_payments,
                    COUNT(CASE WHEN status = 'Failed' THEN 1 END) as failed_transactions,
                    COUNT(CASE WHEN status = 'Failed' THEN 1 END)::FLOAT / COUNT(*) as failure_rate,
                    AVG(CASE WHEN transaction_type = 'Payment' THEN amount END) as avg_payment_amount,
                    STDDEV(CASE WHEN transaction_type = 'Payment' THEN amount END) as payment_amount_stddev,
                    MAX(transaction_date) as last_transaction_date
                FROM transactions
                GROUP BY customer_id
            ) tm ON c.customer_id = tm.customer_id
            LEFT JOIN (
                SELECT 
                    customer_id,
                    COUNT(*) as total_support_tickets,
                    COUNT(CASE WHEN issue_category = 'Billing' THEN 1 END) as billing_tickets,
                    COUNT(CASE WHEN issue_category = 'Technical' THEN 1 END) as technical_tickets,
                    COUNT(CASE WHEN issue_category = 'Cancellation' THEN 1 END) as cancellation_tickets,
                    AVG(satisfaction_score) as avg_satisfaction_score,
                    MIN(satisfaction_score) as min_satisfaction_score,
                    AVG(resolution_time_hours) as avg_resolution_time_hours,
                    COUNT(CASE WHEN escalated = TRUE THEN 1 END)::FLOAT / COUNT(*) as escalation_rate
                FROM support_tickets
                GROUP BY customer_id
            ) sm ON c.customer_id = sm.customer_id
            LEFT JOIN churn_labels ch ON c.customer_id = ch.customer_id
            """
            
            engine = self.get_engine()
            df = pd.read_sql(query, engine)
            logger.info(f"Retrieved {len(df)} rows with {len(df.columns)} features from database")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving ML features: {e}")
            raise

def setup_database():
    """Complete database setup process"""
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Test connection
    logger.info("Testing database connection...")
    if not db_manager.test_connection():
        logger.error("Database connection failed. Please check your connection parameters.")
        return False
    
    # Create schema
    logger.info("Creating database schema...")
    schema_path = '/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/data/sql/schema.sql'
    try:
        db_manager.execute_sql_file(schema_path)
    except Exception as e:
        logger.error(f"Failed to create schema: {str(e)}")
        return False
    
    # Load data
    logger.info("Loading synthetic data...")
    data_dir = '/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/data/raw'
    
    tables = ['customers', 'transactions', 'support_tickets', 'churn_labels']
    
    for table in tables:
        csv_file = os.path.join(data_dir, f'{table}.csv')
        if os.path.exists(csv_file):
            try:
                db_manager.load_csv_to_table(csv_file, table)
            except Exception as e:
                logger.error(f"Failed to load {table}: {str(e)}")
                return False
        else:
            logger.warning(f"CSV file not found: {csv_file}")
    
    # Verify data loading
    logger.info("Verifying data loading...")
    for table in tables:
        try:
            count = db_manager.get_table_count(table)
            logger.info(f"Table {table}: {count} records")
        except Exception as e:
            logger.error(f"Error checking table {table}: {str(e)}")
    
    logger.info("Database setup completed successfully!")
    return True

def get_feature_data():
    """Get preprocessed feature data for ML model"""
    db_manager = DatabaseManager()
    
    query = "SELECT * FROM customer_features;"
    
    try:
        # Use pandas to read data directly from database
        engine = db_manager.get_engine()
        df = pd.read_sql(query, engine)
        logger.info(f"Retrieved {len(df)} records for ML model")
        return df
    except Exception as e:
        logger.error(f"Error retrieving feature data: {str(e)}")
        raise

def get_customer_prediction_data(customer_id: str):
    """Get data for a specific customer prediction"""
    db_manager = DatabaseManager()
    
    query = "SELECT * FROM customer_features WHERE customer_id = %s;"
    
    try:
        result = db_manager.execute_query(query, (customer_id,))
        if result:
            return dict(result[0])
        else:
            logger.warning(f"No data found for customer {customer_id}")
            return None
    except Exception as e:
        logger.error(f"Error retrieving customer data: {str(e)}")
        raise

def main():
    """Main function to run database setup"""
    print("Customer Churn Prediction System - Database Setup")
    print("=" * 50)
    
    # Setup database
    success = setup_database()
    
    if success:
        print("\n✓ Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the ML model training script")
        print("2. Start the FastAPI application")
        print("3. Test the prediction endpoints")
    else:
        print("\n✗ Database setup failed!")
        print("Please check the logs and fix any issues.")

if __name__ == "__main__":
    main()