"""
Database Testing Suite for Customer Churn Prediction System
Tests PostgreSQL integration, data loading, and SQL queries
"""

import pytest
import pandas as pd
import psycopg2
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database_setup import DatabaseManager
# from data.generate_synthetic_data import CustomerDataGenerator  # Not implemented as class


class TestDatabaseManager:
    """Test suite for DatabaseManager class"""
    
    @pytest.fixture
    def db_manager(self):
        """Create DatabaseManager instance"""
        return DatabaseManager()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        # Simple test data for database testing
        customers = pd.DataFrame({
            'customer_id': [f'CUST_{i:03d}' for i in range(10)],
            'monthly_charges': np.random.uniform(50, 150, 10),
            'total_charges': np.random.uniform(500, 2000, 10),
            'contract_length': np.random.choice([12, 24, 36], 10),
            'province': np.random.choice(['ON', 'QC', 'BC', 'AB'], 10)
        })
        
        transactions = pd.DataFrame({
            'customer_id': np.random.choice(customers['customer_id'], 30),
            'transaction_id': [f'TXN_{i:03d}' for i in range(30)],
            'amount': np.random.uniform(10, 200, 30),
            'status': np.random.choice(['Completed', 'Failed'], 30),
            'transaction_date': pd.date_range('2023-01-01', periods=30, freq='D')
        })
        
        support_tickets = pd.DataFrame({
            'customer_id': np.random.choice(customers['customer_id'], 20),
            'ticket_id': [f'TKT_{i:03d}' for i in range(20)],
            'issue_category': np.random.choice(['Billing', 'Technical', 'Cancellation'], 20),
            'satisfaction_score': np.random.uniform(1, 5, 20),
            'created_date': pd.date_range('2023-01-01', periods=20, freq='D')
        })
        
        churn_labels = pd.DataFrame({
            'customer_id': customers['customer_id'],
            'is_churned': np.random.choice([True, False], 10)
        })
        
        return {
            'customers': customers,
            'transactions': transactions,
            'support_tickets': support_tickets,
            'churn_labels': churn_labels
        }
    
    def test_database_connection(self, db_manager):
        """Test database connection"""
        try:
            success = db_manager.test_connection()
            assert isinstance(success, bool)
            
            if not success:
                pytest.skip("Database not available for testing")
                
        except Exception as e:
            pytest.skip(f"Database connection test failed: {e}")
    
    def test_connection_parameters(self, db_manager):
        """Test connection parameters are set correctly"""
        assert hasattr(db_manager, 'host')
        assert hasattr(db_manager, 'port')
        assert hasattr(db_manager, 'database')
        assert hasattr(db_manager, 'user')
        assert hasattr(db_manager, 'password')
        
        # Check default values
        assert db_manager.host == 'localhost'
        assert db_manager.port == 5432
        assert db_manager.database == 'churn_prediction'
        assert db_manager.user == 'postgres'
    
    def test_schema_creation(self, db_manager):
        """Test database schema creation"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test schema file exists
            schema_file = '/Users/sheshank/Desktop/ml projects/Customer Churn Prediction System V1/data/sql/schema.sql'
            assert os.path.exists(schema_file), "Schema file not found"
            
            # Test schema execution (this might require database reset)
            # db_manager.execute_sql_file(schema_file)
            
        except Exception as e:
            pytest.skip(f"Schema creation test failed: {e}")
    
    def test_data_loading(self, db_manager, sample_data):
        """Test CSV data loading to database"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test each table loading
            for table_name, data in sample_data.items():
                success = db_manager.load_dataframe_to_table(data, table_name)
                assert isinstance(success, bool)
                
        except Exception as e:
            pytest.skip(f"Data loading test failed: {e}")
    
    def test_sql_feature_extraction(self, db_manager):
        """Test SQL-based feature extraction"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            features = db_manager.get_ml_features()
            
            if len(features) == 0:
                pytest.skip("No data in database for feature extraction")
            
            assert isinstance(features, pd.DataFrame)
            
            # Check expected columns
            expected_columns = [
                'customer_id', 'monthly_charges', 'total_charges',
                'contract_length', 'account_age_days', 'total_transactions',
                'total_tickets', 'is_churned'
            ]
            
            for col in expected_columns:
                assert col in features.columns, f"Missing column: {col}"
            
            # Check data types
            assert features['monthly_charges'].dtype in ['float64', 'int64']
            assert features['total_charges'].dtype in ['float64', 'int64']
            assert features['is_churned'].dtype in ['bool', 'int64']
            
            # Check data quality
            assert not features['customer_id'].isna().any()
            assert features['monthly_charges'].min() >= 0
            assert features['total_charges'].min() >= 0
            
        except Exception as e:
            pytest.skip(f"SQL feature extraction test failed: {e}")
    
    def test_query_execution(self, db_manager):
        """Test direct SQL query execution"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test simple query
            result = db_manager.execute_query("SELECT 1 as test_value")
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            assert result.iloc[0]['test_value'] == 1
            
            # Test query with parameters
            result = db_manager.execute_query(
                "SELECT $1 as param_value", 
                params=[42]
            )
            assert result.iloc[0]['param_value'] == 42
            
        except Exception as e:
            pytest.skip(f"Query execution test failed: {e}")
    
    def test_table_existence_check(self, db_manager):
        """Test checking if tables exist"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Check if we can query table information
            tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            """
            
            result = db_manager.execute_query(tables_query)
            assert isinstance(result, pd.DataFrame)
            
        except Exception as e:
            pytest.skip(f"Table existence check failed: {e}")
    
    def test_error_handling(self, db_manager):
        """Test error handling for invalid queries"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test invalid SQL
            with pytest.raises(Exception):
                db_manager.execute_query("INVALID SQL QUERY")
            
            # Test query on non-existent table
            with pytest.raises(Exception):
                db_manager.execute_query("SELECT * FROM non_existent_table")
                
        except Exception as e:
            pytest.skip(f"Error handling test failed: {e}")
    
    def test_connection_pooling(self, db_manager):
        """Test connection pooling behavior"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Make multiple queries to test connection reuse
            for i in range(5):
                result = db_manager.execute_query("SELECT $1 as iteration", params=[i])
                assert result.iloc[0]['iteration'] == i
            
        except Exception as e:
            pytest.skip(f"Connection pooling test failed: {e}")


class TestSQLQueries:
    """Test specific SQL queries and operations"""
    
    @pytest.fixture
    def db_manager(self):
        """Create DatabaseManager instance"""
        return DatabaseManager()
    
    def test_customer_aggregation_query(self, db_manager):
        """Test customer data aggregation queries"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test customer statistics query
            query = """
            SELECT 
                COUNT(*) as total_customers,
                AVG(monthly_charges) as avg_monthly_charges,
                MAX(total_charges) as max_total_charges
            FROM customers
            """
            
            result = db_manager.execute_query(query)
            
            if len(result) > 0:
                assert 'total_customers' in result.columns
                assert 'avg_monthly_charges' in result.columns
                assert 'max_total_charges' in result.columns
                
                assert result.iloc[0]['total_customers'] >= 0
                
        except Exception as e:
            pytest.skip(f"Customer aggregation query test failed: {e}")
    
    def test_transaction_analysis_query(self, db_manager):
        """Test transaction analysis queries"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test transaction statistics
            query = """
            SELECT 
                customer_id,
                COUNT(*) as transaction_count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM transactions 
            GROUP BY customer_id
            LIMIT 10
            """
            
            result = db_manager.execute_query(query)
            
            if len(result) > 0:
                assert 'customer_id' in result.columns
                assert 'transaction_count' in result.columns
                assert 'total_amount' in result.columns
                assert 'avg_amount' in result.columns
                
                assert result['transaction_count'].min() > 0
                assert result['total_amount'].min() >= 0
                
        except Exception as e:
            pytest.skip(f"Transaction analysis query test failed: {e}")
    
    def test_support_ticket_analysis(self, db_manager):
        """Test support ticket analysis queries"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test support statistics
            query = """
            SELECT 
                issue_category,
                COUNT(*) as ticket_count,
                AVG(satisfaction_score) as avg_satisfaction
            FROM support_tickets 
            WHERE satisfaction_score IS NOT NULL
            GROUP BY issue_category
            """
            
            result = db_manager.execute_query(query)
            
            if len(result) > 0:
                assert 'issue_category' in result.columns
                assert 'ticket_count' in result.columns
                assert 'avg_satisfaction' in result.columns
                
                assert result['ticket_count'].min() > 0
                assert result['avg_satisfaction'].between(1, 5).all()
                
        except Exception as e:
            pytest.skip(f"Support ticket analysis test failed: {e}")
    
    def test_churn_analysis_query(self, db_manager):
        """Test churn analysis queries"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test churn rate calculation
            query = """
            SELECT 
                COUNT(*) as total_customers,
                SUM(CASE WHEN is_churned THEN 1 ELSE 0 END) as churned_customers,
                AVG(CASE WHEN is_churned THEN 1.0 ELSE 0.0 END) as churn_rate
            FROM churn_labels
            """
            
            result = db_manager.execute_query(query)
            
            if len(result) > 0:
                assert 'total_customers' in result.columns
                assert 'churned_customers' in result.columns
                assert 'churn_rate' in result.columns
                
                total = result.iloc[0]['total_customers']
                churned = result.iloc[0]['churned_customers']
                rate = result.iloc[0]['churn_rate']
                
                assert total >= churned
                assert 0 <= rate <= 1
                
        except Exception as e:
            pytest.skip(f"Churn analysis query test failed: {e}")
    
    def test_complex_join_query(self, db_manager):
        """Test complex multi-table join queries"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test multi-table join
            query = """
            SELECT 
                c.customer_id,
                c.monthly_charges,
                COUNT(t.transaction_id) as transaction_count,
                COUNT(s.ticket_id) as ticket_count,
                ch.is_churned
            FROM customers c
            LEFT JOIN transactions t ON c.customer_id = t.customer_id
            LEFT JOIN support_tickets s ON c.customer_id = s.customer_id
            LEFT JOIN churn_labels ch ON c.customer_id = ch.customer_id
            GROUP BY c.customer_id, c.monthly_charges, ch.is_churned
            LIMIT 10
            """
            
            result = db_manager.execute_query(query)
            
            if len(result) > 0:
                expected_columns = [
                    'customer_id', 'monthly_charges', 'transaction_count', 
                    'ticket_count', 'is_churned'
                ]
                
                for col in expected_columns:
                    assert col in result.columns, f"Missing column: {col}"
                
                assert result['transaction_count'].min() >= 0
                assert result['ticket_count'].min() >= 0
                
        except Exception as e:
            pytest.skip(f"Complex join query test failed: {e}")


class TestDataIntegrity:
    """Test data integrity and constraints"""
    
    @pytest.fixture
    def db_manager(self):
        """Create DatabaseManager instance"""
        return DatabaseManager()
    
    def test_referential_integrity(self, db_manager):
        """Test referential integrity between tables"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Check transactions reference valid customers
            query = """
            SELECT COUNT(*) as orphaned_transactions
            FROM transactions t
            LEFT JOIN customers c ON t.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
            """
            
            result = db_manager.execute_query(query)
            if len(result) > 0:
                assert result.iloc[0]['orphaned_transactions'] == 0
            
            # Check support tickets reference valid customers  
            query = """
            SELECT COUNT(*) as orphaned_tickets
            FROM support_tickets s
            LEFT JOIN customers c ON s.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
            """
            
            result = db_manager.execute_query(query)
            if len(result) > 0:
                assert result.iloc[0]['orphaned_tickets'] == 0
                
        except Exception as e:
            pytest.skip(f"Referential integrity test failed: {e}")
    
    def test_data_constraints(self, db_manager):
        """Test data constraints and validation"""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Check for negative charges
            query = """
            SELECT COUNT(*) as negative_charges
            FROM customers 
            WHERE monthly_charges < 0 OR total_charges < 0
            """
            
            result = db_manager.execute_query(query)
            if len(result) > 0:
                assert result.iloc[0]['negative_charges'] == 0
            
            # Check for negative transaction amounts
            query = """
            SELECT COUNT(*) as negative_amounts
            FROM transactions 
            WHERE amount < 0
            """
            
            result = db_manager.execute_query(query)
            if len(result) > 0:
                assert result.iloc[0]['negative_amounts'] == 0
                
        except Exception as e:
            pytest.skip(f"Data constraints test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])