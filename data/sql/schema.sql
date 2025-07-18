-- Customer Churn Prediction System Database Schema
-- Enterprise database design for telecom/fintech scenario
-- Advanced SQL schema with normalization and business logic

-- Drop existing tables if they exist
DROP TABLE IF EXISTS churn_labels CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Create customers table with realistic Canadian customer data
CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(25),
    date_of_birth DATE,
    province VARCHAR(2) NOT NULL, -- Canadian provinces
    city VARCHAR(50),
    postal_code VARCHAR(7), -- Canadian postal format
    account_creation_date DATE NOT NULL,
    subscription_type VARCHAR(20) NOT NULL, -- Basic, Premium, Enterprise
    monthly_charges DECIMAL(10,2) NOT NULL,
    total_charges DECIMAL(10,2) DEFAULT 0.00,
    contract_length INTEGER DEFAULT 12, -- months
    payment_method VARCHAR(20) NOT NULL, -- Credit Card, Bank Transfer, PayPal
    paperless_billing BOOLEAN DEFAULT FALSE,
    auto_pay BOOLEAN DEFAULT FALSE,
    referral_code VARCHAR(20),
    last_login_date DATE,
    account_status VARCHAR(20) DEFAULT 'Active', -- Active, Suspended, Closed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create transactions table with realistic financial patterns
CREATE TABLE transactions (
    transaction_id VARCHAR(30) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    transaction_date DATE NOT NULL,
    transaction_type VARCHAR(20) NOT NULL, -- Payment, Refund, Fee, Adjustment
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'CAD',
    payment_method VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL, -- Completed, Pending, Failed, Cancelled
    description TEXT,
    merchant_category VARCHAR(50),
    is_recurring BOOLEAN DEFAULT FALSE,
    failure_reason VARCHAR(100), -- For failed transactions
    processing_fee DECIMAL(10,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Create support_tickets table to track customer service interactions
CREATE TABLE support_tickets (
    ticket_id VARCHAR(30) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    created_date DATE NOT NULL,
    issue_category VARCHAR(50) NOT NULL, -- Billing, Technical, Account, Cancellation
    priority VARCHAR(10) NOT NULL, -- Low, Medium, High, Critical
    status VARCHAR(20) NOT NULL, -- Open, In Progress, Resolved, Closed
    resolution_date DATE,
    satisfaction_score INTEGER, -- 1-5 scale
    agent_id VARCHAR(20),
    channel VARCHAR(20), -- Phone, Email, Chat, Social Media
    first_response_time_hours INTEGER,
    resolution_time_hours INTEGER,
    escalated BOOLEAN DEFAULT FALSE,
    description TEXT,
    resolution_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    CONSTRAINT valid_satisfaction_score CHECK (satisfaction_score IS NULL OR satisfaction_score BETWEEN 1 AND 5)
);

-- Create churn_labels table with business logic for churn definition
CREATE TABLE churn_labels (
    customer_id VARCHAR(20) PRIMARY KEY,
    churn_date DATE,
    is_churned BOOLEAN NOT NULL DEFAULT FALSE,
    churn_reason VARCHAR(100), -- Price, Service Quality, Competition, Technical Issues
    churn_type VARCHAR(20), -- Voluntary, Involuntary
    days_since_last_transaction INTEGER,
    last_transaction_date DATE,
    total_customer_value DECIMAL(10,2),
    customer_lifetime_months INTEGER,
    win_back_eligible BOOLEAN DEFAULT TRUE,
    prediction_date DATE NOT NULL, -- When prediction was made
    actual_churn_date DATE, -- When customer actually churned (if they did)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Create indexes for better query performance (important for ML pipelines)
CREATE INDEX idx_customers_province ON customers(province);
CREATE INDEX idx_customers_subscription_type ON customers(subscription_type);
CREATE INDEX idx_customers_account_creation_date ON customers(account_creation_date);
CREATE INDEX idx_customers_last_login_date ON customers(last_login_date);

CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);
CREATE INDEX idx_transactions_status ON transactions(status);

CREATE INDEX idx_support_tickets_customer_id ON support_tickets(customer_id);
CREATE INDEX idx_support_tickets_created_date ON support_tickets(created_date);
CREATE INDEX idx_support_tickets_category ON support_tickets(issue_category);
CREATE INDEX idx_support_tickets_status ON support_tickets(status);

CREATE INDEX idx_churn_labels_prediction_date ON churn_labels(prediction_date);
CREATE INDEX idx_churn_labels_is_churned ON churn_labels(is_churned);

-- Create views for common business queries (showcases advanced SQL)
CREATE VIEW customer_summary AS
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    c.province,
    c.subscription_type,
    c.monthly_charges,
    c.account_creation_date,
    CURRENT_DATE - c.account_creation_date AS days_as_customer,
    c.last_login_date,
    CURRENT_DATE - c.last_login_date AS days_since_last_login,
    ch.is_churned,
    ch.churn_reason
FROM customers c
LEFT JOIN churn_labels ch ON c.customer_id = ch.customer_id;

CREATE VIEW transaction_summary AS
SELECT 
    t.customer_id,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN t.transaction_type = 'Payment' THEN t.amount ELSE 0 END) as total_payments,
    SUM(CASE WHEN t.transaction_type = 'Refund' THEN t.amount ELSE 0 END) as total_refunds,
    COUNT(CASE WHEN t.status = 'Failed' THEN 1 END) as failed_transactions,
    AVG(t.amount) as avg_transaction_amount,
    MAX(t.transaction_date) as last_transaction_date,
    COUNT(CASE WHEN t.transaction_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as transactions_last_30_days
FROM transactions t
GROUP BY t.customer_id;

CREATE VIEW support_summary AS
SELECT 
    s.customer_id,
    COUNT(*) as total_tickets,
    COUNT(CASE WHEN s.status = 'Open' THEN 1 END) as open_tickets,
    COUNT(CASE WHEN s.issue_category = 'Billing' THEN 1 END) as billing_tickets,
    COUNT(CASE WHEN s.issue_category = 'Technical' THEN 1 END) as technical_tickets,
    COUNT(CASE WHEN s.issue_category = 'Cancellation' THEN 1 END) as cancellation_tickets,
    AVG(s.satisfaction_score) as avg_satisfaction_score,
    AVG(s.resolution_time_hours) as avg_resolution_time,
    COUNT(CASE WHEN s.escalated = TRUE THEN 1 END) as escalated_tickets,
    MAX(s.created_date) as last_ticket_date
FROM support_tickets s
GROUP BY s.customer_id;

-- Add comments for documentation
COMMENT ON TABLE customers IS 'Core customer information with Canadian-specific fields';
COMMENT ON TABLE transactions IS 'Financial transactions with comprehensive status tracking';
COMMENT ON TABLE support_tickets IS 'Customer service interactions with satisfaction metrics';
COMMENT ON TABLE churn_labels IS 'Target variable with business logic for churn definition';

COMMENT ON COLUMN customers.province IS 'Canadian province abbreviation (ON, BC, AB, etc.)';
COMMENT ON COLUMN customers.postal_code IS 'Canadian postal code format (A1A 1A1)';
COMMENT ON COLUMN transactions.currency IS 'Currency code, default CAD for Canadian market';
COMMENT ON COLUMN support_tickets.satisfaction_score IS 'Customer satisfaction rating from 1-5';
COMMENT ON COLUMN churn_labels.churn_type IS 'Voluntary (customer choice) or Involuntary (service termination)';