# ML Pipeline Guide

This project contains multiple ML pipeline implementations for different use cases:

## 📁 Pipeline Files Overview

### 1. **notebooks/ml_pipeline_visual.py** 
- **Purpose**: Main comprehensive ML pipeline with visualizations
- **Features**: Full feature engineering, multi-algorithm comparison, 12 visualizations
- **Usage**: Local development, complete analysis
- **Dependencies**: Full requirements.txt (XGBoost, SHAP, etc.)

### 2. **notebooks/ml_pipeline_with_sql.py**
- **Purpose**: SQL-integrated ML pipeline
- **Features**: PostgreSQL feature engineering, SQL-optimized models
- **Usage**: Database-driven ML workflows
- **Dependencies**: PostgreSQL, SQLAlchemy, psycopg2

### 3. **notebooks/ml_pipeline.py**
- **Purpose**: Basic ML pipeline without visualizations
- **Features**: Core ML training, model evaluation
- **Usage**: Lightweight training, CI/CD integration
- **Dependencies**: Basic ML packages only

## 🚀 Recommended Usage

### For CI/CD Pipeline
- The GitHub Actions workflow uses embedded Python scripts
- Focuses on core functionality with minimal dependencies
- Demonstrates basic RandomForest training

### For Local Development
```bash
# Full pipeline with visualizations
python notebooks/ml_pipeline_visual.py

# SQL-integrated pipeline
python notebooks/ml_pipeline_with_sql.py

# Basic pipeline
python notebooks/ml_pipeline.py
```

### For Production
- Use `ml_pipeline_with_sql.py` for database integration
- Use `ml_pipeline_visual.py` for comprehensive analysis
- Modify based on specific requirements

## 🔧 File Consolidation

These files share similar core functionality but serve different purposes:
- **ml_pipeline_visual.py**: Full-featured analysis
- **ml_pipeline_with_sql.py**: Database-focused
- **ml_pipeline.py**: Minimal implementation

Consider consolidating into a single configurable pipeline for production use.