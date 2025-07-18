# Deployment Guide

## 🐳 Complete Deployment Options

### Local Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f churn-api

# Stop services
docker-compose down
```

### Production Deployment

#### Option 1: Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml churn-prediction
```

#### Option 2: Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/
```

#### Option 3: Cloud Platforms

##### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Deploy to ECS
aws ecs update-service --cluster churn-prediction --service churn-api --force-new-deployment
```

##### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/churn-prediction
gcloud run deploy --image gcr.io/PROJECT_ID/churn-prediction --platform managed
```

##### Azure Container Instances
```bash
# Deploy to ACI
az container create --resource-group myResourceGroup --name churn-prediction --image myregistry.azurecr.io/churn-prediction:latest
```

## 📈 Monitoring & Observability

### Prometheus Metrics

The system exposes the following metrics:

- `predictions_total`: Total number of predictions made
- `prediction_duration_seconds`: Prediction request latency
- `api_requests_total`: Total API requests by endpoint and status
- `feedback_total`: Feedback received by type

### Grafana Dashboard

The pre-configured dashboard includes:

- **API Performance**: Request volume, response times, error rates
- **Model Metrics**: Prediction distribution, churn rates, feedback analysis
- **System Health**: CPU, memory, database connections
- **Business Insights**: Customer segments, trend analysis

### Alerts

Configured alerts for:
- API downtime or high error rates
- Unusual prediction patterns
- System resource exhaustion
- Database connectivity issues

## 🔧 Troubleshooting

### System health check
```bash
python tests/test_prerequisites.py
```

### API testing
```bash
python tests/test_api.py
```

### Database connection test
```bash
python -c "from data.database_setup import DatabaseManager; print('DB OK' if DatabaseManager().test_connection() else 'DB FAIL')"
```

### Complete system validation
```bash
python tests/test_step_by_step.py
```

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| PostgreSQL connection failed | `docker-compose restart postgres` |
| API not responding | Check port 8000: `lsof -i :8000` |
| Missing model files | Run `python notebooks/ml_pipeline_visual.py` |
| Docker build fails | Check Docker memory: 4GB+ required |

## 📞 Getting Help

1. **System Validation**: Run `python tests/test_step_by_step.py` first
2. **Check Documentation**: Review guides in project root
3. **Search Issues**: Check existing GitHub issues
4. **Create Issue**: Provide test results and system info

**🔗 Quick Health Check**: `curl http://localhost:8000/health`