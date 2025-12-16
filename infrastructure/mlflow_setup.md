# Remote MLflow Infrastructure Setup Guide

## Overview
This guide documents the setup of a remote MLflow tracking server on AWS EC2 with PostgreSQL backend and S3 artifact storage.

## Architecture

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────┐
│   Local     │         │   AWS EC2        │         │   AWS S3    │
│   Training  │────────▶│   MLflow Server  │────────▶│  Artifacts  │
│   Scripts   │         │   (Port 5000)    │         │   Storage   │
└─────────────┘         └──────────────────┘         └─────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   PostgreSQL     │
                        │   (Neon.tech)    │
                        │   Backend Store  │
                        └──────────────────┘
```

## Prerequisites

- AWS Account with EC2 and S3 access
- PostgreSQL database (Neon.tech free tier recommended)
- SSH key pair for EC2 access

## Step 1: PostgreSQL Backend Setup (Neon.tech)

### 1.1 Create Neon Database

1. Go to [https://neon.tech](https://neon.tech)
2. Sign up for free account
3. Create a new project: `mlops-team10-mlflow`
4. Note the connection string:
   ```
   postgresql://[user]:[password]@[host]/[database]
   ```

### 1.2 Test Connection

```bash
# Install psycopg2 locally
pip install psycopg2-binary

# Test connection
python -c "import psycopg2; conn = psycopg2.connect('postgresql://...')"
```

## Step 2: AWS S3 Bucket Setup

### 2.1 Create S3 Bucket

```bash
# Using AWS CLI
aws s3 mb s3://mlops-team10-mlflow-artifacts --region us-east-1
```

Or via AWS Console:
1. Navigate to S3
2. Create bucket: `mlops-team10-mlflow-artifacts`
3. Region: `us-east-1`
4. Block all public access: ✓
5. Versioning: Enabled (optional)

### 2.2 Create IAM User for S3 Access

```bash
# Create IAM user
aws iam create-user --user-name mlflow-s3-user

# Attach S3 policy
aws iam attach-user-policy --user-name mlflow-s3-user \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create access keys
aws iam create-access-key --user-name mlflow-s3-user
```

Save the `AccessKeyId` and `SecretAccessKey`.

## Step 3: AWS EC2 Instance Setup

### 3.1 Launch EC2 Instance

1. **AMI**: Ubuntu Server 22.04 LTS
2. **Instance Type**: t2.small (or t2.micro for testing)
3. **Key Pair**: Create or use existing
4. **Security Group**: Create new with rules:
   - SSH (22): Your IP
   - Custom TCP (5000): 0.0.0.0/0 (or your IP for security)

### 3.2 Connect to EC2

```bash
ssh -i "your-key.pem" ubuntu@<EC2_PUBLIC_IP>
```

### 3.3 Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Create virtual environment
python3 -m venv mlflow-env
source mlflow-env/bin/activate

# Install MLflow and dependencies
pip install mlflow psycopg2-binary boto3
```

### 3.4 Configure AWS Credentials

```bash
# Install AWS CLI
sudo apt install awscli -y

# Configure credentials
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: us-east-1
# - Default output format: json
```

### 3.5 Start MLflow Server

```bash
# Set environment variables
export BACKEND_STORE_URI="postgresql://[user]:[password]@[host]/[database]"
export ARTIFACT_ROOT="s3://mlops-team10-mlflow-artifacts/mlflow"

# Start MLflow server
mlflow server \
  --backend-store-uri $BACKEND_STORE_URI \
  --default-artifact-root $ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000
```

### 3.6 Run as Background Service (Optional)

Create systemd service file:

```bash
sudo nano /etc/systemd/system/mlflow.service
```

Content:
```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="PATH=/home/ubuntu/mlflow-env/bin"
Environment="BACKEND_STORE_URI=postgresql://[user]:[password]@[host]/[database]"
Environment="ARTIFACT_ROOT=s3://mlops-team10-mlflow-artifacts/mlflow"
ExecStart=/home/ubuntu/mlflow-env/bin/mlflow server \
  --backend-store-uri $BACKEND_STORE_URI \
  --default-artifact-root $ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
sudo systemctl status mlflow
```

## Step 4: Local Configuration

### 4.1 Update MLflow Config

Edit `src/mlflow_config.py`:

```python
# Remote tracking (after AWS EC2 setup)
MLFLOW_TRACKING_URI = "http://<EC2_PUBLIC_IP>:5000"
```

### 4.2 Set AWS Credentials Locally

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

Or add to `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your-access-key
aws_secret_access_key = your-secret-key
region = us-east-1
```

### 4.3 Test Connection

```python
import mlflow

mlflow.set_tracking_uri("http://<EC2_PUBLIC_IP>:5000")
mlflow.set_experiment("test_experiment")

with mlflow.start_run():
    mlflow.log_param("test", "value")
    mlflow.log_metric("metric", 1.0)
```

## Step 5: Verification

### 5.1 Access MLflow UI

Navigate to: `http://<EC2_PUBLIC_IP>:5000`

You should see:
- Experiments list
- Runs with metrics and parameters
- Registered models

### 5.2 Verify S3 Artifacts

```bash
aws s3 ls s3://mlops-team10-mlflow-artifacts/mlflow/ --recursive
```

### 5.3 Verify PostgreSQL

```bash
# Connect to Neon database
psql "postgresql://[user]:[password]@[host]/[database]"

# List tables
\dt

# You should see MLflow tables:
# - experiments
# - runs
# - metrics
# - params
# - tags
```

## Troubleshooting

### Issue: Cannot connect to MLflow server

**Solution**: Check EC2 security group allows inbound traffic on port 5000

### Issue: S3 access denied

**Solution**: Verify IAM user has S3 permissions and credentials are correct

### Issue: PostgreSQL connection error

**Solution**: Check Neon database is running and connection string is correct

## Security Best Practices

1. **Restrict EC2 Access**: Limit port 5000 to specific IPs
2. **Use VPC**: Deploy EC2 in private subnet with VPN access
3. **Rotate Credentials**: Regularly rotate AWS access keys
4. **Enable SSL**: Use HTTPS for MLflow server (requires certificate)
5. **Database Security**: Use strong passwords for PostgreSQL

## Cost Optimization

- **EC2**: Use t2.micro (free tier) or stop instance when not in use
- **S3**: Enable lifecycle policies to archive old artifacts
- **Neon**: Free tier includes 0.5 GB storage (sufficient for this project)

## Estimated Monthly Costs

- EC2 t2.small (24/7): ~$17/month
- S3 storage (1 GB): ~$0.02/month
- Neon PostgreSQL: Free tier
- **Total**: ~$17/month

## Next Steps

1. Update all training scripts to use remote tracking URI
2. Retrain models with remote tracking
3. Verify artifacts are stored in S3
4. Set up monitoring and alerts
