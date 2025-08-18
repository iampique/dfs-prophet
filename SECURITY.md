# Security Guide

This document outlines security best practices and guidelines for the DFS Prophet project.

## 🔒 Pre-Commit Security Checklist

Before committing code, ensure you've completed this security checklist:

### ✅ Environment Variables
- [ ] No `.env` files are being committed
- [ ] No hardcoded API keys in source code
- [ ] No database credentials in code
- [ ] No Qdrant API keys in code
- [ ] No AWS/Cloud credentials in code

### ✅ Secrets and Credentials
- [ ] No private keys or certificates
- [ ] No SSH keys
- [ ] No access tokens
- [ ] No passwords in plain text
- [ ] No API secrets

### ✅ Data and Storage
- [ ] No real user data
- [ ] No production database dumps
- [ ] No log files with sensitive info
- [ ] No cache files with credentials
- [ ] No temporary files with secrets

### ✅ Configuration Files
- [ ] No production configs
- [ ] No staging environment files
- [ ] No local development secrets
- [ ] No override files with credentials

## 🚨 Common Security Mistakes to Avoid

### ❌ Never Commit These Files:
```bash
# Environment files (but .env.example SHOULD be committed)
.env
.env.local
.env.production
.env.staging

# Credentials
secrets.json
credentials.yaml
api_keys.txt
private_key.pem

# Database files
*.db
*.sqlite
*.sqlite3

# Log files
*.log
logs/

# Cache files
.cache/
*.cache

# Temporary files
*.tmp
*.temp
```

### ✅ Safe to Commit:
```bash
# Template files (safe to commit)
.env.example
config.example.yaml
secrets.example.json

# Documentation
README.md
SECURITY.md
CONTRIBUTING.md

# Source code
src/
tests/
scripts/

# Configuration files
pyproject.toml
docker-compose.yml
Dockerfile
```

### ❌ Never Hardcode These in Source Code:
```python
# API Keys
QDRANT_API_KEY = "your-actual-key-here"
AWS_ACCESS_KEY = "AKIA..."
GOOGLE_API_KEY = "AIza..."

# Database Credentials
DATABASE_URL = "postgresql://user:password@host:port/db"
REDIS_PASSWORD = "your-redis-password"

# Secrets
JWT_SECRET = "your-jwt-secret"
ENCRYPTION_KEY = "your-encryption-key"
```

## ✅ Security Best Practices

### 1. Environment Variables
Use environment variables for all sensitive configuration:

```python
# ✅ Good
import os
from dfs_prophet.config import get_settings

settings = get_settings()
qdrant_url = settings.qdrant.url
api_key = settings.qdrant.api_key

# ❌ Bad
qdrant_url = "http://localhost:6333"
api_key = "your-actual-key"
```

### 2. Configuration Management
Use Pydantic Settings for type-safe configuration:

```python
# ✅ Good
from pydantic_settings import BaseSettings

class QdrantSettings(BaseSettings):
    url: str = "http://localhost:6333"
    api_key: str = ""
    
    class Config:
        env_prefix = "QDRANT_"
        env_file = ".env"

# ❌ Bad
qdrant_config = {
    "url": "http://localhost:6333",
    "api_key": "hardcoded-key"
}
```

### 3. Secrets Management
Use proper secrets management:

```python
# ✅ Good - Use environment variables
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ✅ Good - Use secrets manager in production
import boto3
secretsmanager = boto3.client('secretsmanager')
secret = secretsmanager.get_secret_value(SecretId='dfs-prophet-secrets')

# ❌ Bad - Hardcoded secrets
QDRANT_API_KEY = "your-actual-secret-key"
```

## 🔍 Security Scanning

### Pre-Commit Security Checks
```bash
# Run security scan before committing
bandit -r src/ -f json -o security-report.json

# Check for secrets in code
git secrets --scan

# Run safety check for dependencies
safety check
```

### Automated Security Checks
The CI/CD pipeline includes:
- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability scanner
- **Secret scanning**: Detects hardcoded secrets
- **Dependency scanning**: Checks for known vulnerabilities

## 🛡️ Security Tools

### 1. Pre-commit Hooks
Install pre-commit hooks to automatically check for security issues:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### 2. Git Secrets
Install git-secrets to prevent committing secrets:

```bash
# Install git-secrets
git secrets --install
git secrets --register-aws

# Scan repository
git secrets --scan
```

### 3. TruffleHog
Scan for secrets in git history:

```bash
# Install trufflehog
pip install trufflehog

# Scan repository
trufflehog --regex --entropy=False .
```

## 🚨 Incident Response

### If You Accidentally Commit Sensitive Data:

1. **Immediate Actions:**
   ```bash
   # Remove the file from git history
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch path/to/sensitive/file' \
     --prune-empty --tag-name-filter cat -- --all
   
   # Force push to remove from remote
   git push origin --force --all
   ```

2. **Rotate Credentials:**
   - Change all exposed API keys
   - Rotate database passwords
   - Update access tokens
   - Regenerate SSH keys if exposed

3. **Notify Team:**
   - Inform maintainers immediately
   - Document the incident
   - Update security procedures

## 📋 Security Checklist for New Features

When adding new features, ensure:

- [ ] No hardcoded credentials
- [ ] Environment variables for configuration
- [ ] Input validation and sanitization
- [ ] Error handling without information disclosure
- [ ] Proper authentication and authorization
- [ ] Secure communication (HTTPS/TLS)
- [ ] Rate limiting for API endpoints
- [ ] Logging without sensitive data
- [ ] Security headers in responses
- [ ] CORS configuration
- [ ] SQL injection prevention
- [ ] XSS protection
- [ ] CSRF protection

## 🔐 Production Security

### Environment Security
- Use secrets management services (AWS Secrets Manager, HashiCorp Vault)
- Enable encryption at rest and in transit
- Use least privilege access principles
- Regular security audits and penetration testing

### API Security
- Implement proper authentication (JWT, OAuth2)
- Use HTTPS in production
- Implement rate limiting
- Add request/response validation
- Monitor for suspicious activity

### Data Security
- Encrypt sensitive data
- Implement data retention policies
- Regular backups with encryption
- Access logging and monitoring

## 📞 Security Contacts

- **Security Issues**: security@dfsprophet.com
- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/dfs-prophet/issues)
- **Responsible Disclosure**: Please report security vulnerabilities privately

## 📚 Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python-security.readthedocs.io/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [GitHub Security](https://docs.github.com/en/github/managing-security-vulnerabilities)

---

**Remember**: Security is everyone's responsibility. When in doubt, ask before committing!
