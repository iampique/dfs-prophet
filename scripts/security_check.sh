#!/bin/bash

# DFS Prophet Security Check Script
# Run this before committing to ensure no sensitive information is exposed

set -e

echo "🔒 DFS Prophet Security Check"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}❌ $message${NC}"
    else
        echo -e "${YELLOW}⚠️  $message${NC}"
    fi
}

# Function to check for sensitive patterns
check_patterns() {
    local pattern=$1
    local description=$2
    local found=$(git diff --cached --name-only | xargs grep -l "$pattern" 2>/dev/null || true)
    
    if [ -n "$found" ]; then
        print_status "FAIL" "Found $description in: $found"
        return 1
    else
        print_status "PASS" "No $description found"
        return 0
    fi
}

# Function to check for sensitive files
check_files() {
    local pattern=$1
    local description=$2
    local found=$(git diff --cached --name-only | grep "$pattern" || true)
    
    if [ -n "$found" ]; then
        print_status "FAIL" "Found $description: $found"
        return 1
    else
        print_status "PASS" "No $description found"
        return 0
    fi
}

echo "📋 Checking for sensitive information in staged files..."
echo ""

# Initialize counters
failures=0
warnings=0

# Check for environment files
if check_files "\.env" "environment files"; then
    :
else
    ((failures++))
fi

# Check for credential files
if check_files "credentials\|secrets\|api_keys\|tokens" "credential files"; then
    :
else
    ((failures++))
fi

# Check for certificate files
if check_files "\.pem\|\.key\|\.crt\|\.cer\|\.p12\|\.pfx" "certificate files"; then
    :
else
    ((failures++))
fi

# Check for database files
if check_files "\.db\|\.sqlite" "database files"; then
    :
else
    ((failures++))
fi

# Check for log files
if check_files "\.log" "log files"; then
    :
else
    ((failures++))
fi

# Check for cache files
if check_files "\.cache\|cache/" "cache files"; then
    :
else
    ((failures++))
fi

echo ""
echo "🔍 Checking for hardcoded secrets in code..."
echo ""

# Check for common API key patterns
if check_patterns "AKIA[0-9A-Z]{16}" "AWS access keys"; then
    :
else
    ((failures++))
fi

if check_patterns "AIza[0-9A-Za-z-_]{35}" "Google API keys"; then
    :
else
    ((failures++))
fi

if check_patterns "sk-[0-9a-zA-Z]{48}" "OpenAI API keys"; then
    :
else
    ((failures++))
fi

if check_patterns "pk_[0-9a-zA-Z]{24}" "Stripe public keys"; then
    :
else
    ((failures++))
fi

if check_patterns "sk_[0-9a-zA-Z]{24}" "Stripe secret keys"; then
    :
else
    ((failures++))
fi

# Check for common password patterns
if check_patterns "password.*=.*['\"][^'\"]{8,}['\"]" "hardcoded passwords"; then
    :
else
    ((failures++))
fi

if check_patterns "secret.*=.*['\"][^'\"]{8,}['\"]" "hardcoded secrets"; then
    :
else
    ((failures++))
fi

# Check for database URLs with credentials
if check_patterns "postgresql://[^:]+:[^@]+@" "database URLs with credentials"; then
    :
else
    ((failures++))
fi

if check_patterns "mysql://[^:]+:[^@]+@" "MySQL URLs with credentials"; then
    :
else
    ((failures++))
fi

if check_patterns "mongodb://[^:]+:[^@]+@" "MongoDB URLs with credentials"; then
    :
else
    ((failures++))
fi

echo ""
echo "🔧 Checking for security tools..."
echo ""

# Check if bandit is available
if command_exists bandit; then
    print_status "PASS" "Bandit security linter is available"
    echo "   Running bandit scan on staged files..."
    if git diff --cached --name-only --diff-filter=ACM | grep '\.py$' | xargs bandit -f json -o /tmp/bandit-report.json 2>/dev/null; then
        print_status "PASS" "Bandit scan completed"
    else
        print_status "WARN" "Bandit scan found issues (check /tmp/bandit-report.json)"
        ((warnings++))
    fi
else
    print_status "WARN" "Bandit not installed (pip install bandit)"
    ((warnings++))
fi

# Check if safety is available
if command_exists safety; then
    print_status "PASS" "Safety dependency checker is available"
    echo "   Running safety check..."
    if safety check --json --output /tmp/safety-report.json 2>/dev/null; then
        print_status "PASS" "Safety check completed"
    else
        print_status "WARN" "Safety check found vulnerabilities (check /tmp/safety-report.json)"
        ((warnings++))
    fi
else
    print_status "WARN" "Safety not installed (pip install safety)"
    ((warnings++))
fi

# Check if git-secrets is available
if command_exists git-secrets; then
    print_status "PASS" "Git-secrets is available"
    echo "   Running git-secrets scan..."
    if git secrets --scan 2>/dev/null; then
        print_status "PASS" "Git-secrets scan completed"
    else
        print_status "WARN" "Git-secrets found potential secrets"
        ((warnings++))
    fi
else
    print_status "WARN" "Git-secrets not installed"
    ((warnings++))
fi

echo ""
echo "📊 Security Check Summary"
echo "========================"

if [ $failures -eq 0 ] && [ $warnings -eq 0 ]; then
    echo -e "${GREEN}🎉 All security checks passed!${NC}"
    echo "   Your code is ready to commit."
    exit 0
elif [ $failures -eq 0 ] && [ $warnings -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Security check completed with warnings${NC}"
    echo "   Failures: $failures"
    echo "   Warnings: $warnings"
    echo "   Consider addressing warnings before committing."
    exit 0
else
    echo -e "${RED}🚨 Security check failed!${NC}"
    echo "   Failures: $failures"
    echo "   Warnings: $warnings"
    echo ""
    echo "❌ Please fix the issues above before committing."
    echo ""
    echo "💡 Tips:"
    echo "   - Use environment variables for secrets"
    echo "   - Add sensitive files to .gitignore"
    echo "   - Use Pydantic Settings for configuration"
    echo "   - Check SECURITY.md for best practices"
    exit 1
fi
