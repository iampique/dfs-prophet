# 🤝 Contributing to DFS Prophet

Thank you for your interest in contributing to DFS Prophet! This document provides guidelines and information for contributors.

## 🎯 **How to Contribute**

### **Types of Contributions**
- 🐛 **Bug Reports**: Report issues and bugs
- 💡 **Feature Requests**: Suggest new features
- 📝 **Documentation**: Improve docs and examples
- 🔧 **Code Contributions**: Submit pull requests
- 🧪 **Testing**: Add tests or improve test coverage
- 🌟 **Showcase**: Create demos and examples

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.9+
- Git
- Docker (for Qdrant)
- UV package manager (recommended)

### **Development Setup**

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/dfs-prophet.git
   cd dfs-prophet
   ```

2. **Set up development environment**
   ```bash
   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

3. **Start services**
   ```bash
   # Start Qdrant
   docker-compose up -d qdrant
   
   # Start the API (in another terminal)
   uv run python -m dfs_prophet.main
   ```

4. **Run tests**
   ```bash
   # Run all tests
   uv run pytest
   
   # Run with coverage
   uv run pytest --cov=src/dfs_prophet
   ```

## 📝 **Code Style Guidelines**

### **Python Code Style**
- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting
- **MyPy**: Type checking

### **Running Code Quality Tools**
```bash
# Format code
uv run black src/ tests/
uv run isort src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
```

### **Pre-commit Hooks**
```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

## 🧪 **Testing Guidelines**

### **Writing Tests**
- Use **pytest** for all tests
- Follow the existing test structure
- Aim for 90%+ code coverage
- Include both unit and integration tests

### **Test Structure**
```
tests/
├── unit/                    # Unit tests
│   ├── test_vector_engine.py
│   ├── test_embedding_generator.py
│   └── test_api.py
├── integration/             # Integration tests
│   ├── test_multi_vector.py
│   └── test_end_to_end.py
└── conftest.py             # Test configuration
```

### **Running Tests**
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/

# Run with coverage
uv run pytest --cov=src/dfs_prophet --cov-report=html
```

## 📚 **Documentation Guidelines**

### **Code Documentation**
- Use **Google-style docstrings** for all functions and classes
- Include **type hints** for all function parameters and return values
- Add **examples** in docstrings for complex functions

### **Example Docstring**
```python
def search_players(
    query: str,
    limit: int = 10,
    score_threshold: float = 0.5
) -> List[Player]:
    """Search for players using vector similarity.
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score (0.0-1.0)
        
    Returns:
        List of players matching the query
        
    Example:
        >>> players = search_players("elite quarterback", limit=5)
        >>> len(players)
        5
    """
```

### **API Documentation**
- Update **OpenAPI schemas** when adding new endpoints
- Include **request/response examples**
- Document **error codes** and **status codes**

## 🔄 **Pull Request Process**

### **Before Submitting**
1. **Ensure tests pass**
   ```bash
   uv run pytest
   ```

2. **Check code quality**
   ```bash
   uv run black src/ tests/
   uv run isort src/ tests/
   uv run ruff check src/ tests/
   uv run mypy src/
   ```

3. **Update documentation**
   - Update README.md if needed
   - Add docstrings for new functions
   - Update API documentation

### **Pull Request Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Test addition/update
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🐛 **Bug Reports**

### **Bug Report Template**
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS, Ubuntu]
- Python Version: [e.g., 3.9.0]
- DFS Prophet Version: [e.g., 1.0.0]

## Additional Information
Screenshots, logs, etc.
```

## 💡 **Feature Requests**

### **Feature Request Template**
```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature is needed

## Proposed Solution
How you think it should be implemented

## Alternatives Considered
Other approaches you've considered

## Additional Information
Any other relevant information
```

## 🏷️ **Issue Labels**

We use the following labels to categorize issues:
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## 📞 **Getting Help**

### **Communication Channels**
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### **Code of Conduct**
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's coding standards

## 🎉 **Recognition**

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

---

**Thank you for contributing to DFS Prophet! 🚀**
