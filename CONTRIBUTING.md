# Contributing to DFS Prophet

Thank you for your interest in contributing to DFS Prophet! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome contributions in the following areas:

- **🐛 Bug Reports**: Help us identify and fix issues
- **✨ Feature Requests**: Suggest new features and improvements
- **📝 Documentation**: Improve docs, examples, and guides
- **🧪 Tests**: Add or improve test coverage
- **🔧 Code**: Submit bug fixes and new features
- **🎨 UI/UX**: Improve user experience and interface
- **📊 Performance**: Optimize code and improve benchmarks

### Before You Start

1. **Check existing issues**: Search for similar issues or feature requests
2. **Read the docs**: Familiarize yourself with the project structure
3. **Join discussions**: Participate in GitHub Discussions
4. **Set up development environment**: Follow the setup guide below

## 🛠️ Development Setup

### Prerequisites

- Python 3.11+
- Docker
- UV Package Manager
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/dfs-prophet.git
cd dfs-prophet

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Generate demo data
python scripts/generate_synthetic_data.py
```

### Code Quality Tools

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/dfs_prophet --cov-report=html
```

## 📋 Contribution Guidelines

### Code Style

- **Python**: Follow PEP 8 with Black formatting
- **Type Hints**: Use type hints for all functions and methods
- **Docstrings**: Use Google-style docstrings
- **Imports**: Use absolute imports, organized with isort
- **Line Length**: 88 characters (Black default)

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

Examples:
```
feat(api): add batch search endpoint
fix(vector): resolve memory leak in binary quantization
docs(readme): update installation instructions
test(embedding): add unit tests for BGE model
```

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**:
   ```bash
   # Format and lint
   black src/ tests/
   isort src/ tests/
   ruff check src/ tests/
   mypy src/

   # Run tests
   pytest

   # Run integration tests
   pytest tests/integration/
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR Review**:
   - Ensure all CI checks pass
   - Address reviewer feedback
   - Update PR description with details

### Testing Guidelines

#### Unit Tests
- Test individual functions and methods
- Use descriptive test names
- Mock external dependencies
- Aim for 90%+ coverage

#### Integration Tests
- Test API endpoints
- Test database operations
- Test end-to-end workflows

#### Performance Tests
- Benchmark critical operations
- Monitor memory usage
- Test scalability

Example test structure:
```python
import pytest
from dfs_prophet.core import EmbeddingGenerator

class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create EmbeddingGenerator instance."""
        return EmbeddingGenerator()
    
    def test_generate_player_embedding(self, generator):
        """Test player embedding generation."""
        # Test implementation
        pass
    
    @pytest.mark.asyncio
    async def test_batch_embeddings(self, generator):
        """Test batch embedding generation."""
        # Test implementation
        pass
```

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues**: Check if the bug is already reported
2. **Reproduce the issue**: Ensure you can consistently reproduce it
3. **Check documentation**: Verify it's not a configuration issue

### Bug Report Template

```markdown
## Bug Description
Brief description of the issue.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.11.5]
- DFS Prophet: [e.g., 0.1.0]
- Qdrant: [e.g., 1.14.0]

## Additional Information
- Error messages/logs
- Screenshots
- Configuration files
```

## ✨ Feature Requests

### Before Requesting

1. **Search existing issues**: Check if the feature is already requested
2. **Consider alternatives**: Look for existing solutions
3. **Think about implementation**: Consider complexity and impact

### Feature Request Template

```markdown
## Feature Description
Clear description of the feature.

## Problem Statement
What problem does this feature solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
What other approaches were considered?

## Additional Context
- Use cases
- Examples
- Mockups
```

## 📝 Documentation

### Documentation Standards

- **Clear and concise**: Write for the target audience
- **Examples**: Include practical examples
- **Code blocks**: Use syntax highlighting
- **Links**: Link to related documentation
- **Images**: Include diagrams and screenshots when helpful

### Documentation Structure

```
docs/
├── getting-started/
│   ├── installation.md
│   ├── quick-start.md
│   └── configuration.md
├── api/
│   ├── endpoints.md
│   ├── models.md
│   └── examples.md
├── development/
│   ├── architecture.md
│   ├── contributing.md
│   └── testing.md
└── deployment/
    ├── docker.md
    ├── production.md
    └── monitoring.md
```

## 🏷️ Labels and Milestones

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority: high`: High priority issues
- `priority: low`: Low priority issues
- `priority: medium`: Medium priority issues

### Milestones

- `v1.1.0`: Next minor release
- `v1.2.0`: Future minor release
- `v2.0.0`: Major release
- `Backlog`: Future consideration

## 🎯 Development Priorities

### Current Focus Areas

1. **Performance Optimization**
   - Vector search speed improvements
   - Memory usage optimization
   - Batch processing efficiency

2. **Feature Enhancements**
   - Multi-sport support
   - Advanced analytics
   - Real-time data integration

3. **Developer Experience**
   - Better error messages
   - Improved documentation
   - Enhanced testing

4. **Production Readiness**
   - Monitoring and observability
   - Security improvements
   - Scalability enhancements

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful**: Treat others with respect
- **Be inclusive**: Welcome diverse perspectives
- **Be constructive**: Provide helpful feedback
- **Be patient**: Allow time for responses

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions
- **Email**: support@dfsprophet.com

## 🏆 Recognition

### Contributors

We recognize contributors in several ways:

- **Contributors list**: All contributors are listed in the README
- **Release notes**: Contributors are credited in release notes
- **Special thanks**: Significant contributions are highlighted

### Contribution Levels

- **Bronze**: 1-5 contributions
- **Silver**: 6-20 contributions
- **Gold**: 21+ contributions
- **Platinum**: Core team member

## 📞 Getting Help

### Before Asking for Help

1. **Check documentation**: Look for existing answers
2. **Search issues**: Check if your question was already answered
3. **Try debugging**: Attempt to solve the issue yourself

### Where to Ask

- **GitHub Discussions**: General questions and help
- **GitHub Issues**: Bug reports and feature requests
- **Email**: support@dfsprophet.com

### What to Include

- Clear description of the problem
- Steps to reproduce
- Environment details
- Error messages/logs
- What you've already tried

## 📄 License

By contributing to DFS Prophet, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DFS Prophet! 🏈
