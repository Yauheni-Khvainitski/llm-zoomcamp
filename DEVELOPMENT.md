# 🛠️ Development Guide

## Dependency Management

### ✅ Modern Approach (Recommended)

This project uses `pyproject.toml` as the single source of truth and supports both **Poetry** and **pip**:

#### **Option 1: Poetry (Recommended)**
```bash
# Install with Poetry
poetry install

# Install with specific groups
poetry install --with dev,jupyter,tokens
```

#### **Option 2: pip**
```bash
# Install for development
pip install -e ".[dev,jupyter,tokens]"

# Install just core dependencies
pip install -e .

# Install specific groups
pip install -e ".[dev]"          # Development tools
pip install -e ".[jupyter]"      # Notebook support
pip install -e ".[tokens]"       # Token counting
```

### 📋 What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `pyproject.toml` | **Primary config** - Dependencies, tool settings | Always (single source of truth) |
| `requirements.txt` | **Convenience wrapper** - Points to pyproject.toml | CI/CD, compatibility |

### 🚀 Quick Commands

#### **With Poetry**
```bash
# Setup development environment
poetry install

# Run tests
poetry run pytest rag/tests/

# Code quality
poetry run black rag/ && poetry run isort rag/ && poetry run flake8 rag/

# Type checking
poetry run mypy rag/
```

#### **With pip**
```bash
# Setup development environment
pip install -e ".[dev,jupyter,tokens]"

# Run tests
pytest rag/tests/

# Code quality
black rag/ && isort rag/ && flake8 rag/

# Type checking
mypy rag/

# Full CI simulation
pytest rag/tests/ --cov=rag --cov-report=html
black --check rag/
isort --check-only rag/
flake8 rag/
mypy rag/
```

## Adding Dependencies

### ✅ Correct Way
Edit `pyproject.toml`:

```toml
# For runtime dependencies
[project]
dependencies = [
    "new-package>=1.0.0",
]

# For development dependencies
[project.optional-dependencies]
dev = [
    "new-dev-tool>=2.0.0",
]
```

### ❌ Avoid
- Don't add to `requirements.txt` directly
- Don't maintain duplicate dependency lists

## CI/CD Notes

All workflows use pip for consistency:
```yaml
- name: Install dependencies
  run: pip install -e ".[dev,jupyter,tokens]"
```

**Why pip in CI?** While both Poetry and pip work locally, pip is used in CI/CD for:
- Faster installation (no lock file resolution)
- Better caching support in GitHub Actions
- Simpler workflow configuration 