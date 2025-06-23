# RAG System Makefile
# Provides common development tasks

.PHONY: help install install-dev test test-unit test-integration test-coverage lint format type-check security clean docs

# Default target
help:
	@echo "RAG System Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  test-module      Run tests for specific module (e.g., make test-module MODULE=course)"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run all linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security         Run security checks"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean up temporary files"
	@echo "  docs             Generate documentation"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Testing
test:
	python -m pytest rag/tests/ -v

test-unit:
	python -m pytest rag/tests/ -v -m "not integration"

test-integration:
	python -m pytest rag/tests/ -v -m "integration"

test-coverage:
	python -m pytest rag/tests/ --cov=rag --cov-report=html --cov-report=term --cov-report=xml

test-module:
	@if [ -z "$(MODULE)" ]; then \
		echo "Usage: make test-module MODULE=<module_name>"; \
		echo "Example: make test-module MODULE=course"; \
		exit 1; \
	fi
	python -m pytest rag/tests/test_$(MODULE).py -v

# Custom test runner
test-runner:
	python -m rag.tests.test_runner

test-runner-coverage:
	python -m rag.tests.test_runner --coverage

# Code Quality
lint: lint-flake8 lint-pylint

lint-flake8:
	flake8 rag/ --max-line-length=127 --extend-ignore=E203,W503

lint-pylint:
	pylint rag/ --disable=C0114,C0115,C0116,R0903,R0913,W0613

format:
	black rag/
	isort rag/

format-check:
	black --check rag/
	isort --check-only rag/

type-check:
	mypy rag/ --ignore-missing-imports --no-strict-optional

security:
	bandit -r rag/ -ll
	safety scan

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "TODO: Add documentation generation (e.g., with Sphinx)"

# Utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

# CI simulation
ci-test: format-check lint type-check security test-coverage
	@echo "All CI checks passed! ✅"

# Quick development check
quick-check: format lint test-unit
	@echo "Quick development check passed! ✅"

# Full check (like CI)
full-check: clean install-dev ci-test
	@echo "Full check completed! ✅" 