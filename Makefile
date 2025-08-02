# Makefile for tokamak-rl-control-suite
# Type 'make help' for available commands

.PHONY: help install install-dev clean test test-fast lint format typecheck security docs build package

# Default Python interpreter
PYTHON := python3
PIP := pip

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install package in production mode
	$(PIP) install .

install-dev: ## Install package in development mode with all dependencies
	$(PIP) install -e ".[dev,docs,mpi]"
	pre-commit install

install-minimal: ## Install package with minimal dependencies
	$(PIP) install -e .

# Development targets
clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf outputs/
	rm -rf logs/

# Testing targets
test: ## Run all tests with coverage
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-fast: ## Run tests with fail-fast and failed-first
	pytest $(TEST_DIR) -x --ff

test-unit: ## Run only unit tests
	pytest $(TEST_DIR)/unit/ -v

test-integration: ## Run only integration tests
	pytest $(TEST_DIR)/integration/ -v

test-performance: ## Run performance benchmarks
	pytest $(TEST_DIR)/performance/ -v --benchmark-only

test-security: ## Run security-specific tests
	pytest $(TEST_DIR)/security/ -v

# Code quality targets
lint: ## Run linting with ruff
	ruff check $(SRC_DIR) $(TEST_DIR)

lint-fix: ## Run linting with auto-fix
	ruff check --fix $(SRC_DIR) $(TEST_DIR)

format: ## Format code with black
	black $(SRC_DIR) $(TEST_DIR)

format-check: ## Check code formatting without changes
	black --check $(SRC_DIR) $(TEST_DIR)

typecheck: ## Run type checking with mypy
	mypy $(SRC_DIR)

security: ## Run security checks with bandit
	bandit -r $(SRC_DIR) -f json -o bandit-report.json

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

# Documentation targets
docs: ## Build documentation
	sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html

docs-clean: ## Clean documentation build
	rm -rf $(DOCS_DIR)/_build/

docs-serve: ## Serve documentation locally
	$(PYTHON) -m http.server 8080 --directory $(DOCS_DIR)/_build/html

docs-auto: ## Auto-build documentation on changes
	sphinx-autobuild $(DOCS_DIR) $(DOCS_DIR)/_build/html

# Build targets
build: ## Build package distribution
	$(PYTHON) -m build

package: clean build ## Clean and build package
	@echo "Package built successfully!"

# Example and training targets
train-example: ## Run example SAC training
	$(PYTHON) -m tokamak_rl.examples.train_sac

train-dreamer: ## Run example Dreamer training
	$(PYTHON) -m tokamak_rl.examples.train_dreamer

eval-example: ## Run example evaluation
	$(PYTHON) -m tokamak_rl.examples.evaluate

benchmark: ## Run comprehensive benchmarks
	$(PYTHON) -m tokamak_rl.benchmarks.run_all

# Monitoring targets
tensorboard: ## Launch TensorBoard
	tensorboard --logdir outputs/tensorboard --port 6006

jupyter: ## Launch Jupyter Lab
	jupyter lab --port 8888 --ip 0.0.0.0 --allow-root

monitor: ## Launch monitoring dashboard
	$(PYTHON) -m tokamak_rl.monitoring.dashboard

# Data and model targets
download-data: ## Download example experimental data
	$(PYTHON) -m tokamak_rl.data.download_examples

cache-clear: ## Clear physics simulation cache
	rm -rf physics_cache/
	rm -rf equilibrium_cache/

models-download: ## Download pre-trained models
	$(PYTHON) -m tokamak_rl.models.download_pretrained

# Container targets
docker-build: ## Build Docker image
	docker build -t tokamak-rl:latest .

docker-run: ## Run Docker container
	docker run -it --rm --gpus all -p 6006:6006 -p 8888:8888 tokamak-rl:latest

docker-dev: ## Run Docker container in development mode
	docker-compose up -d

# Quality assurance targets
qa: lint typecheck test security ## Run all quality assurance checks

ci: install-dev qa docs build ## Run complete CI pipeline

release-check: ## Check if ready for release
	@echo "Checking release readiness..."
	@$(MAKE) clean
	@$(MAKE) install-dev
	@$(MAKE) qa
	@$(MAKE) docs
	@$(MAKE) build
	@echo "✅ Release checks passed!"

# Development environment
dev-setup: ## Set up complete development environment
	@echo "Setting up development environment..."
	@$(MAKE) install-dev
	@$(MAKE) download-data
	@$(MAKE) models-download
	@echo "✅ Development environment ready!"

# Performance profiling
profile: ## Run performance profiling
	$(PYTHON) -m cProfile -o profile.prof -m tokamak_rl.examples.train_sac
	$(PYTHON) -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"

memory-profile: ## Run memory profiling
	mprof run $(PYTHON) -m tokamak_rl.examples.train_sac
	mprof plot

# Dependency management
deps-update: ## Update all dependencies to latest versions
	pip-compile --upgrade pyproject.toml

deps-check: ## Check for security vulnerabilities in dependencies
	safety check

# Git hooks
hooks-install: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

hooks-uninstall: ## Uninstall git hooks
	pre-commit uninstall
	pre-commit uninstall --hook-type commit-msg

# Version management
version: ## Show current version
	$(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# Help for specific targets
help-test: ## Show testing help
	@echo "Testing Commands:"
	@echo "  make test          - Run all tests with coverage"
	@echo "  make test-fast     - Quick test run (fail-fast)"
	@echo "  make test-unit     - Unit tests only"
	@echo "  make test-integration - Integration tests only"
	@echo "  make test-performance - Performance benchmarks"
	@echo "  make test-security - Security tests only"

help-dev: ## Show development help
	@echo "Development Commands:"
	@echo "  make dev-setup     - Complete development setup"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make qa           - Run all quality checks"
	@echo "  make ci           - Run complete CI pipeline"