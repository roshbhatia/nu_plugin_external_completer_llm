# Makefile for nu_plugin_external_completer_llm

.PHONY: build test clean install dev-test integration-test lint format check all help

# Default target
all: build test

# Build the plugin
build:
	@echo "Building plugin..."
	cargo build

# Build release version
build-release:
	@echo "Building release version..."
	cargo build --release

# Run unit tests
test:
	@echo "Running unit tests..."
	cargo test

# Run integration tests (requires OPENROUTER_API_KEY)
integration-test:
	@echo "Running integration tests..."
	@if [ -z "$(OPENROUTER_API_KEY)" ]; then \
		echo "Error: OPENROUTER_API_KEY not set"; \
		echo "Usage: make integration-test OPENROUTER_API_KEY=your-key"; \
		exit 1; \
	fi
	OPENROUTER_API_KEY="$(OPENROUTER_API_KEY)" cargo test test_llm_complete_integration test_llm_complete_with_context test_llm_complete_with_custom_api_key_env test_external_completer -- --nocapture

# Full test suite with integration tests
dev-test:
	@echo "Running full test suite..."
	@./test_simple.sh

# Lint the code
lint:
	@echo "Running clippy..."
	cargo clippy -- -D warnings

# Format the code
format:
	@echo "Formatting code..."
	cargo fmt

# Check formatting
check-format:
	@echo "Checking format..."
	cargo fmt -- --check

# Security audit
audit:
	@echo "Running security audit..."
	cargo audit

# Check everything
check: check-format lint test
	@echo "All checks passed!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cargo clean

# Install the plugin to Nushell
install: build-release
	@echo "Installing plugin to Nushell..."
	@if command -v nu >/dev/null 2>&1; then \
		echo "Adding plugin to Nushell..."; \
		nu -c "plugin add target/release/nu_plugin_external_completer_llm; plugin use external_completer_llm"; \
		echo "Plugin installed! Run 'nu' and try: llm-complete \"git comm\" --model \"openrouter/google/gemma-2-9b-it:free\""; \
	else \
		echo "Nushell not found. Build completed at: target/release/nu_plugin_external_completer_llm"; \
	fi

# Uninstall the plugin from Nushell
uninstall:
	@echo "Removing plugin from Nushell..."
	@if command -v nu >/dev/null 2>&1; then \
		nu -c "plugin rm external_completer_llm" || echo "Plugin not found or already removed"; \
	else \
		echo "Nushell not found"; \
	fi

# Development setup
dev-setup:
	@echo "Setting up development environment..."
	@echo "Installing Rust dependencies..."
	cargo fetch
	@echo "Installing Python dependencies..."
	pip install litellm
	@echo "Development setup complete!"

# Generate documentation
docs:
	@echo "Generating documentation..."
	cargo doc --open

# Watch and rebuild on changes
watch:
	@echo "Watching for changes..."
	cargo watch -x build

# Watch and test on changes
watch-test:
	@echo "Watching for changes and running tests..."
	cargo watch -x test

# Show help
help:
	@echo "Available targets:"
	@echo "  build           - Build the plugin"
	@echo "  build-release   - Build release version"
	@echo "  test            - Run unit tests"
	@echo "  integration-test- Run integration tests (requires OPENROUTER_API_KEY=key)"
	@echo "  dev-test        - Run full test suite using test_simple.sh"
	@echo "  lint            - Run clippy linter"
	@echo "  format          - Format code with rustfmt"
	@echo "  check-format    - Check if code is formatted"
	@echo "  audit           - Run security audit"
	@echo "  check           - Run all checks (format, lint, test)"
	@echo "  clean           - Clean build artifacts"
	@echo "  install         - Install plugin to Nushell"
	@echo "  uninstall       - Remove plugin from Nushell"
	@echo "  dev-setup       - Set up development environment"
	@echo "  docs            - Generate and open documentation"
	@echo "  watch           - Watch for changes and rebuild"
	@echo "  watch-test      - Watch for changes and run tests"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make integration-test OPENROUTER_API_KEY=your-key"
	@echo "  OPENROUTER_API_KEY=your-key make dev-test"