#!/usr/bin/env bash

# Usage: OPENROUTER_API_KEY="your-key" ./test_simple.sh
set -e

echo "Starting integration tests...."
# Check if API key is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable is not set"
    echo "Usage: OPENROUTER_API_KEY=\"your-api-key\" ./test_simple.sh"
    exit 1
fi

echo "Found OPENROUTER_API_KEY"

# Build the plugin
echo "Building plugin..."
cargo build --release
echo "Plugin built successfully"

echo "Running integration tests..."
OPENROUTER_API_KEY="$OPENROUTER_API_KEY" cargo test test_llm_complete_integration test_llm_complete_with_context test_llm_complete_with_custom_api_key_env test_external_completer -- --nocapture

echo "All tests completed"
echo ""
echo "Usage examples:"
echo "  llm-complete \"git comm\" --model \"openrouter/google/gemma-2-9b-it:free\""
echo "  llm-complete \"docker run\" \"web server setup\" --model \"openrouter/google/gemma-2-9b-it:free\""
echo "  llm-config set --model \"openrouter/google/gemma-2-9b-it:free\""