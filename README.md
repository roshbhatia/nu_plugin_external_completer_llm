# Nu Plugin External Completer LLM

A Nushell plugin that provides AI-powered shell command completions using LiteLLM as a bridge to various AI models (OpenAI, Claude, Gemini, etc.).

## Features

- **AI-Powered Completions**: Generate intelligent command completions using various LLM models
- **Multiple AI Models**: Support for OpenAI GPT, Claude, Gemini, and other models via LiteLLM
- **Provider/Model Format**: Use standard `provider/model` format (e.g., `openai/gpt-4`, `anthropic/claude-3-sonnet`)
- **Flexible API Key Management**: Use standard environment variables or custom ones with `--api-key-from-env`
- **External Completer Integration**: Works as a Nushell external completer for seamless shell experience
- **Configuration Management**: Save and manage AI model preferences and settings
- **Context-Aware**: Provides directory and task context to improve completion accuracy
- **Fallback Handling**: Gracefully falls back to standard completions when AI is unavailable

## Prerequisites

1. **Python 3** with **LiteLLM** installed:
   ```bash
   pip install litellm
   ```

2. **API Keys**: Set environment variables for your preferred AI service:
   ```bash
   # OpenAI
   export OPENAI_API_KEY="your-openai-key"
   
   # Anthropic Claude
   export ANTHROPIC_API_KEY="your-claude-key"
   
   # Google Gemini
   export GOOGLE_API_KEY="your-google-key"
   # OR
   export GEMINI_API_KEY="your-gemini-key"
   
   # Other providers
   export AZURE_API_KEY="your-azure-key"
   export COHERE_API_KEY="your-cohere-key"
   export MISTRAL_API_KEY="your-mistral-key"
   export TOGETHER_API_KEY="your-together-key"
   export GROQ_API_KEY="your-groq-key"
   export OPENROUTER_API_KEY="your-openrouter-key"
   
   # Or use custom environment variable names with --api-key-from-env
   export MY_CUSTOM_API_KEY="your-key"
   ```

## Installation

1. Build the plugin:
   ```bash
   cargo build --release
   ```

2. Register with Nushell:
   ```bash
   plugin add target/release/nu_plugin_external_completer_llm
   plugin use external_completer_llm
   ```

## Commands

### `llm-complete`
Generate AI-powered command completions directly.

```bash
# Basic completion
llm-complete "git comm"

# With context
llm-complete "docker run" "setting up nginx server"

# With custom model and parameters
llm-complete "kubectl get" --model "openai/gpt-4" --temperature 0.1 --max-tokens 100

# Using Claude with custom API key environment variable
llm-complete "docker ps" --model "anthropic/claude-3-sonnet" --api-key-from-env "MY_CLAUDE_KEY"

# Debug mode
llm-complete "npm run" --debug
```

### `external-completer`
External completer interface for Nushell integration.

```bash
# Used internally by Nushell's external completer system
external-completer "git comm" 8
```

To set up as your external completer, add to your Nushell config:

```nushell
$env.config = {
    completions: {
        external: {
            enable: true
            completer: {|spans|
                nu_plugin_external_completer_llm external-completer ($spans | str join " ") ($spans | str join " " | str length)
            }
        }
    }
}
```

### `llm-config`
Configure the plugin settings.

```bash
# Show current configuration
llm-config show

# Update settings
llm-config set --model "anthropic/claude-3-sonnet" --temperature 0.2 --max-tokens 100

# Reset to defaults
llm-config reset
```

## Configuration

The plugin stores configuration in:
- **macOS**: `~/Library/Application Support/nushell/plugins/external_completer_llm.json`
- **Linux**: `~/.config/nushell/plugins/external_completer_llm.json`
- **Windows**: `%APPDATA%\nushell\plugins\external_completer_llm.json`

### Default Configuration
```json
{
  "model": "openai/gpt-3.5-turbo",
  "max_tokens": 150,
  "temperature": 0.3
}
```

## Supported Models

Via LiteLLM, the plugin supports models in `provider/model` format:

- **OpenAI**: `openai/gpt-4`, `openai/gpt-3.5-turbo`, `openai/gpt-4o`, etc.
- **Anthropic**: `anthropic/claude-3-sonnet`, `anthropic/claude-3-haiku`, `anthropic/claude-3-opus`, etc.
- **Google**: `google/gemini-pro`, `google/gemini-1.5-pro`, etc.
- **Azure**: `azure/gpt-4`, `azure/gpt-35-turbo`, etc.
- **Cohere**: `cohere/command-r-plus`, `cohere/command-r`, etc.
- **Mistral**: `mistral/mistral-large`, `mistral/mistral-medium`, etc.
- **Together**: `together/llama-3-70b-chat`, etc.
- **Groq**: `groq/llama3-70b-8192`, `groq/mixtral-8x7b-32768`, etc.
- **OpenRouter**: `openrouter/google/gemma-2-9b-it:free`, `openrouter/meta-llama/llama-3.1-405b-instruct`, etc.
- **And many more**: Check [LiteLLM documentation](https://docs.litellm.ai/docs/providers)

## Examples

```bash
# Git completions
llm-complete "git"
# → ["git add", "git commit", "git push", "git pull", "git status"]

# Docker with context
llm-complete "docker run -p" "need to expose port 80 for web server"
# → ["docker run -p 80:80", "docker run -p 8080:80", "docker run -p 443:80"]

# Kubernetes
llm-complete "kubectl get po"
# → ["kubectl get pods", "kubectl get pods -A", "kubectl get pods -o wide"]

# OpenRouter with free model
llm-complete "npm run" --model "openrouter/google/gemma-2-9b-it:free"
# → ["npm run build", "npm run dev", "npm run test", "npm run start"]
```

## Troubleshooting

1. **Python not found**: Ensure `python3` is in your PATH
2. **LiteLLM import error**: Install with `pip install litellm`
3. **API key issues**: Verify your API keys are set correctly
4. **No completions**: Use `--debug` flag to see error messages

## Development

This plugin is built using the Nushell plugin template and follows Nushell plugin conventions.

### Development Environment

**With Nix** (recommended):
```bash
# Enter development shell
nix-shell

# Or with direnv (if you have it installed)
direnv allow
```

**Without Nix**:
```bash
# Set up development environment
make dev-setup

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Quick Commands

```bash
# Show all available commands
make help

# Build and test
make build
make test

# Development workflow
make watch          # Auto-rebuild on changes
make watch-test     # Auto-test on changes
make check          # Run all checks (format, lint, test)

# Integration testing
make dev-test       # Run integration tests (needs API key)

# Plugin management
make install        # Install to Nushell
make uninstall      # Remove from Nushell
```

### Running Tests

**Unit Tests** (no API key required):
```bash
cargo test
```

**Integration Tests** (requires OPENROUTER_API_KEY and `pip install litellm`):
```bash
# Install Python dependency
pip install litellm

# Run integration tests with API key
OPENROUTER_API_KEY="your-key" ./test_simple.sh

# Or run specific tests
OPENROUTER_API_KEY="your-key" cargo test test_llm_complete_integration -- --nocapture
```

Integration tests validate:
- Real API calls to OpenRouter using free Gemma model
- All three commands (`llm-complete`, `external-completer`, `llm-config`)
- Custom API key environment variable handling
- Context-aware completions and error handling

### Debug Mode
Add `--debug` to any command to see detailed logging:
```bash
llm-complete "git" --debug
```

## License

MIT License - see LICENSE file for details.

