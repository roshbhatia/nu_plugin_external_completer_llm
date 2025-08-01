# Nu Plugin External Completer LLM

A Nushell plugin that provides AI-powered shell command completions using LiteLLM as a bridge to various AI models (OpenAI, Claude, Gemini, etc.).

## Features

- **AI-Powered Completions**: Generate intelligent command completions using various LLM models
- **Multiple AI Models**: Support for OpenAI GPT, Claude, Gemini, and other models via LiteLLM
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
   export OPENAI_API_KEY="your-openai-key"
   # OR
   export ANTHROPIC_API_KEY="your-claude-key"
   # OR other supported API keys
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
llm-complete "kubectl get" --model "gpt-4" --temperature 0.1 --max-tokens 100

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
llm-config set --model "claude-3-sonnet" --temperature 0.2 --max-tokens 100

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
  "model": "gpt-3.5-turbo",
  "max_tokens": 150,
  "temperature": 0.3
}
```

## Supported Models

Via LiteLLM, the plugin supports:

- **OpenAI**: `gpt-4`, `gpt-3.5-turbo`, etc.
- **Anthropic**: `claude-3-sonnet`, `claude-3-haiku`, etc.
- **Google**: `gemini-pro`, `gemini-1.5-pro`, etc.
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
```

## Troubleshooting

1. **Python not found**: Ensure `python3` is in your PATH
2. **LiteLLM import error**: Install with `pip install litellm`
3. **API key issues**: Verify your API keys are set correctly
4. **No completions**: Use `--debug` flag to see error messages

## Development

This plugin is built using the Nushell plugin template and follows Nushell plugin conventions.

### Running Tests
```bash
cargo test
```

### Debug Mode
Add `--debug` to any command to see detailed logging:
```bash
llm-complete "git" --debug
```

## License

MIT License - see LICENSE file for details.

