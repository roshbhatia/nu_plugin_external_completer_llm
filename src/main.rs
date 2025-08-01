use nu_plugin::{MsgPackSerializer, Plugin, PluginCommand, serve_plugin};
use nu_plugin::{EngineInterface, EvaluatedCall};
use nu_protocol::{Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::path::PathBuf;

pub struct ExternalCompleterLlmPlugin;

impl Plugin for ExternalCompleterLlmPlugin {
    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").into()
    }

    fn commands(&self) -> Vec<Box<dyn PluginCommand<Plugin = Self>>> {
        vec![
            Box::new(LlmCompleteCommand),
            Box::new(ExternalCompleterCommand),
            Box::new(LlmConfigCommand),
        ]
    }
}

#[derive(Serialize, Deserialize)]
struct LlmConfig {
    model: String,
    api_key: Option<String>,
    max_tokens: u32,
    temperature: f32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: "gpt-3.5-turbo".to_string(),
            api_key: None,
            max_tokens: 150,
            temperature: 0.3,
        }
    }
}

pub struct LlmCompleteCommand;

impl PluginCommand for LlmCompleteCommand {
    type Plugin = ExternalCompleterLlmPlugin;

    fn name(&self) -> &str {
        "llm-complete"
    }

    fn signature(&self) -> Signature {
        Signature::build(self.name())
            .required("partial_command", SyntaxShape::String, "The partial command to complete")
            .optional("context", SyntaxShape::String, "Additional context about the current directory or task")
            .named("model", SyntaxShape::String, "AI model to use (default: gpt-3.5-turbo)", Some('m'))
            .named("max-tokens", SyntaxShape::Int, "Maximum tokens for completion (default: 150)", Some('t'))
            .named("temperature", SyntaxShape::Number, "Temperature for AI response (default: 0.3)", Some('T'))
            .switch("debug", "Show debug information", Some('d'))
            .input_output_type(Type::Nothing, Type::List(Type::String.into()))
            .category(Category::Experimental)
    }

    fn description(&self) -> &str {
        "Generate intelligent shell command completions using AI via LiteLLM"
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                example: "llm-complete \"git comm\"",
                description: "Complete a git command",
                result: Some(Value::test_list(vec![
                    Value::test_string("git commit"),
                    Value::test_string("git commit -m \"message\""),
                    Value::test_string("git commit --amend"),
                ])),
            },
            Example {
                example: "llm-complete \"docker run\" \"working with nginx container\"",
                description: "Complete docker command with context",
                result: Some(Value::test_list(vec![
                    Value::test_string("docker run -d -p 80:80 nginx"),
                    Value::test_string("docker run -it nginx bash"),
                ])),
            },
        ]
    }

    fn run(
        &self,
        _plugin: &ExternalCompleterLlmPlugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        let partial_command: String = call.req(0)?;
        let context: Option<String> = call.opt(1)?;
        let debug = call.has_flag("debug")?;

        let mut config = load_config();
        
        // Override with command-line parameters
        if let Some(model_value) = call.get_flag::<Value>("model")? {
            config.model = model_value.as_str().unwrap_or(&config.model).to_string();
        }
        if let Some(max_tokens_value) = call.get_flag::<Value>("max-tokens")? {
            config.max_tokens = max_tokens_value.as_int().unwrap_or(config.max_tokens as i64) as u32;
        }
        if let Some(temp_value) = call.get_flag::<Value>("temperature")? {
            config.temperature = temp_value.as_float().unwrap_or(config.temperature as f64) as f32;
        }
        
        // Always check for API keys from environment
        config.api_key = std::env::var("OPENAI_API_KEY").ok().or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());

        match generate_completions(&partial_command, context.as_deref(), &config, debug) {
            Ok(completions) => {
                let values: Vec<Value> = completions
                    .into_iter()
                    .map(|completion| Value::string(completion, call.head))
                    .collect();
                
                Ok(PipelineData::Value(Value::list(values, call.head), None))
            }
            Err(e) => Err(LabeledError::new(format!("Failed to generate completions: {}", e)).with_label("completion error", call.head)),
        }
    }
}

fn generate_completions(
    partial_command: &str,
    context: Option<&str>,
    config: &LlmConfig,
    debug: bool,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let api_key = config.api_key.as_ref()
        .ok_or("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")?;

    let context_text = context.map(|c| format!(" Context: {}", c)).unwrap_or_default();
    
    let prompt = format!(
        "You are a shell command completion assistant. Given a partial command, suggest 3-5 most likely completions.
        
Partial command: \"{}\"{}

Rules:
- Return only valid, executable shell commands
- Focus on the most common and useful completions
- Each completion should be on a separate line 
- No explanations, just the commands
- Consider common flags and arguments for the command",
        partial_command, context_text
    );

    if debug {
        eprintln!("Debug: Using model: {}", config.model);
        eprintln!("Debug: Prompt: {}", prompt);
    }

    let python_script = format!(
        r#"
import litellm
import sys
import os
import json

# Set API keys based on model type
api_key = '{}'
model = '{}'

# Set appropriate environment variable based on model
if model.startswith('gpt-') or model.startswith('text-') or model.startswith('davinci'):
    os.environ['OPENAI_API_KEY'] = api_key
elif model.startswith('claude'):
    os.environ['ANTHROPIC_API_KEY'] = api_key
elif model.startswith('gemini'):
    os.environ['GEMINI_API_KEY'] = api_key
else:
    # Default to OpenAI
    os.environ['OPENAI_API_KEY'] = api_key

try:
    response = litellm.completion(
        model=model,
        messages=[{{'role': 'user', 'content': '{}'}}],
        max_tokens={},
        temperature={}
    )
    
    content = response.choices[0].message.content.strip()
    lines = []
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//') and not line.startswith('*'):
            # Clean up common completion prefixes
            if line.startswith('$'):
                line = line[1:].strip()
            if line and len(line) > 0:
                lines.append(line)
    
    # Limit to reasonable number of completions
    for line in lines[:5]:
        print(line)
            
except ImportError:
    print("Error: litellm package not found. Install with: pip install litellm", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    if {}:
        print(f"LLM Error: {{e}}", file=sys.stderr)
    # Don't exit with error for external completer - just return empty
    pass
"#,
        api_key,
        config.model,
        prompt.replace('\'', "\\'").replace('\n', "\\n").replace('"', "\\\""),
        config.max_tokens,
        config.temperature,
        if debug { "True" } else { "False" }
    );

    let output = Command::new("python3")
        .arg("-c")
        .arg(&python_script)
        .output();

    let output = match output {
        Ok(output) => output,
        Err(e) => {
            if debug {
                eprintln!("Debug: Failed to execute python3: {}", e);
            }
            return Err("Python3 not found or failed to execute. Ensure python3 is installed and litellm package is available.".into());
        }
    };

    let completions = if output.status.success() {
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect()
    } else {
        if debug {
            let error = String::from_utf8_lossy(&output.stderr);
            eprintln!("Debug: Python script error: {}", error);
        }
        // Return empty vector instead of error for graceful fallback
        vec![]
    };

    if debug {
        eprintln!("Debug: Generated {} completions", completions.len());
    }

    Ok(completions)
}

fn get_config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_dir = dirs::config_dir()
        .ok_or("Could not find config directory")?
        .join("nushell")
        .join("plugins");
    
    std::fs::create_dir_all(&config_dir)?;
    Ok(config_dir.join("external_completer_llm.json"))
}

fn load_config() -> LlmConfig {
    match get_config_path() {
        Ok(path) => {
            if path.exists() {
                match std::fs::read_to_string(&path) {
                    Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
                    Err(_) => LlmConfig::default(),
                }
            } else {
                LlmConfig::default()
            }
        }
        Err(_) => LlmConfig::default(),
    }
}

fn save_config(config: &LlmConfig) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_config_path()?;
    let content = serde_json::to_string_pretty(config)?;
    std::fs::write(path, content)?;
    Ok(())
}

pub struct ExternalCompleterCommand;

impl PluginCommand for ExternalCompleterCommand {
    type Plugin = ExternalCompleterLlmPlugin;

    fn name(&self) -> &str {
        "external-completer"
    }

    fn signature(&self) -> Signature {
        Signature::build(self.name())
            .required("line", SyntaxShape::String, "The current command line")
            .required("position", SyntaxShape::Int, "Cursor position in the line")
            .named("model", SyntaxShape::String, "AI model to use", Some('m'))
            .switch("debug", "Show debug information", Some('d'))
            .input_output_type(Type::Nothing, Type::List(Box::new(Type::Record(vec![].into()))))
            .category(Category::Experimental)
    }

    fn description(&self) -> &str {
        "External completer interface for Nushell integration"
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                example: "external-completer \"git comm\" 8",
                description: "Complete command at cursor position",
                result: None, // External completer results vary
            },
        ]
    }

    fn run(
        &self,
        _plugin: &ExternalCompleterLlmPlugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        let line: String = call.req(0)?;
        let position: i64 = call.req(1)?;
        let model = call.get_flag::<Value>("model")?.map(|v| v.as_str().unwrap_or("gpt-3.5-turbo").to_string());
        let debug = call.has_flag("debug")?;

        // Extract the partial command up to the cursor position
        let partial_command = if position >= 0 && (position as usize) <= line.len() {
            &line[..(position as usize)]
        } else {
            &line
        };

        // Get working directory for context
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        let context = if !cwd.is_empty() {
            Some(format!("Current directory: {}", cwd))
        } else {
            None
        };

        let mut config = load_config();
        
        // Override with command-line parameters
        if let Some(model_value) = model {
            config.model = model_value;
        }
        
        // External completer specific settings
        config.max_tokens = 50; // Shorter for external completer
        config.temperature = 0.1; // More deterministic for completions
        
        // Always check for API keys from environment
        config.api_key = std::env::var("OPENAI_API_KEY").ok().or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());

        match generate_completions(partial_command, context.as_deref(), &config, debug) {
            Ok(completions) => {
                let values: Vec<Value> = completions
                    .into_iter()
                    .map(|completion| {
                        // Create record with value and description for external completer
                        Value::record(
                            vec![
                                ("value".to_string(), Value::string(completion.clone(), call.head)),
                                ("description".to_string(), Value::string(format!("AI suggestion: {}", completion), call.head)),
                            ].into_iter().collect(),
                            call.head
                        )
                    })
                    .collect();

                Ok(PipelineData::Value(Value::list(values, call.head), None))
            }
            Err(e) => {
                if debug {
                    eprintln!("LLM completion failed: {}, falling back to empty list", e);
                }
                // Return empty list on error to allow fallback to standard completion
                Ok(PipelineData::Value(Value::list(vec![], call.head), None))
            }
        }
    }
}

pub struct LlmConfigCommand;

impl PluginCommand for LlmConfigCommand {
    type Plugin = ExternalCompleterLlmPlugin;

    fn name(&self) -> &str {
        "llm-config"
    }

    fn signature(&self) -> Signature {
        Signature::build(self.name())
            .optional("action", SyntaxShape::String, "Action: show, set, or reset")
            .named("model", SyntaxShape::String, "Set the default AI model", Some('m'))
            .named("max-tokens", SyntaxShape::Int, "Set maximum tokens", Some('t'))
            .named("temperature", SyntaxShape::Number, "Set temperature", Some('T'))
            .input_output_type(Type::Nothing, Type::Record(vec![].into()))
            .category(Category::Experimental)
    }

    fn description(&self) -> &str {
        "Manage LLM completion configuration"
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                example: "llm-config show",
                description: "Show current configuration",
                result: None,
            },
            Example {
                example: "llm-config set --model gpt-4 --temperature 0.2",
                description: "Update configuration settings",
                result: None,
            },
            Example {
                example: "llm-config reset",
                description: "Reset to default configuration",
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin: &ExternalCompleterLlmPlugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        let action: Option<String> = call.opt(0)?;
        let action = action.as_deref().unwrap_or("show");

        match action {
            "show" => {
                let config = load_config();
                let result = Value::record(
                    vec![
                        ("model".to_string(), Value::string(config.model, call.head)),
                        ("max_tokens".to_string(), Value::int(config.max_tokens as i64, call.head)),
                        ("temperature".to_string(), Value::float(config.temperature as f64, call.head)),
                        ("api_key_set".to_string(), Value::bool(config.api_key.is_some(), call.head)),
                        ("config_path".to_string(), 
                         Value::string(
                             get_config_path().map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|_| "error".to_string()),
                             call.head
                         )),
                    ].into_iter().collect(),
                    call.head
                );
                Ok(PipelineData::Value(result, None))
            }
            "set" => {
                let mut config = load_config();
                
                if let Some(model) = call.get_flag::<Value>("model")? {
                    config.model = model.as_str().unwrap_or(&config.model).to_string();
                }
                
                if let Some(max_tokens) = call.get_flag::<Value>("max-tokens")? {
                    config.max_tokens = max_tokens.as_int().unwrap_or(config.max_tokens as i64) as u32;
                }
                
                if let Some(temperature) = call.get_flag::<Value>("temperature")? {
                    config.temperature = temperature.as_float().unwrap_or(config.temperature as f64) as f32;
                }

                match save_config(&config) {
                    Ok(_) => {
                        Ok(PipelineData::Value(
                            Value::string("Configuration updated successfully", call.head),
                            None
                        ))
                    }
                    Err(e) => Err(LabeledError::new(format!("Failed to save config: {}", e)).with_label("config error", call.head)),
                }
            }
            "reset" => {
                let config = LlmConfig::default();
                match save_config(&config) {
                    Ok(_) => {
                        Ok(PipelineData::Value(
                            Value::string("Configuration reset to defaults", call.head),
                            None
                        ))
                    }
                    Err(e) => Err(LabeledError::new(format!("Failed to reset config: {}", e)).with_label("config error", call.head)),
                }
            }
            _ => Err(LabeledError::new("Invalid action. Use: show, set, or reset").with_label("invalid action", call.head)),
        }
    }
}

fn main() {
    serve_plugin(&ExternalCompleterLlmPlugin, MsgPackSerializer);
}
