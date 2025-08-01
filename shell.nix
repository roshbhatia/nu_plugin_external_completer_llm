{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "nu-plugin-external-completer-llm-dev";

  buildInputs = with pkgs; [
    # Rust toolchain
    rustc
    cargo
    rustfmt
    clippy
    rust-analyzer

    # Nushell
    nushell

    # Python for LiteLLM
    python3
    python3Packages.pip
    python3Packages.setuptools

    # Development tools
    git
    gnumake
    
    # Optional: cargo extensions
    cargo-watch
    cargo-audit
    cargo-edit
    
    # System dependencies
    pkg-config
    openssl
    
    # For building on different platforms
    gcc
    
    # Optional: documentation tools
    mdbook
  ] ++ lib.optionals stdenv.isDarwin [
    # macOS specific dependencies
    darwin.apple_sdk.frameworks.Security
    darwin.apple_sdk.frameworks.SystemConfiguration
  ];

  shellHook = ''
    echo "🚀 Nu Plugin External Completer LLM Development Environment"
    echo "==========================================================="
    echo ""
    
    # Check if we're in the right directory
    if [[ ! -f "Cargo.toml" ]]; then
      echo "⚠️  Warning: Cargo.toml not found. Make sure you're in the project directory."
    else
      echo "📦 Project: $(grep '^name = ' Cargo.toml | cut -d'"' -f2)"
      echo "📝 Version: $(grep '^version = ' Cargo.toml | cut -d'"' -f2)"
    fi
    
    echo ""
    echo "🔧 Available tools:"
    echo "  rustc $(rustc --version | cut -d' ' -f2)"
    echo "  cargo $(cargo --version | cut -d' ' -f2)"
    echo "  nushell $(nu --version | head -n1)"
    echo "  python $(python3 --version | cut -d' ' -f2)"
    echo ""
    
    # Set up Python virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
      echo "🐍 Setting up Python virtual environment..."
      python3 -m venv .venv
      echo "Virtual environment created at .venv/"
    fi
    
    echo "🐍 Activating Python virtual environment..."
    source .venv/bin/activate
    
    # Install Python dependencies
    if ! python -c "import litellm" 2>/dev/null; then
      echo "📦 Installing litellm..."
      pip install litellm
    else
      echo "✅ litellm already installed"
    fi
    
    echo ""
    echo "🎯 Quick commands:"
    echo "  make help           - Show all available make targets"
    echo "  make build          - Build the plugin"
    echo "  make test           - Run unit tests"
    echo "  make dev-test       - Run integration tests (needs OPENROUTER_API_KEY)"
    echo "  make install        - Install plugin to Nushell"
    echo "  make watch          - Watch for changes and rebuild"
    echo ""
    echo "🔑 For integration tests, set your API key:"
    echo "  export OPENROUTER_API_KEY=\"your-key\""
    echo "  make dev-test"
    echo ""
    echo "📚 Documentation:"
    echo "  make docs           - Generate and open Rust docs"
    echo "  cat README.md       - View project README"
    echo ""
    
    # Check if API key is set
    if [[ -n "$OPENROUTER_API_KEY" ]]; then
      echo "✅ OPENROUTER_API_KEY is set"
    else
      echo "⚠️  OPENROUTER_API_KEY not set (needed for integration tests)"
    fi
    
    echo ""
    echo "Happy coding! 🦀"
  '';

  # Environment variables
  RUST_BACKTRACE = "1";
  RUST_LOG = "debug";
  
  # Ensure cargo uses the nix-provided OpenSSL
  PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
  OPENSSL_DIR = "${pkgs.openssl.dev}";
  OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
  OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
}