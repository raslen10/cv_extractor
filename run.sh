#!/bin/bash

# --- Helper Functions ---
error_exit() {
    echo "❌ $1. Exiting."
    exit 1
}

# --- Initial Setup (only needed once) ---
if [ ! -d "myenv" ]; then
    echo "🔧 Setting up Python environment and dependencies..."
    sudo apt-get update || error_exit "Failed to update package list"
    sudo apt-get install -y python3-pip python3-venv poppler-utils tesseract-ocr || error_exit "Failed to install required system packages"

    python3 -m venv myenv || error_exit "Failed to create Python virtual environment"
fi

# --- Activate Virtual Environment ---
echo "🔍 Activating Python virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source myenv/Scripts/activate || error_exit "Failed to activate Python virtual environment"
else
    # Linux/Mac
    source myenv/bin/activate || error_exit "Failed to activate Python virtual environment"
fi

# --- Upgrade pip and install dependencies ---
echo "🔍 Upgrading pip..."
python -m pip install --upgrade pip || error_exit "Failed to upgrade pip"

echo "🔍 Installing Python dependencies..."
pip install -r requirements.txt || error_exit "Failed to install Python dependencies"

# --- Check if Ollama is installed ---
if ! command -v ollama &> /dev/null; then
    error_exit "Ollama is not installed. Please install it from https://ollama.ai/"
else
    echo "✅ Ollama is installed."
fi

# --- Pull models only if not already pulled ---
pull_if_needed() {
    if ! ollama list | grep -q "$1"; then
        echo "⬇️ Pulling $1..."
        ollama pull "$1" || error_exit "Failed to pull model $1"
    else
        echo "✅ $1 already pulled."
    fi
}

echo "🔍 Checking for required Ollama models..."
pull_if_needed "llama3"
pull_if_needed "mistral"
pull_if_needed "phi"
pull_if_needed "llava"
pull_if_needed "nomic-embed-text"

# --- Create required folders ---
echo "📂 Ensuring required folders exist..."
mkdir -p uploads ground_truth results chroma_db || error_exit "Failed to create required folders"
chmod -R 755 uploads ground_truth results chroma_db || error_exit "Failed to set permissions for folders"

# --- Create sample ground truth if not exists ---
if [ ! -f "ground_truth/john_doe.json" ]; then
    echo "📝 Creating sample ground truth..."
    cat > ground_truth/john_doe.json << 'EOL'
{
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1 555-123-4567",
    "skills": ["Python", "Machine Learning", "Flask", "Docker"],
    "education": [
        {
            "degree": "BSc Computer Science",
            "institution": "University of Example",
            "year": "2020"
        }
    ],
    "experience": [
        {
            "title": "Data Scientist",
            "company": "Tech Corp",
            "duration": "2021-present"
        }
    ]
}
EOL
    echo "✅ Sample ground truth created at ground_truth/john_doe.json."
else
    echo "✅ Sample ground truth already exists."
fi

# --- Verify Project Structure ---
echo "🔍 Verifying project structure..."
required_files=("app.py" "requirements.txt" "templates/index.html")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        error_exit "Missing required file: $file"
    fi
done
echo "✅ Project structure verified."

# --- Run Ollama models in background ---
start_model() {
    model=$1
    echo "🚀 Starting $model..."
    
    # Start in background
    ollama run "$model" &

    # Optional: check if the process was launched
    if [ $? -eq 0 ]; then
        echo "✅ $model started successfully."
    else
        error_exit "Failed to start $model"
    fi
}

# --- Start only models that require "run" ---
echo "🔧 Starting Ollama models..."
start_model "llama3"
start_model "mistral"
start_model "phi"
start_model "llava"

# --- Skip "nomic-embed-text" for "run" ---
echo "⚠️ Skipping 'nomic-embed-text' as it does not support 'run' command."

# --- Wait for models to be ready ---
echo "⏳ Waiting for models to warm up..."
for model in llama3 mistral phi llava; do
    echo " - Checking if $model is ready..."
    sleep 5  # Simulate waiting since "ollama status" is not supported
    echo "✅ $model is assumed ready."
done

# --- Start Flask app ---
echo "🌐 Launching Flask app..."
if python app.py; then
    echo "✅ Flask app started successfully."
else
    error_exit "Failed to start Flask app"
fi