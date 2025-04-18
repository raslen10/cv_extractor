# CV Extractor with Open-Source LLMs


A Flask-based web application that extracts structured information from CVs using open-source LLMs (Llama3, Mistral, Phi-2) via Ollama, with comprehensive evaluation against ground truth data.

## 📌 Features

- **Multi-format Support**: Handles both text-based and scanned PDFs
- **Multi-model Comparison**: Processes CVs with 3 different LLMs
- **RAG Pipeline**: Uses ChromaDB with Nomic embeddings for information retrieval
- **Comprehensive Evaluation**: Field-level precision, recall, and F1 scores
- **Web Interface**: Simple UI for uploading CVs and viewing results

## 🧠 Models Used

| Model       | Type          | Use Case                     | Size   |
|-------------|---------------|------------------------------|--------|
| Llama3      | LLM           | Information extraction       | 8B/70B |
| Mistral     | LLM           | Information extraction       | 7B     |
| Phi-2       | LLM           | Information extraction       | 2.7B   |
| LLaVA       | Vision LLM    | OCR for scanned documents    | 7B     |
| Nomic-Embed | Embedding     | Vector embeddings for RAG    | 137M   |

## 🔧 Technical Stack

- **Backend**: Python, Flask
- **LLM Server**: Ollama
- **Vector DB**: ChromaDB
- **OCR**: PyMuPDF, LLaVA
- **Evaluation**: scikit-learn metrics
- **Frontend**: HTML/CSS, Bootstrap

## 📊 Evaluation Logic

### Field-level Comparison

1. **String Fields** (Name, Email, Phone):
   - Exact match comparison (case-insensitive)
   - Binary scoring (1 for match, 0 for no match)

2. **List Fields** (Skills, Education, Experience):
   - **Skills**: Set comparison of skill terms
   - **Education/Experience**: JSON structure comparison
   - Metrics calculated using:
     - True Positives (TP): Correctly extracted items
     - False Positives (FP): Incorrectly extracted items
     - False Negatives (FN): Missed ground truth items

### Metrics Calculation

For each model and field, we calculate:

```python
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Accuracy = (TP + TN) / (TP + FP + TN + FN)
Overall ModeOverall Model Scoring
Weighted average of field-level F1 scores, with weights:

Contact info (name/email/phone): 30%

Skills: 20%

Education: 25%

Experience: 25%

🚀 Installation
1. Prerequisites
Python 3.9+

2. Ollama Setup
To set up Ollama, follow these steps:
    . Download and install Ollama from 'https://ollama.com/'
    . Start the Ollama server
```
ollama serve
``` 
    . Pull the required models
```
ollama pull llama3
ollama pull mistral
ollama pull phi
ollama pull llava
ollama pull nomic-embed-text
``` 
    . Finally, I installed the Ollama Python package using:
``` 
pip install ollama
``` 

3. Project Setup
``` 
# Clone repository
git clone https://github.com/raslen10/cv-extractor.git
cd cv-extractor

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads ground_truth results chroma_db
```

## ⚙️ Configuration

### Ground Truth Preparation

Place JSON files in `ground_truth/` matching CV filenames.

Example: `john_doe.pdf` → `ground_truth/john_doe.json`.

Format:

```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "+1234567890",
  "skills": ["Python", "Machine Learning"],
  "education": [
     {
        "degree": "BSc Computer Science",
        "institution": "University X",
        "year": "2020"
     }
  ],
  "experience": [
     {
        "title": "Data Scientist",
        "company": "Company Y",
        "duration": "2021-present"
     }
  ]
}
```



## 🏃 Running the Application

### **Option 1: Using `run.sh` (Recommended)**

```bash
# Make the script executable (only needed once)
chmod +x [run.sh](http://_vscodecontentref_/0)

# Run the script
[./run.sh]
Access the web interface at: [http://localhost:5000](http://localhost:5000).
The run.sh script will:

Set up the Python environment (if not already set up).
Install all required dependencies.
Pull the necessary models using Ollama.
Start the Flask application.
Once the application is running, open your browser and navigate to:

http://localhost:5000


---

### **Partie 2**

```markdown
### **Option 2: Running Manually via Terminal**

#### **Step 1: Start the Ollama Server**
In a separate terminal, start the Ollama server:
```bash
ollama serve
python -m venv myenv
myenv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


---

### **Partie 3**

```markdown
#### **Step 3: Pull Required Models**
Ensure the required models are pulled using Ollama:
```bash
ollama pull llama3
ollama pull mistral
ollama pull phi
ollama pull llava
ollama pull nomic-embed-text

python app.py

Once the application is running, open your browser and navigate to:
http://localhost:5000
## 📂 Project Structure

```plaintext
cv-extractor/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── static/                # Static files (CSS, JS)
├── templates/             # HTML templates
├── uploads/               # Uploaded CVs
├── ground_truth/          # Ground truth JSON files
├── results/               # Evaluation results (CSV/JSON)
└── chroma_db/             # ChromaDB vector storage
```

## 📝 Usage Instructions

1. Upload a CV (PDF or image).
2. The system will:
    - Extract text (using OCR if needed).
    - Process with all three LLMs.
    - Compare with ground truth (if available).
    - Generate evaluation metrics.
3. View results in the web interface.
4. Download evaluation reports as CSV/JSON.

## 📈 Sample Evaluation Output

### CSV Report (`results/Data_Analyst_comparison.csv`):

```csv
model,name_match,email_match,phone_match,skills_match,education_match,experience_match
llama3,1,1,1,1,1,1
mistral,1,1,1,0,1,0
phi,1,1,1,0,0,0
```

### JSON Metrics (`results/Data_Analyst_metrics.json`):

```json
{
  "llama3": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0
  },
  "mistral": {
    "accuracy": 0.6666666666666666,
    "precision": 1.0,
    "recall": 0.6666666666666666,
    "f1_score": 0.8
  },
  "phi": {
    "accuracy": 0.5,
    "precision": 1.0,
    "recall": 0.5,
    "f1_score": 0.6666666666666666
  }
}
```

## 🛠️ Troubleshooting

### Common Issues:

- **Ollama connection refused**:
  - Ensure `ollama serve` is running.
  

- **Missing ground truth**:
  - Verify filename matches between CV and ground truth.
  - Check JSON format is valid.

- **OCR failures**:
  - Ensure LLaVA model is pulled (`ollama pull llava`).
  - Check image quality of scanned PDFs.

## 📜 License

MIT License - See `LICENSE` for details.

## 📬 Contact

For questions or support, please contact: `guesmiraslen2@gmail.com`.
