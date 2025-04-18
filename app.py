import os
import fitz
import json
import logging
import pandas as pd
import time
import csv
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from io import BytesIO
import base64
from pdf2image import convert_from_path
import ollama
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Configuration
app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
GROUND_TRUTH_FOLDER = 'ground_truth'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(GROUND_TRUTH_FOLDER, exist_ok=True)

# Models to compare
MODELS = ["llama3", "mistral", "phi"]
EMBEDDING_MODEL = "nomic-embed-text"
OCR_MODEL = "llava"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
ef = embedding_functions.OllamaEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    url="http://localhost:11434"
)

# Create collections for each model
collections = {
    model: chroma_client.get_or_create_collection(
        name=f"cv_{model}",
        embedding_function=ef
    ) for model in MODELS
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cv_extractor.log'),
        logging.StreamHandler()
    ]
)

class CVProcessor:
    """Handles CV text extraction and processing using LLMs"""
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text from PDF or image using OCR when needed"""
        print(f"\n[1/5] Starting text extraction for: {os.path.basename(file_path)}")
        try:
            if file_path.lower().endswith('.pdf'):
                print(" - Trying direct text extraction from PDF...")
                with fitz.open(file_path) as doc:
                    text = "\n".join(page.get_text() for page in doc)
                    if len(text.strip()) > 100:
                        print(" - Success! Found sufficient text in PDF")
                        return text
                
                print(" - Falling back to OCR (scanned PDF detected)")
                images = convert_from_path(file_path)
                text = ""
                for i, img in enumerate(images):
                    print(f" - Processing page {i+1} with LLaVA OCR...")
                    text += CVProcessor._extract_with_llava(img) + "\n"
                return text
            
            else:  # Image file
                print(" - Processing image file with LLaVA OCR...")
                img = Image.open(file_path)
                return CVProcessor._extract_with_llava(img)
                
        except Exception as e:
            logging.error(f"Text extraction failed: {str(e)}")
            print(f"!! Text extraction failed: {str(e)}")
            return ""

    @staticmethod
    def _extract_with_llava(img: Image.Image) -> str:
        """Use LLaVA model for OCR processing"""
        try:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            print(" - Sending to LLaVA for OCR processing...")
            response = ollama.generate(
                model=OCR_MODEL,
                prompt=f"Extract all text from this CV image accurately. Return only the raw text with no additional commentary:\n<img src='data:image/jpeg;base64,{img_str}'>",
                options={'temperature': 0}
            )
            print(" - OCR completed successfully")
            return response['response']
        except Exception as e:
            print(f"!! LLaVA OCR failed: {str(e)}")
            return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for RAG processing"""
        print(f"[2/5] Chunking text into {chunk_size} character segments with {overlap} overlap...")
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(' '.join(words[i:i+chunk_size]))
        print(f" - Created {len(chunks)} chunks")
        return chunks

    @staticmethod
    def build_extraction_prompt(text: str) -> str:
        """Create structured extraction prompt for LLMs"""
        print("[3/5] Building extraction prompt...")
        return f"""
        Extract the following information from this CV in JSON format:
        - name (string)
        - email (string)
        - phone (string)
        - education (list of objects with degree, institution, year)
        - skills (list)
        - experience (list of objects with title, company, duration)
        
        Return ONLY valid JSON with these exact field names. No additional text or explanation.
        
        CV Content:
        {text}
        """

    @staticmethod
    def process_cv(file_path: str) -> Dict[str, Any]:
        """Process a CV file through the full pipeline"""
        filename = os.path.basename(file_path)
        print(f"\n=== Processing CV: {filename} ===")
        
        # Step 1: Extract text
        text = CVProcessor.extract_text(file_path)
        if not text:
            raise ValueError("Failed to extract text from CV")
        print(f" - Extracted {len(text)} characters")
        
        # Step 2: Chunk and store in ChromaDB
        chunks = CVProcessor.chunk_text(text)
        doc_id = os.path.splitext(filename)[0]
        
        print("[4/5] Storing in ChromaDB...")
        for model in MODELS:
            print(f" - Processing with {model} embeddings...")
            for i, chunk in enumerate(chunks):
                collections[model].upsert(
                    ids=[f"{doc_id}_{i}"],
                    documents=[chunk],
                    metadatas=[{"source": filename, "chunk_num": i}]
                )
        
        # Step 3: Extract with all models
        print("[5/5] Extracting structured data with LLMs...")
        extractions = {}
        for model in MODELS:
            print(f" - Extracting with {model}...")
            try:
                start_time = time.time()
                response = ollama.generate(
                    model=model,
                    prompt=CVProcessor.build_extraction_prompt(text),
                    format='json',
                    options={'temperature': 0}
                )
                extraction_time = time.time() - start_time
                extractions[model] = json.loads(response['response'])
                print(f"   - {model} completed in {extraction_time:.1f}s")
            except Exception as e:
                logging.error(f"Extraction failed with {model}: {str(e)}")
                print(f"!! Extraction failed with {model}: {str(e)}")
                extractions[model] = {"error": str(e)}
        
        print("=== Processing complete! ===")
        return {
            "filename": filename,
            "text": text[:500] + "..." if len(text) > 500 else text,
            "extractions": extractions
        }

class Evaluator:
    """Handles evaluation against ground truth and metric calculations"""
    
    @staticmethod
    def find_ground_truth_file(base_name: str) -> str:
        """
        Find matching ground truth file in the ground_truth directory
        
        Args:
            base_name: Filename without extension (e.g., "john_doe")
            
        Returns:
            Path to the ground truth file
            
        Raises:
            FileNotFoundError: If no matching file is found
        """
        for filename in os.listdir(GROUND_TRUTH_FOLDER):
            if filename.lower().startswith(base_name.lower()) and filename.endswith(".json"):
                return os.path.join(GROUND_TRUTH_FOLDER, filename)
        raise FileNotFoundError(f"No ground truth file found for: {base_name}")

    @staticmethod
    def compare_extractions(extractions: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        Compare extracted data with ground truth and calculate metrics
        
        Args:
            extractions: Dictionary of extractions from different models
            ground_truth: Ground truth data
            
        Returns:
            Tuple of (comparisons, metrics_summary)
            - comparisons: List of field-by-field comparison results
            - metrics_summary: Calculated metrics for each model
        """
        comparisons = []
        metrics_summary = {}
        expected_fields = ["name", "email", "phone", "skills", "education", "experience"]

        for model, extracted_data in extractions.items():
            comparison = {"model": model}
            y_true = []
            y_pred = []

            for field in expected_fields:
                gt_val = ground_truth.get(field)
                ex_val = extracted_data.get(field)
                
                # Handle different field types
                if field in ["name", "email", "phone"]:
                    match = int(str(gt_val).lower() == str(ex_val).lower())
                else:  # List fields
                    if isinstance(gt_val, list) and isinstance(ex_val, list):
                        if field == "skills":
                            match = int(set(str(x).lower() for x in gt_val) == set(str(x).lower() for x in ex_val))
                        else:
                            match = int(gt_val == ex_val)
                    else:
                        match = 0
                
                comparison[f"{field}_match"] = match
                y_true.append(1)  # Ground truth is always correct
                y_pred.append(match)

            comparisons.append(comparison)
            
            # Calculate metrics
            metrics_summary[model] = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0)
            }

        return comparisons, metrics_summary

    @staticmethod
    def save_results(comparisons: List[Dict], metrics: Dict[str, Dict], filename: str) -> Tuple[str, str]:
        """
        Save evaluation results to CSV and JSON files
        
        Args:
            comparisons: Field-by-field comparison results
            metrics: Calculated metrics for each model
            filename: Original CV filename to use in output filenames
            
        Returns:
            Tuple of (csv_path, json_path) for saved files
        """
        base_name = os.path.splitext(filename)[0]
        csv_path = os.path.join(RESULTS_FOLDER, f"C:/Users/HP/Desktop/CV_extractor/results/Data_Analyst_comparison.csv{base_name}_comparison.csv")
        json_path = os.path.join(RESULTS_FOLDER, f"C:/Users/HP/Desktop/CV_extractor/results/{base_name}_metrics.json")
        
        # Save comparison CSV
        fieldnames = ["model"] + [f"{field}_match" for field in ["name", "email", "phone", "skills", "education", "experience"]]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparisons)
        
        # Save metrics JSON
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return csv_path, json_path

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main application route handling file upload and processing"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        try:
            # Save uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # Process CV
            processor = CVProcessor()
            result = processor.process_cv(file_path)
            
            # Load ground truth
            evaluator = Evaluator()
            try:
                base_name = os.path.splitext(file.filename)[0]
                gt_file = evaluator.find_ground_truth_file(base_name)
                with open(gt_file, 'r') as f:
                    ground_truth = json.load(f)
                
                # Compare with ground truth
                comparisons, metrics = evaluator.compare_extractions(result["extractions"], ground_truth)
                
                # Save results
                csv_path, json_path = evaluator.save_results(comparisons, metrics, file.filename)
                result.update({
                    "evaluation": {
                        "comparisons": comparisons,
                        "metrics": metrics,
                        "csv_path": os.path.basename(csv_path),
                        "json_path": os.path.basename(json_path)
                    }
                })
                
                return render_template('index.html', 
                                    result=result,
                                    show_evaluation=True,
                                    ground_truth=ground_truth)
            
            except FileNotFoundError as e:
                return render_template('index.html', 
                                    result=result,
                                    show_evaluation=False,
                                    warning=f"No ground truth found for comparison: {str(e)}")
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            return render_template('index.html', error=f"Processing failed: {str(e)}")
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Route for downloading result files"""
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Route for serving static files"""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    print("Starting CV Extractor Application")
    print(f"Available models: {', '.join(MODELS)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"OCR model: {OCR_MODEL}")
    print("\nServer running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)