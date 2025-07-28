# setup_project.py
import os
import subprocess
import sys
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- 1. Define Project Structure and Files ---
PROJECT_DIRS = [
    "Collection 1/PDFs",
    "Collection 2/PDFs",
    "Collection 3/PDFs",
    "models"
]
REQUIREMENTS_CONTENT = """
torch --index-url https://download.pytorch.org/whl/cpu
transformers
sentence-transformers
faiss-cpu
pymupdf
sentencepiece
numpy<2
scikit-learn
"""

# --- 2. Model Configuration ---
RETRIEVER_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
GENERATOR_MODEL_NAME = 't5-small'
RETRIEVER_PATH = os.path.join('models', 'all-mpnet-base-v2')
GENERATOR_PATH = os.path.join('models', 't5-small')

def setup_directories():
    """Create necessary directories for all collections."""
    print("--- Setting up directories ---")
    for dir_name in PROJECT_DIRS:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Directory '{dir_name}' ensured.")

def create_requirements():
    """Create requirements.txt file."""
    print("\n--- Creating requirements.txt ---")
    with open('requirements.txt', 'w') as f:
        f.write(REQUIREMENTS_CONTENT.strip())
    print("requirements.txt created.")

def install_packages():
    """Install packages from requirements.txt."""
    print("\n--- Installing required packages ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during package installation: {e}")

def download_models():
    """Download and save the necessary models from Hugging Face."""
    print("\n--- Downloading and saving models (this may take a while) ---")
    if not os.path.exists(RETRIEVER_PATH):
        print(f"Downloading retriever model: {RETRIEVER_MODEL_NAME}...")
        retriever = SentenceTransformer(RETRIEVER_MODEL_NAME)
        retriever.save(RETRIEVER_PATH)
        print(f"Retriever saved to '{RETRIEVER_PATH}'.")
    else:
        print(f"Retriever already exists at '{RETRIEVER_PATH}'.")

    if not os.path.exists(GENERATOR_PATH):
        print(f"Downloading generator model: {GENERATOR_MODEL_NAME}...")
        tokenizer = T5Tokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        generator = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL_NAME)
        generator.save_pretrained(GENERATOR_PATH)
        tokenizer.save_pretrained(GENERATOR_PATH)
        print(f"Generator saved to '{GENERATOR_PATH}'.")
    else:
        print(f"Generator already exists at '{GENERATOR_PATH}'.")

if __name__ == "__main__":
    setup_directories()
    create_requirements()
    install_packages()
    download_models()
    print("\n--- Project setup complete! ---")
