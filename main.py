# main.py
import os
import json
import time
import datetime
import sys
import fitz  # PyMuPDF
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Constants and Configuration ---
RETRIEVER_PATH = 'models/all-mpnet-base-v2'
GENERATOR_PATH = 'models/t5-small'
EMBEDDING_DIM = 768

def clean_generated_text(text: str) -> str:
    match = re.search(r'[a-zA-Z0-9]', text)
    if match:
        return text[match.start():]
    return text

def parse_document_sections(folder_path: str) -> list:
    sections = []
    print(f"--- Parsing documents from: {folder_path} ---")
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.pdf'):
            continue
        
        doc = fitz.open(os.path.join(folder_path, filename))
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks", sort=True)
            for i, block in enumerate(blocks):
                text_content = block[4].strip()
                if len(text_content) > 100:
                    sections.append({
                        "document_name": filename, "page_number": page_num + 1,
                        "block_id": f"p{page_num+1}-b{i}", "content_text": text_content.replace("\n", " ")
                    })
    print(f"Parsed {len(sections)} text blocks.")
    return sections

def extract_key_topics(sections: list, top_n: int = 4) -> list:
    print("--- Dynamically extracting key topics ---")
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=100)
    corpus = [s['content_text'] for s in sections]
    vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    return list(feature_names[-top_n:])

class DocumentIntelligenceSystem:
    def __init__(self):
        print("--- Initializing Document Intelligence System ---")
        self.retriever = SentenceTransformer(RETRIEVER_PATH)
        self.tokenizer = T5Tokenizer.from_pretrained(GENERATOR_PATH)
        self.generator = T5ForConditionalGeneration.from_pretrained(GENERATOR_PATH)
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.section_references = []
        print("System initialized successfully.")

    def build_index(self, sections: list):
        print("--- Building semantic index ---")
        self.index.reset()
        self.section_references = sections
        embeddings = self.retriever.encode([s['content_text'] for s in sections], show_progress_bar=True)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def search(self, query: str, top_k: int) -> list:
        query_embedding = self.retriever.encode([query])
        _, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        return [self.section_references[idx] for idx in indices[0]]

    def analyze_and_refine(self, content_text: str) -> str:
        prompt = f"summarize: {content_text}"
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.generator.generate(inputs.input_ids, num_beams=4, min_length=40, max_length=150, early_stopping=True)
        return clean_generated_text(self.tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    def generate_section_title(self, content_text: str) -> str:
        prompt = f"Generate a short, concise title for the following text: {content_text}"
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        title_ids = self.generator.generate(inputs.input_ids, num_beams=2, max_length=20, early_stopping=True)
        return clean_generated_text(self.tokenizer.decode(title_ids[0], skip_special_tokens=True))

def process_collection(collection_name: str, system: DocumentIntelligenceSystem):
    """Runs the end-to-end analysis pipeline for a single collection."""
    print(f"\n{'='*20} PROCESSING COLLECTION: {collection_name} {'='*20}")
    
    collection_path = collection_name
    docs_folder = os.path.join(collection_path, "PDFs")
    query_file = os.path.join(collection_path, "challenge1b_input.json")
    output_file = os.path.join(collection_path, "challenge1b_output.json")

    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
        persona_dict = query_data.get("persona", {})
        job_dict = query_data.get("job_to_be_done", {})
        persona = persona_dict.get("role", "")
        job_to_be_done = job_dict.get("task", "")
    except Exception as e:
        print(f"Error reading query file at '{query_file}': {e}")
        return

    parsed_sections = parse_document_sections(docs_folder)
    if not parsed_sections:
        print(f"No sections parsed for {collection_name}. Skipping.")
        return
        
    system.build_index(parsed_sections)
    
    main_query = f"{persona} {job_to_be_done}"
    sub_queries = [main_query] + extract_key_topics(parsed_sections)
    print(f"Using {len(sub_queries)} dynamically generated queries: {sub_queries}")
    
    all_results = []
    seen_ids = set()
    for query in sub_queries:
        k = 3 if query == main_query else 1
        results = system.search(query, top_k=k)
        for res in results:
            unique_id = f"{res['document_name']}-{res['block_id']}"
            if unique_id not in seen_ids:
                all_results.append(res)
                seen_ids.add(unique_id)

    print(f"\nFound {len(all_results)} unique, relevant sections.")
    
    extracted_sections_output = []
    subsection_analysis_output = []
    
    print("\n--- Generating titles and refining top sections ---")
    for i, section in enumerate(all_results):
        refined_text = system.analyze_and_refine(section["content_text"])
        generated_title = system.generate_section_title(refined_text)
        
        extracted_sections_output.append({
            "document": section["document_name"], "section_title": generated_title,
            "importance_rank": i + 1, "page_number": section["page_number"]
        })
        
        subsection_analysis_output.append({
            "document": section["document_name"], "refined_text": refined_text,
            "page_number": section["page_number"]
        })

    final_output = {
        "metadata": {"input_documents": [f for f in os.listdir(docs_folder) if f.lower().endswith('.pdf')], "persona": persona, "job_to_be_done": job_to_be_done, "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": subsection_analysis_output
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- Analysis complete for {collection_name}. Results saved to '{output_file}'. ---")

def main():
    start_time = time.time()
    
    system = DocumentIntelligenceSystem()
    
    # Find all directories in the current path that start with "Collection"
    collection_folders = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('Collection')]
    
    if not collection_folders:
        print("No collection folders found. Please create folders like 'Collection 1', 'Collection 2', etc.")
        sys.exit(1)
        
    print(f"Found {len(collection_folders)} collections to process: {collection_folders}")
    
    for collection_name in collection_folders:
        process_collection(collection_name, system)
        
    print(f"\nProcessed all collections in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
