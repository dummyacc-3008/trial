import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def extract_txt_queswise(raw_text: str):
    """Splits text based on the +++ separator."""
    chunks = re.split(r"\++\n", raw_text)
    return chunks

print("Starting artifact generation...")

# --- 1. Load and Chunk Data ---
print("Loading GEN_AI 3.txt...")
with open("GEN_AI 3.txt", 'r', encoding='utf-8') as f:
    raw_text = f.read()

page_chunks = extract_txt_queswise(raw_text)
page_chunks = page_chunks[3:]
page_chunks_dicts = [{"text": chunk} for chunk in page_chunks]

# This is the list of pure text strings
text_chunks_list = [chunk['text'] for chunk in page_chunks_dicts]
print(f"Loaded and chunked {len(text_chunks_list)} documents.")

# --- 2. Save Text Chunks ---
with open("text_chunks.pkl", "wb") as f:
    pickle.dump(text_chunks_list, f)
print("Saved text_chunks.pkl successfully.")

# --- 3. Load Embedding Model ---
print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 4. Generate Embeddings ---
print("Generating embeddings for text chunks...")
# We encode the list of strings, not the list of dicts
embeddings = model.encode(
    text_chunks_list, 
    show_progress_bar=True, 
    convert_to_numpy=True
)
print(f"Generated {len(embeddings)} embeddings.")

# --- 5. Build and Save FAISS Index ---
embedding_matrix = np.array(embeddings).astype('float32')
dimension = embedding_matrix.shape[1]

print(f"Building FAISS index with dimension {dimension}...")
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

print(f"Saving FAISS index to GenAI.index...")
faiss.write_index(index, "GenAI.index")

print("---")
print("✅ All artifacts created and synchronized successfully! ---")
print(f"  - text_chunks.pkl ({len(text_chunks_list)} items)")
print(f"  - GenAI.index ({index.ntotal} vectors)")
print("\nYou can now run your Streamlit app.")