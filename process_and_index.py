import json
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX") 

JSON_FILE_PATH = "scraped_articles.json"

#  safe limit for metadata size (in characters)
# pinecone's limit is 40960 bytes. We use 38000 characters as a safe buffer.
METADATA_CHARACTER_LIMIT = 38000

# The embedding model 
MODEL_NAME = 'all-MiniLM-L6-v2'
ARTICLES_NAMESPACE = "full-articles"
CHUNKS_NAMESPACE = "article-chunks"

# --- 2. INITIALIZE SERVICES ---
def initialize_services():
    """Initializes the embedding model and Pinecone connection."""
    print("Initializing services...")
    print(f"Loading embedding model: '{MODEL_NAME}'...")
    embedding_model = SentenceTransformer(MODEL_NAME, device='cpu')
    
    print(f"Connecting to Pinecone index: '{PINECONE_INDEX_NAME}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    time.sleep(1)
    print("Services initialized successfully.")
    print("-" * 50)
    return embedding_model, index


# --- 3. LOAD & PROCESS DATA ---
def load_data(file_path):
    """Loads the scraped article data from a JSON file."""
    print(f"Loading data from '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} articles.")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        return []

def process_and_upsert(articles, model, index):
    """Processes articles and upserts them into Pinecone using the Parent Chunking strategy."""
    
    BATCH_SIZE = 100
    
    # Text splitter for the small, searchable chunks
    searchable_chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)#one for searching
    
    # NEW: Text splitter for the large parent documents to ensure they fit in metadata
    parent_chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=METADATA_CHARACTER_LIMIT, chunk_overlap=0)#one for fitting data in pinecone

    # A sparse dummy vector to satisfy Pinecone's non-zero vector requirement
    dimension = model.get_sentence_embedding_dimension()
    dummy_vector = [0.0] * dimension
    dummy_vector[0] = 0.00001 
    
    searchable_chunks_batch = []
    articles_processed = 0

    for article in articles:
        url = article.get("url")
        content = article.get("content")

        if not url or not content:
            continue

        print(f"Processing article: {url}")
        
        # --- NEW: Step 1 - Split and store the full article in compliant parent chunks ---
        parent_chunks = parent_chunk_splitter.split_text(content)
        parent_vectors_to_upsert = []
        for i, parent_chunk_text in enumerate(parent_chunks):
            parent_id = f"{url}-part-{i}"
            parent_vectors_to_upsert.append((parent_id, dummy_vector, {"content": parent_chunk_text}))
        
        # Upsert the parent chunks to the ARTICLES namespace
        if parent_vectors_to_upsert:
            index.upsert(vectors=parent_vectors_to_upsert, namespace=ARTICLES_NAMESPACE)

        # --- Step 2 - Process and store the small, searchable chunks ---
        searchable_chunks = searchable_chunk_splitter.split_text(content)
        
        for i, chunk_text in enumerate(searchable_chunks):
            embedding = model.encode(chunk_text).tolist()
            chunk_id = f"{url}-chunk-{i}"
            
            metadata = {"source_url": url, "chunk_text": chunk_text}
            
            searchable_chunks_batch.append((chunk_id, embedding, metadata))
            
            # When the batch is full, upsert it to the CHUNKS namespace
            if len(searchable_chunks_batch) >= BATCH_SIZE:
                print(f" > Upserting a batch of {len(searchable_chunks_batch)} searchable chunks...")
                index.upsert(vectors=searchable_chunks_batch, namespace=CHUNKS_NAMESPACE)
                searchable_chunks_batch.clear()
        
        articles_processed += 1

    # Upsert any remaining vectors in the last batch
    if searchable_chunks_batch:
        print(f" > Upserting the final batch of {len(searchable_chunks_batch)} searchable chunks...")
        index.upsert(vectors=searchable_chunks_batch, namespace=CHUNKS_NAMESPACE)

    print(f"\n{'-'*50}\n✅ Finished processing. A total of {articles_processed} articles were indexed.")


# --- 4. MAIN EXECUTION ---
def main():
    """The main function to run the entire pipeline."""
    embedding_model, pinecone_index = initialize_services()
    all_articles = load_data(JSON_FILE_PATH)
    
    if all_articles:
        process_and_upsert(all_articles, embedding_model, pinecone_index)

if __name__ == "__main__":
    main()

# --- 5. EXAMPLE RETRIEVAL LOGIC (for your future RAG agent) ---
def retrieve_full_context(query, model, index):
    """An example of how to retrieve the full article context using the two-part storage."""
    print(f"\n--- Retrieving context for query: '{query}' ---")
    
    # 1. Search for the most relevant searchable CHUNK
    query_vector = model.encode(query).tolist()
    search_results = index.query(
        vector=query_vector,
        top_k=1,
        namespace=CHUNKS_NAMESPACE,
        include_metadata=True
    )
    
    if not search_results.matches:
        return "Sorry, I could not find any relevant information."
        
    # 2. Get the source URL from the best chunk's metadata
    source_url = search_results.matches[0].metadata['source_url']
    print(f" > Relevant source found: {source_url}")
    
    # 3. Fetch all PARENT CHUNKS associated with that URL
    full_text = ""
    i = 0
    while True:
        parent_id = f"{source_url}-part-{i}"
        fetch_result = index.fetch(ids=[parent_id], namespace=ARTICLES_NAMESPACE)
        
        # If the fetch result is empty, we've reached the end of the article
        if not fetch_result.vectors:
            break
            
        full_text += fetch_result.vectors[parent_id].metadata['content']
        i += 1
        
    print(f" > Reconstructed full article from {i} parent chunks.")
    return full_text