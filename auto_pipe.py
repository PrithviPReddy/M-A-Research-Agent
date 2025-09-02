import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# --- 1. CONFIGURATION ---
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

# Scraping Configuration
BASE_WEBSITE_URL = "https://imaa-institute.org/publications/"
# Set the number of pages to check for new articles. 
# For daily runs, checking the first 5-10 pages is usually enough.
PAGES_TO_CHECK = 3

# Pinecone & Model Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
METADATA_CHARACTER_LIMIT = 38000
ARTICLES_NAMESPACE = "full-articles"
CHUNKS_NAMESPACE = "article-chunks"


# --- 2. INITIALIZE SERVICES (runs only once) ---
def initialize_services():
    """Initializes all necessary services for the pipeline."""
    print("Initializing services...")
    
    # Embedding Model
    print(f"Loading embedding model: '{EMBEDDING_MODEL}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    
    # Pinecone
    print(f"Connecting to Pinecone index: '{PINECONE_INDEX_NAME}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    time.sleep(1)
    print(" Services initialized successfully.")
    return embedding_model, index


# --- 3. CORE PIPELINE FUNCTIONS ---

def get_processed_urls_from_pinecone(index, model):
    """
    Fetches all unique source URLs already present in the Pinecone index.
    This replaces the need for a local all_links.json file.
    """
    print("Fetching list of already processed URLs from Pinecone...")
    processed_urls = set()
    # Query with a dummy vector to get a large sample of vectors.
    # Note: For massive indexes, a more robust pagination method would be needed.
    dimension = model.get_sentence_embedding_dimension()
    dummy_query = [0.0] * dimension
    
    try:
        query_response = index.query(
            vector=dummy_query,
            top_k=10000, # Max allowed value
            namespace=CHUNKS_NAMESPACE,
            include_metadata=False # We only need the IDs
        )
        for match in query_response['matches']:
            # The ID is in the format "https://...-chunk-i", we need to extract the base URL.
            base_url = "-chunk-".join(match['id'].split('-chunk-')[:-1])
            processed_urls.add(base_url)
        print(f"Found {len(processed_urls)} existing URLs in the database.")
    except Exception as e:
        print(f"Warning: Could not fetch existing URLs from Pinecone. Will proceed assuming none exist. Error: {e}")
        
    return processed_urls

def scrape_and_process_new_articles(index, model, processed_urls):
    """
    Crawls the website, finds new articles, scrapes them, and upserts them directly.
    """
    print("\n--- Starting Scraping and Processing Phase ---")
    
    # Create the list of paginated URLs to check
    paginated_urls = [BASE_WEBSITE_URL]
    paginated_urls.extend([f"{BASE_WEBSITE_URL}?e-page-8fbddee={i}" for i in range(2, PAGES_TO_CHECK + 1)])
    
    new_articles_found = 0
    for page_url in paginated_urls:
        print(f"\nCrawling page: {page_url}")
        
        # Step 1: Get all article links from the current page
        try:
            response = requests.get(page_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            link_elements = soup.find_all('a', class_='elementor-post__read-more')
        except requests.RequestException as e:
            print(f"   > Error fetching page: {e}. Skipping.")
            continue
            
        # Step 2: For each link, check if it's new. If so, scrape and process.
        for link_element in link_elements:
            href = link_element.get('href')
            if not href:
                continue
            
            absolute_url = urljoin(page_url, href)
            
            # THE CORE LOGIC: Skip if we've already processed this URL
            if absolute_url in processed_urls:
                continue
            
            # We found a new article!
            print(f"   > Found new article: {absolute_url}")
            
            # Step 2a: Scrape its content
            try:
                article_response = requests.get(absolute_url)
                article_response.raise_for_status()
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                # Using the more robust selector for content
                content_element = article_soup.find('div', class_='elementor-widget-theme-post-content')
                if not content_element:
                    print(f"     > Could not find content for article. Skipping.")
                    continue
                content = ' '.join(content_element.get_text().split())
            except requests.RequestException as e:
                print(f"     > Error scraping content: {e}. Skipping.")
                continue

            # Step 2b: Process and upsert the new article directly
            _process_and_upsert_single_article(absolute_url, content, index, model)
            processed_urls.add(absolute_url) # Add to our set to avoid re-processing in this run
            new_articles_found += 1
            time.sleep(1) # Be polite to the server
            
    print(f"\n--- Pipeline Run Complete ---")
    print(f"Found and processed {new_articles_found} new articles.")


def _process_and_upsert_single_article(url, content, index, model):
    """A helper function to handle the indexing logic for one article."""
    print(f"     > Processing and indexing...")
    
    # Text splitters
    parent_chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=METADATA_CHARACTER_LIMIT, chunk_overlap=0)
    searchable_chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Dummy vector for parent chunks
    dimension = model.get_sentence_embedding_dimension()
    dummy_vector = [0.0] * dimension
    dummy_vector[0] = 0.00001

    # Upsert parent chunks
    parent_chunks = parent_chunk_splitter.split_text(content)
    parent_vectors = [(f"{url}-part-{i}", dummy_vector, {"content": text}) for i, text in enumerate(parent_chunks)]
    index.upsert(vectors=parent_vectors, namespace=ARTICLES_NAMESPACE)

    # Upsert searchable chunks
    searchable_chunks = searchable_chunk_splitter.split_text(content)
    searchable_vectors = []
    for i, chunk_text in enumerate(searchable_chunks):
        embedding = model.encode(chunk_text).tolist()
        metadata = {"source_url": url, "chunk_text": chunk_text}
        searchable_vectors.append((f"{url}-chunk-{i}", embedding, metadata))
    
    # Upsert in batches if necessary (though for a single article, it's usually one batch)
    if searchable_vectors:
        index.upsert(vectors=searchable_vectors, namespace=CHUNKS_NAMESPACE)
    print(f"     > Successfully indexed.")


# --- 4. MAIN EXECUTION ---
def main():
    """Main function to run the entire automated pipeline."""
    embedding_model, pinecone_index = initialize_services()
    processed_urls = get_processed_urls_from_pinecone(pinecone_index, embedding_model)
    scrape_and_process_new_articles(pinecone_index, embedding_model, processed_urls)

if __name__ == "__main__":
    main()




# how to run the file and then set a timer to run the file and then execute the pipe 


# 1. Open the Crontab Editor:
# Open your terminal and type:



# crontab -e

# 2. Add the Scheduled Task:
# This will open a text editor. Go to the bottom of the file and add the following line. You must use the full paths to your Python interpreter and your script.

# Code snippet

# # Run the M&A Research Agent data pipeline every day at 2:00 AM
# 0 2 * * * /full/path/to/your/myenv/bin/python /full/path/to/your/automated_pipeline.py >>/full/path/to/your/pipeline.log 2>&1
