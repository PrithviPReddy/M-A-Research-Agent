import os
import re
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from neo4j import GraphDatabase

# --- 1. CONFIGURATION ---
load_dotenv()

# # Gemini API Configuration
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GENERATIVE_MODEL = 'gemini-2.5-pro'

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GENERATIVE_MODEL = "gpt-5-mini" # Or any other one , as per your choice

# Neo4j Database Configuration
# These are the default values for a local Neo4j Desktop database
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password" # password 

# Path to your source data
JSON_FILE_PATH = "scraped_articles.json"


# --- 2. INITIALIZE SERVICES ---
print("Initializing services...")

# # Configure Gemini API
# genai.configure(api_key=GOOGLE_API_KEY)
# llm_model = genai.GenerativeModel(GENERATIVE_MODEL)

# Configure OpenAI Client
llm_model = OpenAI(api_key=OPENAI_API_KEY)

# Configure Neo4j Driver
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print(" Neo4j connection successful.")
except Exception as e:
    print(f" Error connecting to Neo4j: {e}")
    exit()

print(" Services initialized successfully.")
print("-" * 50)


# --- 3. EXTRACTION AND GRAPH POPULATION LOGIC ---

# This is the specialized prompt to instruct Gemini to extract a knowledge graph
EXTRACTION_PROMPT = """
From the provided text, extract key entities and the relationships between them.
Your goal is to build a knowledge graph for M&A (Mergers and Acquisitions) analysis.

**Instructions:**
1.  Identify entities of the following types: Company, Person, Industry, FinancialValue.
2.  Identify the relationships between these entities. Key relationships include: ACQUIRED, IS_CEO_OF, OPERATES_IN, DEAL_VALUE_IS.
3.  Return the output as a single, valid JSON object containing two keys: "entities" and "relationships".
4.  Do not include any entities or relationships that are not explicitly mentioned in the text.

**Example JSON Output:**
{{
  "entities": [
    {{"name": "Microsoft", "type": "Company"}},
    {{"name": "Satya Nadella", "type": "Person"}},
    {{"name": "USD 621 Billion", "type": "FinancialValue"}}
  ],
  "relationships": [
    {{"source": "Satya Nadella", "target": "Microsoft", "type": "IS_CEO_OF"}},
    {{"source": "SomeDealEvent", "target": "USD 621 Billion", "type": "DEAL_VALUE_IS"}}
  ]
}}

**Text to Analyze:**
---
{text}
---
"""

def add_to_graph(tx, graph_data):
    """
    A function to write the extracted entities and relationships to the Neo4j graph.
    This function will be executed within a database transaction.
    """
    # CHANGE: Use .get() to safely access keys.
    # This prevents an error if the LLM returns a JSON without "entities" or "relationships".
    entities = graph_data.get("entities", [])
    relationships = graph_data.get("relationships", [])

    # Using MERGE to avoid creating duplicate nodes. It acts like "get or create".
    for entity in entities:
        # The first part (e.g., `e:Company`) sets the label for the node.
        tx.run("MERGE (e:%s {name: $name})" % entity['type'], name=entity['name'])

    for rel in relationships:
        # This Cypher query finds the source and target nodes and creates a relationship between them.
        tx.run("""
            MATCH (source {name: $source_name})
            MATCH (target {name: $target_name})
            MERGE (source)-[r:%s]->(target)
        """ % rel['type'], source_name=rel['source'], target_name=rel['target'])

def process_articles():
    """
    The main function to orchestrate the pipeline:
    Load articles -> Extract KG -> Populate Neo4j.
    """
    # 1. Load the articles from the source file
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        print(f"Successfully loaded {len(articles)} articles from '{JSON_FILE_PATH}'.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        return
# Set the article number you want to start from (e.g., 380)
# Remember that list indices start at 0, so article 380 is at index 379.
    start_index = 91

    for i, article in enumerate(articles[start_index:], start=start_index):
        content = article.get("content")
        url = article.get("url")
        if not content:
            continue

        print(f"\nProcessing article {i + 1}/{len(articles)}: {url}")

        try:
            # 2a. Send content to OpenAI for extraction
            response = llm_model.chat.completions.create(
                model=GENERATIVE_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Your task is to extract a knowledge graph from the user's text and respond ONLY with a valid JSON object."},
                    {"role": "user", "content": EXTRACTION_PROMPT.format(text=content)}
                ],
                response_format={"type": "json_object"} # This forces the model to return JSON
            )

            # Get the JSON string directly from the response
            json_response_text = response.choices[0].message.content
            graph_data = json.loads(json_response_text)
            # --- END OF CHANGE ---

            # 2b. Write the extracted data to Neo4j
            with neo4j_driver.session() as session:
                session.write_transaction(add_to_graph, graph_data)
            
            print(f" > Successfully extracted and added {len(graph_data.get('entities',[]))} entities and {len(graph_data.get('relationships',[]))} relationships to the graph.")
            
            time.sleep(1) 

        except json.JSONDecodeError:
            print(f" >  Warning: Could not decode JSON from the LLM's response, even after cleaning. Skipping this article.")
        except Exception as e:
            print(f" > An unexpected error occurred: {e}")
            # --- ADD THESE 3 LINES FOR DEBUGGING ---
            print("--- RAW FAULTY RESPONSE FROM GEMINI ---")
            print(response.text)
            print("--- END OF RAW RESPONSE ---")

    # 3. Close the database connection
    neo4j_driver.close()
    print("\n{'-'*50}\n Knowledge Graph build complete.")

    # 3. Close the database connection
    neo4j_driver.close()
    print("\n{'-'*50}\n Knowledge Graph build complete.")


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    process_articles()
