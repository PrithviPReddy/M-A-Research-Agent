import os
import json
import time
import gradio as gr
from pinecone import Pinecone
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
# NEW: Neo4j Credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Models and Namespaces
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
GENERATIVE_MODEL = 'gemini-2.5-pro'
ARTICLES_NAMESPACE = "full-articles"
CHUNKS_NAMESPACE = "article-chunks"
JSON_FILE_PATH = "scraped_articles.json"
CONFIDENCE_THRESHOLD = 0.5


# --- 2. INITIALIZE SERVICES (runs only once) ---
print("Initializing services...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel(GENERATIVE_MODEL)

# Initialize Neo4j Driver
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print("Neo4j connection successful.")
except Exception as e:
    print(f" Error connecting to Neo4j: {e}")
    neo4j_driver = None

print(" Services initialized successfully.")
print("-" * 50)


# --- 3. HELPER FUNCTIONS (No changes here) ---

def _reconstruct_article_from_pinecone(url, index_instance):
    """Helper function to fetch all parent chunks and reconstruct a full article."""
    full_text = ""
    i = 0
    while True:
        parent_id = f"{url}-part-{i}"
        fetch_result = index_instance.fetch(ids=[parent_id], namespace=ARTICLES_NAMESPACE)
        if not fetch_result.vectors:
            break
        full_text += fetch_result.vectors[parent_id].metadata['content']
        i += 1
    return full_text

def get_all_article_urls():
    """Reads the source JSON to get a list of all article URLs for the dropdown."""
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [item['url'] for item in data]
    except FileNotFoundError:
        return ["Error: scraped_articles.json not found."]


# --- 4. CORE AGENT LOGIC ---

# --- Q&A (Vector Search) AGENT FUNCTIONS ---
def run_semantic_search(query):
    """The existing RAG pipeline using Pinecone for semantic search."""
    print(f"\n1a. Running SEMANTIC search for: '{query}'")
    query_vector = embedding_model.encode(query).tolist()
    
    search_results = pinecone_index.query(
        vector=query_vector, top_k=5, namespace=CHUNKS_NAMESPACE, include_metadata=True
    )
    
    if not search_results.matches or search_results.matches[0].score < CONFIDENCE_THRESHOLD:
        urls = {match.metadata['source_url'] for match in search_results.matches}
        idk_message = "I couldn't find a confident answer in the documents. However, these are the 5 most closely related articles I found:\n" + "\n".join(f"- {url}" for url in urls)
        return idk_message

    top_urls = {match.metadata['source_url'] for match in search_results.matches[:3]}
    
    context = ""
    for url in top_urls:
        article_text = _reconstruct_article_from_pinecone(url, pinecone_index)
        context += f"\n\n--- Article from {url} ---\n{article_text}"
    
    prompt = f"""**Role:** You are an expert M&A analyst and strategic advisor.

    **Task:** Your task is to analyze the provided articles to form a reasoned, forward-looking projection or opinion in response to the user's question. 
    While the provided text may not contain a direct answer, you must use the trends, data points, and expert opinions within it as the foundation for your analysis. 
    Do not introduce external information.

    **Output Format:**
    1.  **Disclaimer:** Start your response with a brief disclaimer stating that this is a projection based on the provided data, not a certainty.
    2.  **Analysis:** Provide your detailed analysis and prediction in a clear, structured manner.
    3.  **Reasoning:** After your analysis, create a "Reasoning" section. For each key point in your analysis, cite the source URL and provide a brief quote or data point from the articles that supports that part of your reasoning.

    ---
    **Provided Articles:**
    {context}
    ---
    **User's Question:**
    {query}
    """# Keeping your detailed prompt
    
    response = llm_model.generate_content(prompt)
    return response.text

# --- NEW: KNOWLEDGE GRAPH AGENT FUNCTIONS ---
def route_query(query):
    """Uses an LLM to classify the user's query."""
    print(f"\n1b. Routing query: '{query}'")
    prompt = f"""
    You are a query router. Your task is to classify the user's question into one of two categories: "semantic" or "graph".

    - Use "semantic" for broad, conceptual, or analytical questions (e.g., "what might happen if...", "what is the market sentiment...", "summarize trends...").
    - Use "graph" for specific, factual questions that involve relationships between entities like companies, people, or industries (e.g., "who acquired company X?", "list all CEOs in the telecom industry", "which companies were involved in deals over $1B?").

    Respond with ONLY the word "semantic" or "graph".

    User question: "{query}"
    """
    try:
        response = llm_model.generate_content(prompt)
        route = response.text.strip().lower()
        print(f"   > Route determined: '{route}'")
        return route
    except Exception as e:
        print(f"   > Error in routing, defaulting to semantic search. Error: {e}")
        return "semantic"

def generate_cypher_query(query):
    """Uses an LLM to convert a natural language question into a Cypher query."""
    print("2b. Generating Cypher query...")
    prompt = f"""
    You are an expert Neo4j developer. Your task is to convert a natural language question into a Cypher query based on the following graph schema.

    **Schema:**
    - Nodes: `Company`, `Person`, `Industry`, `FinancialValue`
    - Node Properties: All nodes have a `name` property.
    - Relationships: `ACQUIRED`, `IS_CEO_OF`, `OPERATES_IN`, `DEAL_VALUE_IS`

    **Instructions:**
    - Construct a Cypher query that answers the user's question.
    - The query should return a table of results.
    - Respond with ONLY the Cypher query. Do not add any explanation or formatting.

    **User Question:** "{query}"
    """
    try:
        response = llm_model.generate_content(prompt)
        cypher_query = response.text.strip().replace("```cypher", "").replace("```", "")
        print(f"   > Generated Cypher: {cypher_query}")
        return cypher_query
    except Exception as e:
        print(f"   > Error generating Cypher: {e}")
        return None

def execute_graph_query(cypher_query):
    """Connects to Neo4j, runs the Cypher query, and formats the result."""
    print("3b. Executing graph query...")
    if not neo4j_driver:
        return "Error: Neo4j connection not available."
    
    try:
        with neo4j_driver.session() as session:
            result = session.run(cypher_query)
            # Format the result into a markdown table or simple string
            records = [record.data() for record in result]
            if not records:
                return "I found no data in the knowledge graph that answers your question."
            
            # Simple formatting for the result
            formatted_result = ""
            for record in records:
                formatted_result += str(record) + "\n"
            print(f"   > Found {len(records)} records.")
            return formatted_result
    except Exception as e:
        print(f"   > Error executing Cypher query: {e}")
        return f"There was an error querying the knowledge graph: {e}"

# --- UPDATED: MAIN CHAT INTERFACE FUNCTION ---
def chat_interface_fn(message, history):
    """
    The main chat function with the new query router.
    """
    route = route_query(message)
    
    if "graph" in route:
        # If it's a graph question, run the KG pipeline
        cypher = generate_cypher_query(message)
        if cypher:
            return execute_graph_query(cypher)
        else:
            return "Sorry, I couldn't translate your question into a graph query."
    else:
        # Otherwise, run the existing semantic search pipeline
        return run_semantic_search(message)

# --- REPORT AGENT FUNCTION (No changes here) ---
def generate_report(article_url, report_topic):
    # This function remains the same as before
    print(f"\n1. Generating report on '{report_topic}' for article: {article_url}")
    article_content = _reconstruct_article_from_pinecone(article_url, pinecone_index)
    if not article_content:
        return f"Error: Could not retrieve content for the selected article."
    
    prompt = f"""
**Role:** You are a professional research analyst tasked with writing a detailed report.

**Task:** Based ONLY on the provided article text, write a comprehensive report on the following topic: "{report_topic}". The report should be well-structured, clear, and insightful, extracting all relevant information from the text.

**Output Format:**
1.  **Title:** Create a suitable title for the report.
2.  **Report Body:** Write the full report, using headings, subheadings, and bullet points as appropriate to structure the information.
3.  **Conclusion:** End with a brief concluding summary.

---
**Provided Article Text:**
{article_content}
---
""" # Keeping your detailed prompt
    
    response = llm_model.generate_content(prompt)
    return response.text

# --- 5. GRADIO UI WITH TABS ---
print("Launching Gradio Interface with Tabs...")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# M&A Research Agent ")
    
    with gr.Tabs():
        # --- Q&A Agent Tab ---
        with gr.TabItem("Hybrid Q&A Chatbot"):
            
            # --- START OF CHANGE ---
            # 1. Create a Chatbot component with a custom height
            chatbot_component = gr.Chatbot(height=600) # You can adjust this number

            # 2. Pass this component to the ChatInterface
            gr.ChatInterface(
                fn=chat_interface_fn,
                chatbot=chatbot_component, # This is the new argument
                title="Conversational Q&A with Knowledge Graph",
                description="Ask complex factual questions..."
            )

            
        with gr.TabItem("Report Generator"):
            # The report generator UI remains the same
            gr.Markdown("## Generate a Detailed Report from a Single Article")
            article_dropdown = gr.Dropdown(choices=get_all_article_urls(), label="Select an Article", interactive=True)
            report_topic_input = gr.Textbox(label="Report Topic", placeholder="e.g., Executive Summary, SWOT Analysis")
            generate_button = gr.Button("Generate Report", variant="primary")
            report_output = gr.Markdown(label="Generated Report")
            generate_button.click(fn=generate_report, inputs=[article_dropdown, report_topic_input], outputs=[report_output])

demo.launch(server_name="0.0.0.0", server_port=7861)