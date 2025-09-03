# M-A-Research-Agent
M&amp;A Research Agent

# AI M&A Research Agent

This project is an advanced, end-to-end AI research agent designed to automate Mergers & Acquisitions (M&A) due diligence. It leverages a hybrid knowledge base, combining semantic search with a structured knowledge graph, to provide nuanced, conversational answers, generate detailed reports, and answer complex factual queries.

The system features an automated data pipeline that scrapes financial articles, processes them into a multi-modal knowledge base, and serves the insights through an interactive web application.



---

## ## Key Features

* **Automated Data Pipeline**: A script (`automated_pipeline.py`) that runs on a schedule to find and scrape new articles, ensuring the agent's knowledge base is always up-to-date.
* **Hybrid Knowledge Base**: The agent's core strength. It combines two powerful data retrieval methods:
    * **Vector Database (Pinecone)**: For semantic, conceptual, and inferential questions, powered by a state-of-the-art RAG (Retrieval-Augmented Generation) pipeline.
    * **Knowledge Graph (Neo4j)**: For structured, factual, and multi-step questions. An LLM-powered pipeline extracts entities (Companies, People, etc.) and relationships (ACQUIRED, IS_CEO_OF, etc.) from articles and stores them in a graph.
* **Intelligent Query Router**: An LLM-based router that analyzes the user's question and intelligently decides whether to query the vector database or the knowledge graph.
* **Multi-Agent Web Interface (Gradio)**:
    * **Q&A Chatbot**: A conversational agent that can synthesize information from multiple sources, provide predictive analysis, and cite its sources.
    * **Report Generator**: A tool to generate detailed, structured reports (e.g., executive summaries, risk analyses) on specific, user-selected articles.
* **Advanced Prompt Engineering**: Specialized prompts designed to enable "analyst mode" for predictive questions, "Text-to-Cypher" for natural language graph queries, and structured report generation.

---

## ## Technology Stack

* **Core Language**: Python
* **AI & Machine Learning**:
    * **LLMs**: Google Gemini / OpenAI GPT
    * **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
    * **Text Processing**: `LangChain` (for intelligent chunking)
    * **Core ML Framework**: `torch`
* **Databases**:
    * **Vector DB**: Pinecone
    * **Graph DB**: Neo4j (with Cypher query language)
* **Data Ingestion**:
    * **Web Scraping**: `requests`, `BeautifulSoup`
* **Application & UI**:
    * **Web Framework**: Gradio
* **Environment**:
    * `dotenv` for managing credentials.

---

## ## Setup and Installation

Follow these steps to set up and run the project locally.

### ### Prerequisites

* Python 3.10+
* [Neo4j Desktop](https://neo4j.com/download/) installed and a local database running.

### ### 1. Clone the Repository

```bash
git clone <your-github-repo-url>
cd <your-repo-name>
```

### ### 2. Set Up a Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

### ### 3. Install Dependencies

Install all required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### ### 4. Configure Environment Variables

Create a file named `.env` in the root of your project directory and add your API keys and database credentials.

```env
# Pinecone
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
PINECONE_INDEX="your-pinecone-index-name"

# Google Gemini
GOOGLE_API_KEY="YOUR_GOOGLE_AI_KEY"

# Neo4j Database
# (Use your Windows Host IP if running the script in WSL)
NEO4J_URI="bolt://localhost:7687" 
NEO4J_USER="neo4j"
NEO4J_PASSWORD="YOUR_NEO4J_DATABASE_PASSWORD"
```

### ### 5. Run the Pipelines

**First, build the knowledge base.** You only need to do this once, or whenever you want to add new data.

```bash
# This script finds new articles and adds them to both Pinecone and Neo4j
python automated_pipeline.py
```

**Then, launch the application.**

```bash
# This script starts the Gradio web interface
python app.py
```

Open the local URL provided in your terminal (e.g., `http://127.0.0.1:7860`) to interact with your AI Research Agent.
