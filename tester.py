from neo4j import GraphDatabase
import os

# --- Configuration ---
# Make sure these match your Neo4j setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://172.28.64.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# --- Connection Test ---
driver = None  # Initialize driver to None
try:
    # Attempt to create a driver instance
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Check if the connection is valid
    driver.verify_connectivity()
    
    print("✅ Connection to Neo4j was successful!")

except Exception as e:
    print(f"❌ Failed to connect to Neo4j. Error: {e}")

finally:
    # Always close the driver connection when done
    if driver:
        driver.close()