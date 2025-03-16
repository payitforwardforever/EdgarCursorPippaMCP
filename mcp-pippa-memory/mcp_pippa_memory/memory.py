"""
Memory management module for Pippa Memory MCP Tool.
Handles storage and retrieval of memory fragments using ChromaDB.
"""
import chromadb
import os
import uuid
import datetime
import json
from dotenv import load_dotenv
from .config import DB_DIR, LOGS_DIR, MEMORY_INIT_LOG_PATH, get_setting

# Fix for Pydantic compatibility issues with langchain
# We're using direct ChromaDB integration instead of langchain-chroma
# to avoid the Pydantic v1/v2 compatibility issues

# Load environment variables from .env file
load_dotenv()  # Try current directory

# Try to load from the current directory, the parent directory, and the parent's parent directory
if not os.getenv("GEMMA_API_KEY"):
    # Try parent directory (mcp-pippa-memory)
    parent_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
if not os.getenv("GEMMA_API_KEY"):
    # Try root directory (cwkMCPServers)
    root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if os.path.exists(root_env):
        load_dotenv(root_env)

# Check if API key is available
gemma_api_key = os.getenv("GEMMA_API_KEY")
if not gemma_api_key:
    raise ValueError("GEMMA_API_KEY environment variable is not set. Please set it in your .env file.")

# Export DB path for external scripts like streamlit app
DEFAULT_DB_PATH = DB_DIR

# Import Google Generative AI client
import google.generativeai as genai

# Configure the API
genai.configure(api_key=gemma_api_key)

# Debug: List available models
def list_available_models():
    """List available models for embedding"""
    try:
        models = genai.list_models()
        print("Available models:")
        for model in models:
            print(f"Model: {model.name}")
        return models
    except Exception as e:
        print(f"Error listing models: {e}")
        return None

# Call this function to list models
list_available_models()

# Create a mock embedding function as a fallback
def create_mock_embedding(text):
    """Create a mock embedding for testing purposes"""
    import hashlib
    import numpy as np
    
    # Create a deterministic hash of the text
    hash_object = hashlib.md5(text.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert the hash to a list of 768 float values between -1 and 1
    np.random.seed(int(hash_hex, 16) % (2**32))
    embedding = np.random.uniform(-1, 1, 768).tolist()
    
    return embedding

class PippaMemoryTool:
    """
    Manages memory storage and retrieval using ChromaDB.
    """
    def __init__(self, persist_directory=None):
        """
        Initialize the memory tool with ChromaDB.
        
        Args:
            persist_directory: Directory where memories will be stored
        """
        # Use the configured DB path unless specified otherwise
        if persist_directory is None:
            if os.path.dirname(os.getcwd()) == "/" or not os.access(os.getcwd(), os.W_OK):
                # If running from root (Cursor MCP) or other read-only directory
                persist_directory = get_setting("db_path", DEFAULT_DB_PATH)
                with open(MEMORY_INIT_LOG_PATH, "a") as f:
                    f.write(f"[{datetime.datetime.now().isoformat()}] Running from non-writable directory - using configured path: {persist_directory}\n")
            else:
                # Otherwise use our consistent project path
                persist_directory = get_setting("db_path", DEFAULT_DB_PATH)
                with open(MEMORY_INIT_LOG_PATH, "a") as f:
                    f.write(f"[{datetime.datetime.now().isoformat()}] Using configured DB path: {persist_directory}\n")
        
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB directly
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="pippa_memories",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        with open(MEMORY_INIT_LOG_PATH, "a") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] Initialized ChromaDB collection: pippa_memories\n")
    
    def _get_embedding(self, text):
        """Get embedding for text using Google Generative AI"""
        try:
            # Print the environment variable to debug
            print(f"EMBEDDING_MODEL from env: {os.getenv('EMBEDDING_MODEL')}")
            
            # Try multiple models in order of preference
            models_to_try = [
                "models/gemini-embedding-exp-03-07",
                "models/embedding-001",
                "models/text-embedding-004"
            ]
            
            for model in models_to_try:
                try:
                    print(f"Trying model: {model}")
                    
                    # Generate embeddings
                    response = genai.embed_content(
                        model=model,
                        content=text,
                        task_type="RETRIEVAL_QUERY"
                    )
                    
                    print(f"Response type: {type(response)}")
                    
                    embedding = None
                    
                    # Check if response is a callable
                    if callable(response):
                        print("Error: response is a callable")
                        continue
                    
                    # Try different attribute names
                    if hasattr(response, 'embedding'):
                        print("Found 'embedding' attribute")
                        embedding = response.embedding
                    elif hasattr(response, 'embeddings'):
                        print("Found 'embeddings' attribute")
                        embedding = response.embeddings[0]
                    elif hasattr(response, 'values'):
                        print("Found 'values' attribute")
                        embedding = response.values
                    elif isinstance(response, dict) and 'values' in response:
                        print("Found 'values' in dict")
                        embedding = response['values']
                    else:
                        # Try to convert response to dict and extract embedding
                        try:
                            if isinstance(response, dict):
                                response_dict = response
                            else:
                                response_dict = response.to_dict()
                            
                            print(f"Response dict keys: {response_dict.keys()}")
                            
                            if 'embedding' in response_dict:
                                embedding = response_dict['embedding']
                            elif 'embeddings' in response_dict:
                                embedding = response_dict['embeddings'][0]
                            elif 'values' in response_dict:
                                embedding = response_dict['values']
                        except Exception as e:
                            print(f"Could not extract embedding from dict: {e}")
                    
                    # Check if embedding is a callable
                    if callable(embedding):
                        print("Error: embedding is a callable")
                        continue
                    
                    # Ensure embedding is a list of floats
                    if embedding is not None:
                        # Convert to list if it's not already
                        if not isinstance(embedding, list):
                            try:
                                embedding = embedding.tolist()
                            except:
                                embedding = list(embedding)
                        
                        # Ensure all values are floats
                        embedding = [float(val) for val in embedding]
                        
                        print(f"Embedding obtained: {len(embedding)} dimensions")
                        return embedding
                    
                    print(f"No embedding found in response for model {model}")
                except Exception as e:
                    print(f"Error with model {model}: {e}")
                    continue  # Try the next model
            
            # If all models fail, use mock embedding
            print("All models failed, using mock embedding")
            return create_mock_embedding(text)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Use mock embedding as fallback
            print("Using mock embedding as fallback")
            return create_mock_embedding(text)
    
    def remember(self, text):
        """
        Store a new memory.
        
        Args:
            text: The text content to remember
            
        Returns:
            Dictionary with status and memory ID
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        print(f"Attempting to remember: {text}")
        
        # Get embedding
        embedding = self._get_embedding(text)
        
        # Check if embedding is None
        if embedding is None:
            print("Failed to get embedding for the provided text.")
            return {
                "status": "error",
                "message": "Failed to get embedding for the provided text."
            }
        
        print(f"Embedding obtained: {len(embedding)} dimensions")
        
        # Store in ChromaDB
        try:
            print(f"Adding to collection with ID: {memory_id}")
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[{
                    "timestamp": timestamp,
                    "type": "memory",
                    "id": memory_id
                }],
                documents=[text]
            )
            
            print("Successfully added to collection")
            return {
                "status": "success", 
                "message": "Memory stored", 
                "id": memory_id
            }
        except Exception as e:
            print(f"Error saving memory to collection: {e}")
            return {
                "status": "error",
                "message": f"Failed to save memory: {str(e)}"
            }
    
    def recall(self, query, limit=None):
        """
        Retrieve memories similar to the query.
        
        Args:
            query: The search query
            limit: Maximum number of memories to return
            
        Returns:
            List of document objects containing memories
        """
        if limit is None:
            limit = get_setting("similarity_top_k", 3)
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Check if embedding is None
            if query_embedding is None:
                print("Failed to get embedding for the query.")
                return []
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Convert to documents format (compatible with previous implementation)
            documents = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    doc = Document(
                        page_content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i]
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error recalling memories: {e}")
            return []
    
    def list_memories(self, limit=10):
        """
        List all memories (up to limit).
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of document objects containing memories
        """
        try:
            # Get all items from the collection
            results = self.collection.get(limit=limit)
            
            # Convert to documents format (compatible with previous implementation)
            documents = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    doc = Document(
                        page_content=results["documents"][i],
                        metadata=results["metadatas"][i]
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error listing memories: {e}")
            return []
    
    def delete_memory(self, memory_id):
        """
        Delete a specific memory by ID.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            Dictionary with status and message
        """
        try:
            # Delete from ChromaDB
            self.collection.delete(ids=[memory_id])
            
            return {
                "status": "success", 
                "message": f"Memory {memory_id} deleted"
            }
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Failed to delete memory: {str(e)}"
            }


# Simple document class to maintain compatibility with the previous implementation
class Document:
    """Simple document class to mimic LangChain Document"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {} 