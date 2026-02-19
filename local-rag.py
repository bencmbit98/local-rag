# ============================================================================
# One Time Machine GitHub Setup
#   This section provides step-by-step instructions for setting up GitHub and VS Code
#   for your local RAG project. Follow these steps once per machine to enable version control 
#   and remote repository management.
#   Note: You can skip this setup if you already have Git and VS Code configured with GitHub.
#   For detailed instructions, see: 
#       https://docs.github.com/en/get-started/quickstart/set-up-git
#   For SSH key troubleshooting, see: 
#   https://docs.github.com/en/authentication/connecting-to-github-with-ssh/troubleshooting-ssh
#   Summary of Steps:
# =============================================================================
# Step 1: Install VS Code and Git
#   - Download and install Visual Studio Code: https://code.visualstudio.com/
#   - Download and install Git: https://git-scm.com/downloads   
#   - During Git installation, 
#       you can choose to use Git from the command line 
#       and also install Git Bash for a better terminal experience.

# Step 2: Create SSH key (once per machine)
#   bash: ssh-keygen -t ed25519 -C "your_email@example.com"

# Step 3: Add SSH key to GitHub account (copy ~/.ssh/id_ed25519.pub)
#   Add public key to GitHub: Github Settings > SSH and GPG keys > New SSH key
#   Note: If you have multiple SSH keys, you may need to configure ~/.ssh/config
#   Example ~/.ssh/config entry:
#   Host github.com
#       HostName github.com
#       User git
#       IdentityFile ~/.ssh/id_ed25519  
#   Note: After setup, test SSH connection with:
#       bash: ssh -T git@github.com

# Step 4: Starting a New Python Project (Inside VS Code)
#   a. Create a new folder for your project
#   b. Open that folder in VS Code

# Step 5: Create virtual environment
#   bash: python3 -m venv venv

# Step 6: Activate virtual environment
#   - On Windows: venv\Scripts\activate
#   - On macOS/Linux: source venv/bin/activate

# Step 7: Create .gitignore file in VS Code and add common Python ignores:
#   .gitignore content:
#       venv/
#       __pycache__/    
#       *.pyc
#       .vscode/

# Step 8: Install necessary Python packages (e.g., ollama, chromadb, pypdf)
#   bash: pip install ollama chromadb pypdf
#   bash: pip freeze > requirements.txt  # Save dependencies to requirements.txt

# Step 9: Handle OpenAI API Key Safely
#   a. Create a .env file in your project root (add to .gitignore)
#   b. Add your API key to .env:
#       OPENAI_API_KEY=your_openai_api_key_here 
#   c. Load .env variables in your Python code using python-dotenv:
#       from dotenv import load_dotenv
#       from openai import OpenAI
#       import os
#       load_dotenv()
#       client = OpenAI()
#       #?# api_key = os.getenv("OPENAI_API_KEY") 

#  Step 10: Push Project to GitHub (Inside VS Code)
#   a. Initialize Git repository: 
#       git init
#   b. Add files: 
#       git add .
#   c. Commit changes: 
#       git commit -m "Initial commit"
#   d. Create a new repository on GitHub (e.g., "my-local-rag-project")
#   e. Push to GitHub: 
#       git remote add origin https://github.com/your-username/your-repo-name.git
#   f. Push changes: 
#       git push -u origin main

#  Step 11: Using Project on Another Machine
#   a. Clone the repository:
#       git clone https://github.com/your-username/your-repo-name.git
#   b. Navigate to the project folder:
#       cd your-repo-name
#   c. Create and Activate the virtual environment:
#       python3 -m venv venv
#       - On Windows: venv\Scripts\activate
#       - On macOS/Linux: source venv/bin/activate
#   d. Install dependencies:
#       pip install -r requirements.txt
#   e. Remember to set up your .env file with the API key on the new machine as well.

#   Step 12 Daily Workflow (All Inside VS Code)
#   a. Pull latest changes from GitHub:
#       git pull origin main    
#   b. Make code changes and test locally
#   c. Add and commit changes:
#       git add .
#       git commit -m "Describe your changes here"
#   d. Push changes to GitHub:
#       git push origin main
#   
# ========================================================================





# ============================================================================
# Local RAG (Retrieval Augmented Generation) System
# Uses Ollama for embeddings and chat, ChromaDB for vector storage
# ============================================================================

import ollama
import chromadb
import os
from pypdf import PdfReader
import glob

# Model configuration
EMBED_MODEL = "nomic-embed-text"  # Embedding model for converting text to vectors
CHAT_MODEL = "llama3.2:3b"        # Chat model for generating responses

def _print_ollama_server_help():
    """Display help message if Ollama server is not running."""
    print("Ollama server is not reachable.")
    print("Start it with: ollama serve")
    print("Then verify local models with: ollama list")


def embed_with_ollama(texts: list[str], model: str) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Ollama.
    
    Args:
        texts: List of text strings to embed
        model: Name of the embedding model to use
    
    Returns:
        List of embedding vectors (each vector is a list of floats)
    """
    if not texts:
        return []

    try:
        # Try newer Ollama API (embed endpoint)
        response = ollama.embed(model=model, input=texts)
        embeddings = response.get("embeddings")
        if embeddings and isinstance(embeddings, list):
            return embeddings
    except Exception:
        pass

    # Fallback to older API (embeddings endpoint) - process one at a time
    vectors = []
    for text in texts:
        try:
            response = ollama.embeddings(model=model, prompt=text)
            embedding = response.get("embedding")
            if not embedding:
                raise RuntimeError("Missing embedding vector in Ollama response.")
            vectors.append(embedding)
        except Exception as exc:
            message = str(exc)
            # Check if it's a server connection error
            if "connect" in message.lower() or "refused" in message.lower() or "operation not permitted" in message.lower():
                _print_ollama_server_help()
            raise RuntimeError(
                f"Failed to generate embeddings with model '{model}'. "
                "Check the model name with `ollama list`."
            ) from exc
    return vectors


def load_pdf_documents(folder_path="docs"):
    """
    Load and chunk PDF documents from a specified folder.
    
    Args:
        folder_path: Directory containing PDF files (default: 'docs')
    
    Returns:
        List of text chunks (approximately 2000 characters each)
    """
    documents = []
    if not os.path.exists(folder_path):
        print(f"Create '{folder_path}' folder and add PDF files first!")
        return documents
    
    # Find all PDF files in the folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path}...")
        try:
            reader = PdfReader(pdf_path)
            # Extract text from all pages
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text.strip():  # Skip empty pages
                    full_text += text + "\n\n"
            
            # Split into ~2000 character chunks for better retrieval
            chunks = [full_text[i:i+2000] for i in range(0, len(full_text), 2000)]
            documents.extend(chunks)
            print(f"  Added {len(chunks)} chunks from {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"  Error reading {pdf_path}: {e}")
    
    return documents


# ============================================================================
# INITIALIZATION: Load PDFs and setup vector database
# ============================================================================

documents = load_pdf_documents()
print(f"Loaded {len(documents)} total chunks from PDFs")

# Initialize ChromaDB with persistent storage
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_docs")

# Index documents with embeddings if any were loaded
if documents:
    try:
        doc_ids = [f"pdf_doc_{i}" for i in range(len(documents))]
        # Generate embeddings for all documents
        document_embeddings = embed_with_ollama(documents, EMBED_MODEL)
        # Store documents, IDs, and embeddings in ChromaDB
        collection.upsert(
            documents=documents,
            ids=doc_ids,
            embeddings=document_embeddings,
        )
        print(f"✅ PDFs indexed successfully with model '{EMBED_MODEL}'!")
    except Exception as exc:
        print(f"Failed to index documents: {exc}")


def rag_query(query, llm_model="llama3.2:3b"):
    """
    Query the RAG system: retrieve relevant documents and generate an answer.
    
    Args:
        query: User's question
        llm_model: Chat model to use for generating responses
    
    Returns:
        Generated answer based on relevant document context
    """
    # Generate embedding for the query
    query_embeddings = embed_with_ollama([query], EMBED_MODEL)
    
    # Retrieve top 3 most relevant documents from ChromaDB
    results = collection.query(query_embeddings=query_embeddings, n_results=3)
    
    # Check if any relevant documents were found
    if not results.get("documents") or not results["documents"][0]:
        return "I could not find relevant context in the indexed PDFs."
    
    # Combine retrieved documents into a single context string
    context = "\n\n".join([doc for doc in results['documents'][0]])
    
    # Build the prompt with context and question
    prompt = f"""Using ONLY this context from PDF documents, answer accurately.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    try:
        # Send prompt to the chat model and get response
        response = ollama.chat(model=llm_model, messages=[{'role': 'user', 'content': prompt}])
    except Exception as exc:
        message = str(exc)
        # Check if it's a server connection error
        if "connect" in message.lower() or "refused" in message.lower() or "operation not permitted" in message.lower():
            _print_ollama_server_help()
        raise
    
    return response['message']['content']


# ============================================================================
# INTERACTIVE CHAT LOOP
# ============================================================================

print("\n🤖 PDF RAG ready! Ask questions about your docs:")
while True:
    query = input("\nQ: ")
    # Exit on 'quit', 'exit', or 'q'
    if query.lower() in ['quit', 'exit', 'q']: 
        break
    print("🤖", rag_query(query, CHAT_MODEL))