import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION = "code"

CHUNK_LINES = 40      # lines per chunk
CHUNK_OVERLAP = 10    # lines shared between neighbouring chunks
TOP_K = 5             # chunks retrieved per search (naive and agentic)
MAX_RETRIES = 3       # max retrieve -> grade -> rewrite loops
