import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env FIRST — before any HuggingFace-adjacent imports
load_dotenv(Path(__file__).parent.parent.parent / ".env")

os.environ["UNSLOTH_COMPILE_LOCATION"] = "cache/unsloth/"
