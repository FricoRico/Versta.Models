import os

os.environ["UNSLOTH_COMPILE_LOCATION"] = "cache/unsloth/"

# Import unsloth first to fix import order issues with transformers
import unsloth
