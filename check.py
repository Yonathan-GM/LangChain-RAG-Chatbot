import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

print("Available embedding models:")
print("-" * 50)

for model in genai.list_models():
    if "embed" in model.name:
        print(f"✓ {model.name}")