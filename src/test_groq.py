
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv(r"E:\ML Project 1\.env")

api_key = os.getenv("GROQ_API_KEY")

# Connect to Groq
client = Groq(api_key=api_key)

# Send a test request
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain customer support in two simple sentences."
        }
    ],
    temperature=0.3
)

# Display AI response
print("\n--- GROQ API RESPONSE ---")
print(response.choices[0].message.content)
