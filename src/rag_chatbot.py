
import os
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Load API key
load_dotenv(r"E:\ML Project 1\.env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

# 2. Connect to Groq
client = Groq(api_key=api_key)

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Load updated FAISS index
index = faiss.read_index(
    r"E:\ML Project 1\data\policy_index.faiss"
)

# 5. Load updated policy chunks
with open(
    r"E:\ML Project 1\data\policy_chunks.txt",
    "r",
    encoding="utf-8"
) as file:
    chunks = file.readlines()


# 6. Retrieve relevant policies
def retrieve_policies(query):

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(query_embedding, 3)

    retrieved_policies = []

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved_policies.append(chunks[idx].strip())

    return retrieved_policies


# 7. Generate response using Groq
def generate_response(query):

    policies = retrieve_policies(query)

    context = "\n\n".join(policies)

    prompt = f"""
You are Prince_Mart's AI customer support assistant.

Answer the customer using ONLY the retrieved policy information.

RETRIEVED POLICIES:
{context}

CUSTOMER QUESTION:
{query}

STRICT INSTRUCTIONS:

1. Give polite, professional and clear responses.
2. Never invent company rules, procedures, fees or timelines.
3. Do not assume that an order, payment, refund or cancellation
   has been verified or approved.
4. Do not claim to access live order or delivery information.
5. Ask for relevant missing information when necessary.
6. If the retrieved policies do not contain the answer,
   clearly explain that the information is unavailable.
7. Recommend human support when verification or approval is required.
8. Do not provide unsupported instructions.
9. Keep the response concise and customer-friendly.

Write only the response to the customer.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


# 8. Run chatbot
print("\n======================================")
print(" PRINCE_MART AI CUSTOMER SUPPORT")
print("======================================")

while True:

    query = input("\nEnter customer query (type exit to stop): ")

    if query.strip().lower() == "exit":
        print("Thank you for contacting Prince_Mart!")
        break

    if not query.strip():
        print("Please enter a valid question.")
        continue

    try:
        answer = generate_response(query)

        print("\nPRINCE_MART SUPPORT RESPONSE:")
        print(answer)

    except Exception as e:
        print("\nAn error occurred:", e)
