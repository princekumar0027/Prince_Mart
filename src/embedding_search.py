
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Load updated policy document
file_path = r"E:\ML Project 1\data\policies\company_policies.txt"

with open(file_path, "r", encoding="utf-8") as file:
    policy_text = file.read()

# Step 2: Split policies into chunks
chunks = re.split(
    r"(?=POLICY \d+:|GENERAL CUSTOMER SUPPORT GUIDELINES|DISCLAIMER)",
    policy_text
)

chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Generate embeddings
embeddings = model.encode(chunks)

embeddings = np.array(embeddings, dtype="float32")

# Step 5: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Step 6: Save updated index
faiss.write_index(
    index,
    r"E:\ML Project 1\data\policy_index.faiss"
)

# Step 7: Save updated policy chunks
with open(
    r"E:\ML Project 1\data\policy_chunks.txt",
    "w",
    encoding="utf-8"
) as file:
    for chunk in chunks:
        file.write(chunk.replace("\n", " ") + "\n")

print("\nUPDATED POLICY EMBEDDINGS CREATED!")
print("Total policy chunks:", len(chunks))
print("Embedding dimensions:", dimension)
print("FAISS index updated successfully!")
