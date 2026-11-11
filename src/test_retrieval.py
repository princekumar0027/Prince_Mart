
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load saved FAISS index
index = faiss.read_index(
    r"E:\ML Project 1\data\policy_index.faiss"
)

# Load policy chunks
with open(
    r"E:\ML Project 1\data\policy_chunks.txt",
    "r",
    encoding="utf-8"
) as file:
    chunks = file.readlines()

# Customer query
query = "I received a damaged product. Can I get a refund?"

# Convert query into embedding
query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding,
    dtype="float32"
)

# Retrieve the two most relevant policies
distances, indices = index.search(query_embedding, 2)

print("\nCUSTOMER QUERY:")
print(query)

print("\nRETRIEVED POLICIES:")

for i, idx in enumerate(indices[0]):
    print(f"\nResult {i + 1}")
    print("Distance:", distances[0][i])
    print("Policy:", chunks[idx])
