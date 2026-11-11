
import re

# Load the policy document
file_path = r"E:\ML Project 1\data\policies\company_policies.txt"

with open(file_path, "r", encoding="utf-8") as file:
    policy_text = file.read()

# Split document into individual policy sections
chunks = re.split(
    r"(?=POLICY \d+:|GENERAL CUSTOMER SUPPORT GUIDELINES|DISCLAIMER:)",
    policy_text
)

# Remove empty sections and separator lines
chunks = [
    re.sub(r"\n?={5,}\n?", "\n", chunk).strip()
    for chunk in chunks
    if chunk.strip()
]

# Display the chunks
print("TOTAL POLICY CHUNKS:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)
