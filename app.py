import os
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Prince_Mart | AI Support", page_icon="🛍️", layout="wide")

# 2. CUSTOM CSS
st.markdown("""
<style>
.stApp {background-color: #f4f7fc;}
h1, h2, h3 {color: #102a43;}
.main-title {color: #123c69; font-size: 35px; font-weight: 800;}
.subtitle {color: #627d98; font-size: 15px;}
.stButton > button {
    background-color: #164e8a; color: white; border-radius: 8px;
    border: none; padding: 9px 18px;
}
.stButton > button:hover {background-color: #0d315c; color: white;}
div[data-testid="stChatMessage"] {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# 3. LOAD API KEY
load_dotenv(r"E:\ML Project 1\.env")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Groq API key not found. Check your .env file.")
    st.stop()
client = Groq(api_key=api_key)

# 4. LOAD EMBEDDING MODEL AND FAISS INDEX
@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.read_index(r"E:\ML Project 1\data\policy_index.faiss")
    with open(r"E:\ML Project 1\data\policy_chunks.txt", "r", encoding="utf-8") as file:
        policy_chunks = file.readlines()
    return embedding_model, faiss_index, policy_chunks

try:
    model, index, chunks = load_resources()
except Exception as e:
    st.error(f"Could not load RAG resources: {e}")
    st.stop()

# 5. RETRIEVE RELEVANT POLICIES
def retrieve_policies(query):
    query_embedding = np.array(model.encode([query]), dtype="float32")
    distances, indices = index.search(query_embedding, 3)
    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved.append(chunks[idx].strip())
    return retrieved

# 6. GENERATE RESPONSE USING GROQ
def generate_response(query, policies):
    context = "\n\n".join(policies)
    prompt = f"""
You are Prince_Mart's AI customer support assistant.
Answer using ONLY information explicitly stated in the retrieved policies.

RETRIEVED POLICY INFORMATION:
{context}

CUSTOMER QUESTION:
{query}

STRICT INSTRUCTIONS:
1. Be polite, professional and customer-friendly.
2. Use only facts and procedures explicitly stated in the retrieved policies.
3. Never invent email addresses, website links, contact forms, phone numbers or contact channels.
4. Never invent return procedures, refund methods, fees, delivery dates or approval decisions.
5. Do not claim a request has been submitted, forwarded, approved or processed.
6. Do not claim to access live order or payment information.
7. If information is missing, clearly state it is not specified in the available policies.
8. Ask for relevant customer information only when the policy requires it.
9. Recommend human support when verification or approval is required.
10. Do not repeat irrelevant policy information.
11. Never request passwords, PINs, CVVs or OTPs.

A customer requesting an action does not mean it has been approved or completed.
If the policies do not contain sufficient information, do not guess.
Return only the customer-facing response.
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# 7. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# 8. SIDEBAR
with st.sidebar:
    st.markdown("## 🛍️ Prince_Mart")
    st.caption("AI Customer Support System")
    st.divider()
    st.markdown("### Suggested Questions")

    suggestions = [
        "How can I request a refund?",
        "My order has been shipped. Can I cancel it?",
        "My payment was deducted but order is not confirmed.",
        "I received a damaged product. What should I do?",
        "How can I request a product replacement?"
    ]

    for i, question in enumerate(suggestions):
        if st.button(question, use_container_width=True, key=f"suggestion_{i}"):
            st.session_state.pending_query = question
            st.rerun()

    st.divider()
    st.markdown("### Conversation")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = ""
        st.session_state.feedback = {}
        st.rerun()
    st.caption("Powered by RAG, FAISS and Groq AI")

# 9. MAIN HEADER
st.markdown('<div class="main-title">Prince_Mart</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-Powered Customer Support Assistant</div>',
    unsafe_allow_html=True
)
st.divider()
st.markdown("### Welcome to Prince_Mart 👋")
st.write(
    "I'm your AI customer support assistant. "
    "Ask me about refunds, returns, cancellations, payments, delivery and product issues."
)

# 10. QUICK SUPPORT CATEGORIES
st.markdown("### Quick Support Categories")
col1, col2, col3, col4, col5 = st.columns(5)

if col1.button("💰 Refund", use_container_width=True):
    st.session_state.pending_query = "How can I request a refund?"
    st.rerun()
if col2.button("📦 Return", use_container_width=True):
    st.session_state.pending_query = "What is your return policy?"
    st.rerun()
if col3.button("❌ Cancellation", use_container_width=True):
    st.session_state.pending_query = "Can I cancel my order?"
    st.rerun()
if col4.button("💳 Payment", use_container_width=True):
    st.session_state.pending_query = "What should I do if payment was deducted but my order is not confirmed?"
    st.rerun()
if col5.button("🚚 Delivery", use_container_width=True):
    st.session_state.pending_query = "What is the expected delivery time?"
    st.rerun()

# 11. DISPLAY CHAT HISTORY AND FEEDBACK
for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            with st.expander("View Retrieved Policies"):
                for i, policy in enumerate(message.get("policies", []), start=1):
                    st.markdown(f"**Retrieved section {i}**")
                    st.write(policy)
                    
            st.write("Was this answer helpful?")
            feedback_key = str(message_index)
            selected_feedback = st.session_state.feedback.get(feedback_key)
            feedback_col1, feedback_col2 = st.columns(2)

            with feedback_col1:
                if st.button("👍 Helpful", key=f"like_{message_index}",
                             disabled=selected_feedback is not None):
                    st.session_state.feedback[feedback_key] = "Helpful"
                    st.rerun()
            with feedback_col2:
                if st.button("👎 Not Helpful", key=f"dislike_{message_index}",
                             disabled=selected_feedback is not None):
                    st.session_state.feedback[feedback_key] = "Not Helpful"
                    st.rerun()

            if selected_feedback:
                st.caption(f"Thank you! Your feedback: {selected_feedback}")

# 12. CUSTOMER QUERY INPUT
pending_query = st.session_state.pending_query
st.session_state.pending_query = ""
query = st.chat_input("Ask Prince_Mart anything about your order...")
if pending_query:
    query = pending_query

# 13. PROCESS CUSTOMER QUERY
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating response..."):
            try:
                policies = retrieve_policies(query)
                answer = generate_response(query, policies)
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "policies": policies
                })

                new_message_index = len(st.session_state.messages) - 1
                feedback_key = str(new_message_index)
                st.write("Was this answer helpful?")
                feedback_col1, feedback_col2 = st.columns(2)

                with feedback_col1:
                    if st.button("👍 Helpful", key=f"like_{new_message_index}"):
                        st.session_state.feedback[feedback_key] = "Helpful"
                        st.rerun()
                with feedback_col2:
                    if st.button("👎 Not Helpful", key=f"dislike_{new_message_index}"):
                        st.session_state.feedback[feedback_key] = "Not Helpful"
                        st.rerun()

                with st.expander("View Retrieved Policies"):
                    for i, policy in enumerate(policies, start=1):
                        st.markdown(f"**Retrieved section {i}**")
                        st.write(policy)

            except Exception as e:
                error_message = (
                    "Sorry, I could not process your request. Please try again."
                )
                st.error(error_message)
                st.caption(str(e))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "policies": []
                })


# ==========================================
# 14. CUSTOMER SUPPORT ANALYTICS
# ==========================================

st.divider()

st.markdown("""
<style>
.analytics-heading {
    color: #718096;
    font-size: 15px;
    font-weight: 400;
    margin-bottom: 2px;
}
.analytics-caption {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 400;
}
div[data-testid="stMetric"] {
    background-color: #f7f9fc;
    border: 1px solid #edf1f5;
    border-radius: 8px;
    padding: 10px 14px;
}
div[data-testid="stMetricLabel"] {
    color: #8492a6;
    font-size: 12px;
    font-weight: 400;
}
div[data-testid="stMetricValue"] {
    color: #64748b;
    font-size: 22px;
    font-weight: 400;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="analytics-heading">Customer Support Analytics</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="analytics-caption">Current session summary</div>',
    unsafe_allow_html=True
)

total_queries = sum(
    1 for message in st.session_state.messages
    if message["role"] == "user"
)

helpful_count = sum(
    1 for value in st.session_state.feedback.values()
    if value == "Helpful"
)

not_helpful_count = sum(
    1 for value in st.session_state.feedback.values()
    if value == "Not Helpful"
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Total Queries", total_queries)

with metric2:
    st.metric("Helpful", helpful_count)

with metric3:
    st.metric("Not Helpful", not_helpful_count)

# 15. FOOTER
st.divider()
st.caption(
    "Prince_Mart is a hypothetical company created for academic demonstration. "
    "This assistant provides policy-based guidance and does not verify live "
    "orders or process transactions."
)
