import streamlit as st
import pandas as pd
import re
import os
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ---------------------------
# 1. Build or Load FAISS Index
# ---------------------------
def build_faiss_index(csv_path="geeta_dataset.csv", index_path="geeta_faiss_index"):
    df = pd.read_csv(csv_path)
    df['chapter'] = df['chapter'].astype(int)
    df['verse'] = df['verse'].astype(int)
    df['sanskrit'] = df['sanskrit'].astype(str).str.strip()
    df['english'] = df['english'].astype(str).str.strip()

    docs = []
    verse_dict = {}
    for _, row in df.iterrows():
        doc = Document(
            page_content=row["english"],
            metadata={
                "chapter": row["chapter"],
                "verse": row["verse"],
                "sanskrit": row["sanskrit"],
                "english": row["english"],
                "id": f"{row['chapter']}.{row['verse']}"
            }
        )
        docs.append(doc)
        verse_dict[(row["chapter"], row["verse"])] = doc

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(index_path)
    return vector_db, verse_dict

def load_faiss_index(csv_path="geeta_dataset.csv", index_path="geeta_faiss_index"):
    try:
        vector_db = FAISS.load_local(
            index_path,
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        df = pd.read_csv(csv_path)
        verse_dict = {(int(r.chapter), int(r.verse)): Document(
            page_content=r.english,
            metadata={
                "chapter": int(r.chapter),
                "verse": int(r.verse),
                "sanskrit": r.sanskrit.strip(),
                "english": r.english.strip(),
                "id": f"{r.chapter}.{r.verse}"
            }
        ) for _, r in df.iterrows()}
        return vector_db, verse_dict
    except:
        return build_faiss_index(csv_path, index_path)

# ---------------------------
# 2. Load Hugging Face Inference Client
# ---------------------------
@st.cache_resource
def load_generator(model_name="meta-llama/Llama-3-2-1B"):
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        st.error("Missing Hugging Face token. Add it in Streamlit → Secrets.")
        st.stop()
    return InferenceClient(model=model_name, token=token)

# ---------------------------
# 3. Cleanup Response
# ---------------------------
def clean_response(text: str) -> str:
    text = re.sub(r"Please.*", "", text, flags=re.I)
    text = re.sub(r"(🙏\s*){2,}", "🙏", text)
    return text.strip()

# ---------------------------
# 4. GeetaGPT Function
# ---------------------------
def geeta_gpt(query, vector_db, verse_dict, client, top_k=4, similarity_threshold=0.7):
    # Direct chapter/verse lookup
    verse_pattern = re.search(r"chapter\s*(\d+)[^\d]+verse\s*(\d+)", query.lower())
    if verse_pattern:
        chapter, verse = map(int, verse_pattern.groups())
        if (chapter, verse) in verse_dict:
            d = verse_dict[(chapter, verse)].metadata
            return f"""**Chapter {chapter}, Verse {verse}**

📜 *Sanskrit*:
{d.get('sanskrit', 'Sanskrit text unavailable')}

🌍 *Translation*:
{d.get('english', 'English translation unavailable')}

🕉️ *Krishna’s Guidance*:
My dear Arjuna, reflect on this teaching. Perform your duty with sincerity, but remain unattached to the fruits. May this wisdom guide you. 🙏"""

    # Semantic search
    docs_with_scores = vector_db.similarity_search_with_score(query, k=top_k)
    relevant_docs = [d for d, score in docs_with_scores if score >= similarity_threshold]

    verses_context = "\n\n".join([
        f"[Chapter {d.metadata['chapter']}, Verse {d.metadata['verse']}]\n"
        f"Sanskrit: {d.metadata['sanskrit']}\nEnglish: {d.metadata['english']}"
        for d in relevant_docs
    ]) if relevant_docs else "No directly relevant verses found."

    prompt = f"""
You are GeetaGPT, the eternal voice of Shree Krishna from the Bhagavad Gita.
Context:
{verses_context}

User Question: {query}

Answer as Shree Krishna (3-5 sentences). End with: "May this wisdom guide you. 🙏"
"""

    # Hugging Face InferenceClient usage
    response = client.text_generation(
    prompt,  # positional argument
    parameters={"max_new_tokens": 250}
)
generated_text = response[0]["generated_text"]


    return clean_response(generated_text)

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("🕉️ GeetaGPT")
st.write("Ask any question about the Bhagavad Gita or for life guidance.")

vector_db, verse_dict = load_faiss_index()
client = load_generator()

query = st.text_input("Enter your question:")
if query:
    with st.spinner("🕉️ Consulting Krishna..."):
        answer = geeta_gpt(query, vector_db, verse_dict, client)
        st.markdown(answer)
