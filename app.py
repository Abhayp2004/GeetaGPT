import streamlit as st
import pandas as pd
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# ---------------------------
# Load FAISS index & dataset
# ---------------------------
@st.cache_resource
def load_resources():
    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS vector store
    vector_db = FAISS.load_local("geeta_faiss_index", embeddings)

    # Load dataset for direct verse lookup
    df = pd.read_csv("geeta_dataset.csv")
    verse_dict = {
        (int(r.chapter), int(r.verse)): Document(
            page_content=r.english,
            metadata={
                "chapter": int(r.chapter),
                "verse": int(r.verse),
                "sanskrit": r.sanskrit.strip(),
                "english": r.english.strip(),
                "id": f"{r.chapter}.{r.verse}"
            }
        )
        for _, r in df.iterrows()
    }

    # Use sentence-transformers model for lightweight text generation / semantic search
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return vector_db, verse_dict, model


vector_db, verse_dict, model = load_resources()


# ---------------------------
# Helper functions
# ---------------------------
def clean_response(text: str) -> str:
    text = re.sub(r"Please.*", "", text, flags=re.I)
    text = re.sub(r"(🙏\s*){2,}", "🙏", text)
    text = re.sub(
        r"(May this wisdom guide you[^\n]*){2,}", "May this wisdom guide you. 🙏"
    )
    return text.strip()


def geeta_gpt(query: str) -> str:
    # Direct Chapter/Verse lookup
    verse_pattern = re.search(r"chapter\s*(\d+)[^\d]+verse\s*(\d+)", query.lower())
    if verse_pattern:
        chapter, verse = verse_pattern.groups()
        chapter, verse = int(chapter), int(verse)
        if (chapter, verse) in verse_dict:
            d = verse_dict[(chapter, verse)].metadata
            return f"""**Chapter {chapter}, Verse {verse}**

📜 Sanskrit:
{d['sanskrit']}

🌍 Translation:
{d['english']}

🕉️ Krishna’s Guidance:
My dear Arjuna, perform your duty with sincerity but remain unattached to results. 🙏"""

    # Semantic search for general queries
    docs_with_scores = vector_db.similarity_search_with_score(query, k=5)
    relevant_docs = [d for d, score in docs_with_scores if score >= 0.6]

    if relevant_docs:
        verses_context = "\n\n".join(
            [
                f"[Chapter {d.metadata['chapter']}, Verse {d.metadata['verse']}]\n"
                f"Sanskrit: {d.metadata['sanskrit']}\nEnglish: {d.metadata['english']}"
                for d in relevant_docs
            ]
        )
        answer = (
            f"My dear friend, here are relevant verses from the Bhagavad Gita:\n\n"
            f"{verses_context}\n\n"
            f"Reflect on them and perform your duty without attachment. May this wisdom guide you. 🙏"
        )
    else:
        answer = "My dear friend, meditate on your duty, act selflessly, and remain unattached to results. 🙏"

    return clean_response(answer)


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🕉️ GeetaGPT")
st.write(
    "Ask any question from the Bhagavad Gita or provide Chapter & Verse (e.g., 'Chapter 2, Verse 47')"
)

query = st.text_input("Your Question")
if st.button("Ask"):
    if query.strip():
        response = geeta_gpt(query)
        st.markdown(response)
