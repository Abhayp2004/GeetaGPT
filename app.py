import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load dataset and embeddings
# ---------------------------
@st.cache_resource
def load_resources():
    # Load dataset
    df = pd.read_csv("geeta_dataset.csv")
    df['chapter'] = df['chapter'].astype(int)
    df['verse'] = df['verse'].astype(int)
    df['sanskrit'] = df['sanskrit'].astype(str).str.strip()
    df['english'] = df['english'].astype(str).str.strip()

    # Create verse dictionary for direct lookup
    verse_dict = {
        (row.chapter, row.verse): row
        for _, row in df.iterrows()
    }

    # Initialize embeddings model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Compute embeddings for all verses
    embeddings = model.encode(df['english'].tolist(), convert_to_numpy=True)

    return df, verse_dict, model, embeddings

df, verse_dict, model, verse_embeddings = load_resources()


# ---------------------------
# Helper functions
# ---------------------------
def clean_response(text: str) -> str:
    text = re.sub(r"Please.*", "", text, flags=re.I)
    text = re.sub(r"(🙏\s*){2,}", "🙏", text)
    text = re.sub(r"(May this wisdom guide you[^\n]*){2,}", "May this wisdom guide you. 🙏")
    return text.strip()


def geeta_gpt(query: str) -> str:
    # Direct Chapter/Verse lookup
    verse_pattern = re.search(r"chapter\s*(\d+)[^\d]+verse\s*(\d+)", query.lower())
    if verse_pattern:
        chapter, verse = map(int, verse_pattern.groups())
        if (chapter, verse) in verse_dict:
            row = verse_dict[(chapter, verse)]
            return f"""**Chapter {chapter}, Verse {verse}**

📜 Sanskrit:
{row.sanskrit}

🌍 Translation:
{row.english}

🕉️ Krishna’s Guidance:
My dear Arjuna, perform your duty with sincerity but remain unattached to results. 🙏"""

    # Semantic search
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, verse_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:5]  # top 5

    relevant_verses = [
        df.iloc[i] for i in top_indices if similarities[i] >= 0.6
    ]

    if relevant_verses:
        verses_context = "\n\n".join(
            [
                f"[Chapter {row.chapter}, Verse {row.verse}]\n"
                f"Sanskrit: {row.sanskrit}\nEnglish: {row.english}"
                for row in relevant_verses
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
st.write("Ask any question from the Bhagavad Gita or provide Chapter & Verse (e.g., 'Chapter 2, Verse 47')")

query = st.text_input("Your Question")
if st.button("Ask"):
    if query.strip():
        response = geeta_gpt(query)
        st.markdown(response)
