import streamlit as st
import pandas as pd
import re
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import pipeline
import os
from huggingface_hub import login

login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])


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
# 2. Load LLM Generator
# ---------------------------
def load_generator(model_name="meta-llama/Llama-3.2-1B"):
    hf_token = os.environ.get("HF_TOKEN")  # add your Hugging Face token in Secrets
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=0,
        token=hf_token
    )
    return generator

# ---------------------------
# 3. Cleanup Response
# ---------------------------
def clean_response(text: str) -> str:
    text = re.sub(r"Please.*", "", text, flags=re.I)
    text = re.sub(r"(🙏\s*){2,}", "🙏", text)
    text = re.sub(r"(May this wisdom guide you[^\n]*){2,}", "May this wisdom guide you. 🙏", text)
    return text.strip()

# ---------------------------
# 4. GeetaGPT Function
# ---------------------------
def geeta_gpt(query, vector_db, verse_dict, generator, top_k=4, similarity_threshold=0.7):
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
    if not relevant_docs:
        return "This teaching is not directly in the verses provided, but I will guide you in the spirit of the Gita. May this wisdom guide you. 🙏"

    verses_context = "\n\n".join([
        f"[Chapter {d.metadata['chapter']}, Verse {d.metadata['verse']}]\n"
        f"Sanskrit: {d.metadata['sanskrit']}\nEnglish: {d.metadata['english']}"
        for d in relevant_docs
    ])

    system_prompt = (
        "You are GeetaGPT, the eternal voice of Shree Krishna from the Bhagavad Gita. "
        "Answer questions with clarity, compassion, and authority.\n"
        "Rules:\n"
        "1. Base answers on provided verses.\n"
        "2. Quote at least one relevant verse.\n"
        "3. Keep it concise (3-5 sentences).\n"
        "4. Maintain the tone of Krishna guiding Arjuna.\n"
        "5. End with: 'May this wisdom guide you. 🙏'\n"
    )

    prompt = f"""{system_prompt}

Context (Bhagavad Gita verses):
{verses_context}

User's Question: {query}

Answer as Shree Krishna:
"""

    raw_out = generator(prompt, max_new_tokens=250, temperature=0.3, do_sample=True)[0]["generated_text"]
    answer = raw_out.split("Answer as Shree Krishna:")[-1].strip()

    # Remove repeated sentences
    seen = set()
    result = []
    for sentence in re.split(r'(?<=[.!?])\s+', answer):
        s = sentence.strip()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return ' '.join(result)

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("🕉️ GeetaGPT")
st.write("Ask any question about the Bhagavad Gita or for life guidance.")

vector_db, verse_dict = load_faiss_index()
generator = load_generator()

query = st.text_input("Enter your question:")
if query:
    with st.spinner("🕉️ Consulting Krishna..."):
        answer = geeta_gpt(query, vector_db, verse_dict, generator)
        st.markdown(answer)

