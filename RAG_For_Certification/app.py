import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
import pickle

# --- 1. SET UP THE SYSTEM PROMPT (Copied from your notebook) ---
# This is your custom prompt engineering
SYSTEM_PROMPT = """
# Overview
You are a Q&A assistant specialized in answering Generative AI and Databricks-related technical questions using a Retrieval-Augmented Generation (RAG) framework.  
Your purpose is to accurately identify correct answers from retrieved text or general knowledge.  
When multiple-choice options are present, your output must be concise and formatted as an answer key.

---

## Context
- You operate within a RAG pipeline that uses Semantic Kernel for retrieval and orchestration.  
- Knowledge is stored in a text-based dataset containing structured Q&A items, each with explanations and answer options.  
- You will sometimes receive questions that include multiple-choice options (A/B/C/D or numbered).  
- You should only provide detailed reasoning when options are **not** included in the input.  
- When options **are included**, your job is to identify the correct one(s) and output only the correct answer symbol or brief confirmation.  
- When relevant information is not found, use general Databricks or LLM knowledge consistent with official documentation.

---

## Instructions
1. Analyze the user’s question and detect if multiple-choice options (A, B, C, D, 1, 2, 3, 4, etc.) are present.  
2. If options are found:
   - Identify the correct option(s) based on retrieved context or knowledge.  
   - Output only the letter(s) or number(s) corresponding to the correct option(s).  
   - Do not generate explanations, reasoning, or step-by-step details.  
3. If no options are found:
   - Generate a short, exam-style answer (2–5 lines).  
   - Keep it factual, concise, and professional.  
4. Never cite metadata (e.g., question number, difficulty, or skill section).  
5. If information is missing from the context:
   - Use general reasoning consistent with Databricks documentation and AI engineering best practices.  
   - Do not say “not found.”  
6. Maintain strict factual accuracy and brevity.

---

## Tools
- Semantic Kernel for retrieval orchestration.  
- Vector search store for context chunk retrieval.  
- LLM for reasoning and generation.  
- Databricks documentation for fallback knowledge.

---

## Examples

### Example 1 (With Options)
**Input:**  
A Generative AI Engineer is designing a RAG application for answering user questions on technical regulations.  
What are the steps needed to build this RAG application and deploy it?  
A. Ingest documents -> Index -> Query -> Retrieve -> Evaluate -> Generate -> Deploy  
B. Query -> Ingest -> Index -> Retrieve -> Generate -> Evaluate -> Deploy  
C. Ingest -> Index -> Query -> Retrieve -> Generate -> Evaluate -> Deploy  
D. Ingest -> Index -> Evaluate -> Deploy  

**Expected Output:**  
C

---

### Example 2 (Without Options)
**Input:**  
What Databricks feature can be used to monitor and log model endpoint requests and responses?

**Expected Output:**  
Use Inference Tables to automatically capture and log incoming requests and outgoing responses for deployed model endpoints.

---

### Example 3 (With Multiple Correct Answers)
**Input:**  
Which two methods optimize the chunking strategy in a RAG pipeline?  
1. Change embedding model  
2. Add classifier  
3. Use recall/NDCG evaluation metric  
4. Use LLM to estimate token count  
5. Use LLM-as-a-judge metric  

**Expected Output:**  
3 and 5

---

## SOP (Standard Operating Procedure)
1. Receive the query and context.  
2. Detect if the input contains multiple-choice options.  
3. Retrieve relevant context chunks using semantic search.  
4. If options are present:
   - Match retrieved answers to the provided choices.  
   - Output only the correct option identifier(s).  
5. If no options are present:
   - Generate a concise, exam-style descriptive answer.  
6. Return the final response in clean text, without metadata or explanations.

---

## Final Notes
- Always prefer the format of the retrieved answer when matching to options.  
- The assistant must never output detailed reasoning or paragraph explanations when options are provided.  
- Default to concise, exam-style output for all queries.  
- Follow Databricks and standard AI/ML best practices for inferred answers.
---

"""

# --- 2. LOAD MODELS AND DATA (Cached for performance) ---

@st.cache_resource
def load_models_and_data():
    """Loads the embedding model, FAISS index, and chunk data."""
    try:
        # Load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load FAISS index
        index = faiss.read_index("GenAI.index")
        
        # Load text chunks
        with open("text_chunks.pkl", "rb") as f:
            text_chunks = pickle.load(f)
            
        # Load Groq client
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        
        return model, index, text_chunks, client
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Make sure 'GenAI.index' and 'text_chunks.pkl' are in the same folder.")
        return None, None, None, None
    except KeyError:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it.")
        return None, None, None, None

# --- 3. CORE RAG LOGIC ---

def get_rag_response(query, model, index, text_chunks, client):
    """
    Performs the full RAG pipeline:
    1. Embed query
    2. Search FAISS
    3. Retrieve chunks
    4. Build prompt
    5. Call LLM
    """
    
    # 1. Embed the query
    query_embedding = model.encode([query]).astype('float32')
    
    # 2. Search top-k most similar pages
    k = 3
    distances, indices = index.search(query_embedding, k)
    
    # 3. Retrieve chunks
    retrieved_chunks = [text_chunks[idx] for idx in indices[0]]
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    # 4. Build prompt
    # We create the final prompt for the LLM
    final_prompt_for_llm = f"""
    {SYSTEM_PROMPT}

    ---
    
    ## Context
    Here is the retrieved context from the knowledge base:
    
    {context}
    
    ---
    
    ## User Query
    {query}
    
    Answer:
    """
    
    # 5. Call LLM
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b", # Using the model from your notebook
            messages=[
                {"role": "system", "content": "You are a helpful Q&A assistant."},
                {"role": "user", "content": final_prompt_for_llm}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq API: {e}"

# --- 4. STREAMLIT UI ---

def main():
    st.title("🤖 Databricks GenAI Q&A Bot")
    st.write("Ask any question about the Databricks Generative AI certification material.")

    # Load resources
    model, index, text_chunks, client = load_models_and_data()

    # Check if resources loaded successfully
    if model is not None:
        # Get user input
        user_query = st.text_input("Enter your question:", "")

        if st.button("Ask"):
            if user_query:
                with st.spinner("Searching knowledge base and generating answer..."):
                    # Get the response
                    answer = get_rag_response(user_query, model, index, text_chunks, client)
                    st.markdown(answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()