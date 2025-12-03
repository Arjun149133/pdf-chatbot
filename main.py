from fastapi import FastAPI, UploadFile
from utils import extract_pdf_text, chunk_text, create_faiss_index
from sentence_transformers import SentenceTransformer

from openai import OpenAI
#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

app = FastAPI()
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

chunks = []
index = None
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile):
    global chunks, index

    file_path = f"./{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_pdf_text(file_path)

    chunks = chunk_text(text)

    index, _ = create_faiss_index(chunks)

    return {
        "message": "PDF processed successfully",
        "chunks": len(chunks)
    }

@app.post("/ask")
async def ask_question(query: str):
    global chunks, index

    if index is None:
        return {"error": "Upload a pdf first"}
    
    q_embed = model.encode([query]).astype("float32")

    D, I = index.search(q_embed, 3)

    relevant = [chunks[i] for i in I[0]]

    return {
        "query": query,
        "context_used": relevant
    }

@app.post("/ask-ai")
async def ask_ai(query: str):
    global chunks, index

    if index is None:
        return {"error": "Upload a PDF first"}
    
    q_embed = model.encode([query]).astype("float32")
    D, I = index.search(q_embed, 3)

    relevant = "\n\n".join([chunks[i] for i in I[0]])

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {relevant}

    Question:
    {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content
    }