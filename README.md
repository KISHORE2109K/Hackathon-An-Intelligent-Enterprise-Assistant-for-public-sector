# Hackathon An Intelligent Enterprise Assistant for public sector

### PROBLEM TITLE: "Intelligent Enterprise Assistant: Enhancing Organizational Efficiency through AI-driven Chatbot Integration"

### Problem ID: SIH1706

### Problem Statement Description:

Develop a chatbot using deep learning and natural language processing techniques to accurately understand and respond to queries from employees of a large public sector organization.

The chatbot should be capable of handling diverse questions related to HR policies, IT support, company events, and other organizational matters. (Hackathon students/teams to use publicly available sample information for HR Policy, IT Support, etc. available on internet.) Develop document processing capabilities for the chatbot to analyse and extract information from documents uploaded by employees.

This includes summarizing a document or extracting text (keyword information) from documents relevant to organizational needs. (Hackathon students/teams can use any 8 to 10 page document for demonstration). Ensure the chatbot architecture is scalable to handle minimum 5 users parallelly. This includes optimizing response time (Response Time should not exceed 5 seconds for any query unless there is a technical issue like connectivity, etc.) Enable 2FA (2 Factor Authentication – email id type) in the chatbot for enhancing the security level of the chatbot.

### Code:
```python
from fastapi import FastAPI, UploadFile, File, Form
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pdfplumber, io, time, random
from pyngrok import ngrok
import uvicorn
import threading

app = FastAPI(title="Intelligent Enterprise Assistant")

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_model = pipeline("text2text-generation", model="t5-small")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

user_sessions = {}

@app.get("/")
def home():
    return {"message": "Welcome to Intelligent Enterprise Assistant 👋"}

@app.post("/request-otp")
def request_otp(email: str = Form(...)):
    otp = str(random.randint(100000, 999999))
    user_sessions[email] = otp
    return {"message": "OTP generated (for demo)", "otp": otp}

@app.post("/verify-otp")
def verify_otp(email: str = Form(...), otp: str = Form(...)):
    if user_sessions.get(email) == otp:
        return {"message": f"Login successful for {email}"}
    return {"error": "Invalid OTP"}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    text = ""
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(await file.read())) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    else:
        text = (await file.read()).decode(errors="ignore")
    text = text.strip().replace("\n", " ")
    if len(text) < 200:
        return {"summary": "Document too short for summarization."}
    summary = summarizer(text[:3000], max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    return {"summary": summary}

@app.post("/chat")
def chat(query: str = Form(...), email: str = Form(...)):
    start = time.time()
    if email not in user_sessions:
        return {"error": "Please verify OTP first."}
    context = (
        "HR policies include leave, salary, and promotion rules. "
        "IT support helps with login issues, system errors, and email setup. "
        "Company events involve training, awards, and wellness activities."
    )
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = qa_model(prompt, max_length=100)[0]["generated_text"]
    elapsed = round(time.time() - start, 2)
    return {"answer": response, "response_time": f"{elapsed}s"}

@app.get("/health")
def health():
    return {"status": "ok", "active_users": len(user_sessions)}

def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_app)
thread.start()

ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN_HERE")
public_url = ngrok.connect(8000)
print("🚀 Public URL:", public_url)
```

