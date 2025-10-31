# Hackathon An Intelligent Enterprise Assistant for public sector

### PROBLEM TITLE: "Intelligent Enterprise Assistant: Enhancing Organizational Efficiency through AI-driven Chatbot Integration"

### Problem ID: SIH1706

### Problem Statement Description:

Develop a chatbot using deep learning and natural language processing techniques to accurately understand and respond to queries from employees of a large public sector organization.

The chatbot should be capable of handling diverse questions related to HR policies, IT support, company events, and other organizational matters. (Hackathon students/teams to use publicly available sample information for HR Policy, IT Support, etc. available on internet.) Develop document processing capabilities for the chatbot to analyse and extract information from documents uploaded by employees.

This includes summarizing a document or extracting text (keyword information) from documents relevant to organizational needs. (Hackathon students/teams can use any 8 to 10 page document for demonstration). Ensure the chatbot architecture is scalable to handle minimum 5 users parallelly. This includes optimizing response time (Response Time should not exceed 5 seconds for any query unless there is a technical issue like connectivity, etc.) Enable 2FA (2 Factor Authentication – email id type) in the chatbot for enhancing the security level of the chatbot.

### Code:
```python
!pip install transformers sentence-transformers pdfplumber torch --quiet

from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pdfplumber, io, random

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

user_sessions = {}

def request_otp(email):
    otp = str(random.randint(100000, 999999))
    user_sessions[email] = otp
    print(f"\n[Demo OTP for {email} → {otp}]")
    return otp

def verify_otp(email, otp):
    return user_sessions.get(email) == otp

def summarize_document(path):
    text = ""
    if path.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    text = text.strip().replace("\n", " ")
    if len(text) < 200:
        return "Document too short for summarization."
    summary = summarizer(text[:3000], max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    return summary

def chat(email):
    if email not in user_sessions:
        print("⚠️ Please verify OTP first.")
        return
    print("\nChatbot ready. Type 'quit' to exit.\n")
    context = (
        "HR policies include leave, salary, and promotion rules. "
        "IT support helps with login issues, system errors, and email setup. "
        "Company events involve training, awards, and wellness activities."
    )
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Chatbot: Goodbye 👋")
            break
        response = qa_model(question=query, context=context)
        print(f"Chatbot: {response['answer']}\n")

email = input("Enter your email: ")
otp = request_otp(email)
user_otp = input("Enter OTP shown above: ")

if verify_otp(email, user_otp):
    print("✅ Login successful!")
    print("\n--- Document Summarization ---")
    sample_text = "This HR policy document describes employee leave rules, benefits, and working conditions."
    with open("sample.txt", "w") as f:
        f.write(sample_text)
    print("Summary:", summarize_document("sample.txt"))
    chat(email)
else:
    print("❌ Invalid OTP")

```
### Output:
<img width="522" height="527" alt="image" src="https://github.com/user-attachments/assets/5d94818a-adc3-44fb-a531-d2588269f4ee" />


