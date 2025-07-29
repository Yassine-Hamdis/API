from fastapi import FastAPI, UploadFile, File, Query
from PyPDF2 import PdfReader
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import io
from fastapi.middleware.cors import CORSMiddleware


nltk.download("punkt")
nltk.download("stopwords")

app = FastAPI()



# Add CORS middleware
origins = ["*"]  # Allow all origins, or specify allowed URLs

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # You can put ["http://localhost:8000"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.post("/pdf-summary")
async def pdf_summary(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    summary = summarize_text(text)
    return {"summary": summary}

def summarize_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    freq = {}

    for word in words:
        word = word.lower()
        if word not in stop_words and word not in string.punctuation:
            freq[word] = freq.get(word, 0) + 1

    sentences = sent_tokenize(text)
    sentence_scores = {}

    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq[word]

    # Top 3 sentences
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    return " ".join(summary_sentences)

@app.post("/weather-stats")
async def weather_stats(file: UploadFile = File(...), column: str = Query(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in file."}

    stats = df[column].describe().to_dict()
    return {"column": column, "stats": stats}
