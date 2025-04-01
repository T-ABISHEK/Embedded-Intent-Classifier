🤖 Embedded Intent Classifier

A production-grade FastAPI application that uses **sentence embeddings** to classify user prompts into 4 categories:

1. Search-based (e.g. weather, time, news)
2. Math-based queries
3. Casual conversation
4. Image generation prompts

Built using **Python**, **BGE SentenceTransformer**, **Scikit-learn**, and **FastAPI**.

---

🚀 Features

- BGE Embeddings via `sentence-transformers`
- Random Forest Classifier
- REST API using FastAPI
- JSON response with label predictions

---


---

📦 Installation

```bash
git clone https://github.com/T-ABISHEK/Embedded-Intent-Classifier.git
cd Embedded-Intent-Classifier
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

---

📁 Prepare Dataset

Create a `data/labeled_prompts.csv` like:

```csv
prompt,label
"What's the weather like today?",1
"What is 7 multiplied by 8?",2
"Tell me a joke.",3
"Generate an image of a cyberpunk city",4
```

---

🧪 Train the Model

```bash
python main.py
```
This saves the model to `model/classifier.pkl`

---

🚀 Run the API

```bash
uvicorn api:app --reload
```
Then open: [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI

---

📬 Sample Prediction

```json
POST /predict
{
  "text": "What is the current time in London?"
}

Response:
{
  "label": 1
}
```
