import pickle
from embed_classifier.embedder import Embedder

class Predictor:
    def __init__(self, model_path='model/classifier.pkl'):
        print("📦 Loading classifier...")
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        print("✅ Classifier loaded.")
        self.embedder = Embedder()
        print("✅ Embedder ready.")

    def predict(self, text):
        print(f"🧠 Predicting for: {text}")
        vector = self.embedder.embed_single(text)
        print("📐 Vectorized input.")
        return self.classifier.predict([vector])[0]
