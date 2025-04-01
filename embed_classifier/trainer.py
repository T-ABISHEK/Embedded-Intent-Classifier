import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from embed_classifier.embedder import Embedder
import os

def train_classifier(prompts, labels, save_dir='model'):
    os.makedirs(save_dir, exist_ok=True)
    embedder = Embedder()
    X = embedder.embed(prompts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    with open(f"{save_dir}/classifier.pkl", "wb") as f:
        pickle.dump(clf, f)