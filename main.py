from embed_classifier.loader import load_csv
from embed_classifier.trainer import train_classifier

if __name__ == '__main__':
    prompts, labels = load_csv("data/labeled_prompts.csv")
    train_classifier(prompts, labels)