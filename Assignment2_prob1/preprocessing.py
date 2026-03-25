import os
import re


def read_files():
    """
    Reads all .txt files from raw folder
    """
    folder_path = "../Data/raw"
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(text)

    return documents


def clean_text(text):
    """
    Clean text:
    - lowercase
    - remove numbers
    - remove punctuation
    """

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text):
    """
    Split text into words
    """
    return text.split()


def preprocess_documents(documents):
    """
    Apply cleaning + tokenization
    """
    processed = []

    for doc in documents:
        cleaned = clean_text(doc)
        tokens = tokenize(cleaned)

        # removing small words (just to improve quality)
        tokens = [w for w in tokens if len(w) > 2]

        processed.append(tokens)

    return processed


def save_corpus(corpus):
    """
    Save cleaned corpus
    """
    output_path = "../Data/cleaned/corpus.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(" ".join(doc) + "\n")

    print(" Corpus saved at:", output_path)


def dataset_stats(corpus):
    """
    Print dataset statistics
    """
    total_docs = len(corpus)
    total_tokens = sum(len(doc) for doc in corpus)

    vocab = set()
    for doc in corpus:
        vocab.update(doc)

    print("\n Dataset Stats")
    print("Documents:", total_docs)
    print("Tokens:", total_tokens)
    print("Vocabulary:", len(vocab))


if __name__ == "__main__":

    print(" Starting preprocessing...")

    raw_docs = read_files()
    corpus = preprocess_documents(raw_docs)

    save_corpus(corpus)
    dataset_stats(corpus)