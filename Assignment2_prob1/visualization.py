import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# import your models
from word2vec_full import (
    load_corpus,
    build_vocab,
    generate_skipgram_pairs,
    generate_cbow_pairs,
    train_skipgram,
    train_cbow
)

# -------------------------------
# Train both models (same settings)
# -------------------------------
def get_embeddings():

    corpus = load_corpus()
    word2idx, idx2word = build_vocab(corpus)

    sg_pairs = generate_skipgram_pairs(corpus, word2idx, window=4)
    cbow_pairs = generate_cbow_pairs(corpus, word2idx, window=4)

    print("Training Skip-gram for visualization...")
    sg_embeddings = train_skipgram(len(word2idx), sg_pairs, epochs=10)

    print("Training CBOW for visualization...")
    cbow_embeddings = train_cbow(len(word2idx), cbow_pairs, epochs=10)

    return sg_embeddings, cbow_embeddings, idx2word


# -------------------------------
# PCA Visualization
# -------------------------------
def plot_embeddings(embeddings, idx2word, title, filename, num_words=60):

    words = list(idx2word.values())[:num_words]
    vectors = embeddings[:num_words]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 8))

    for i, word in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=8)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.savefig(f"../Output/{filename}")
    plt.close()


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    print("🚀 Generating visualization for both models...")

    sg_embeddings, cbow_embeddings, idx2word = get_embeddings()

    # Skip-gram visualization
    plot_embeddings(
        sg_embeddings,
        idx2word,
        "Skip-gram Embedding Visualization (PCA)",
        "skipgram_visualization.png"
    )

    # CBOW visualization
    plot_embeddings(
        cbow_embeddings,
        idx2word,
        "CBOW Embedding Visualization (PCA)",
        "cbow_visualization.png"
    )

    print("✅ Both visualizations saved in Output folder!")