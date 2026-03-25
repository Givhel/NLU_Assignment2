import numpy as np
import random

# -------------------------------
# Stopwords
# -------------------------------
stopwords = set([
    "the", "is", "and", "are", "to", "of", "in", "for",
    "on", "with", "as", "by", "an", "be", "this", "that",
    "it", "from", "or", "at", "was", "but",
    "may", "can", "will", "such", "also"
])

# -------------------------------
# Load corpus
# -------------------------------
def load_corpus():
    corpus = []

    with open("../Data/cleaned/corpus.txt", "r") as f:
        for line in f:
            words = line.strip().split()
            words = [w for w in words if w not in stopwords]
            corpus.append(words)

    return corpus


# -------------------------------
# Build vocab
# -------------------------------
def build_vocab(corpus):
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    return word2idx, idx2word


# -------------------------------
# Skip-gram pairs
# -------------------------------
def generate_skipgram_pairs(corpus, word2idx, window=4):
    pairs = []

    for sentence in corpus:
        for i, word in enumerate(sentence):
            target = word2idx[word]

            for j in range(max(0, i-window), min(len(sentence), i+window+1)):
                if i != j:
                    context = word2idx[sentence[j]]
                    pairs.append((target, context))

    return pairs


# -------------------------------
# CBOW pairs
# -------------------------------
def generate_cbow_pairs(corpus, word2idx, window=4):
    pairs = []

    for sentence in corpus:
        for i in range(len(sentence)):
            context = []

            for j in range(max(0, i-window), min(len(sentence), i+window+1)):
                if i != j:
                    context.append(word2idx[sentence[j]])

            target = word2idx[sentence[i]]
            pairs.append((context, target))

    return pairs


# -------------------------------
# Negative sampling
# -------------------------------
def get_negative_samples(vocab_size, positive, k):
    negatives = []
    while len(negatives) < k:
        neg = random.randint(0, vocab_size - 1)
        if neg != positive:
            negatives.append(neg)
    return negatives


# -------------------------------
# Skip-gram training
# -------------------------------
def train_skipgram(vocab_size, pairs, dim=100, epochs=25, lr=0.01, neg_samples=5):

    W = np.random.randn(vocab_size, dim)
    W_context = np.random.randn(vocab_size, dim)

    for epoch in range(epochs):
        loss = 0

        for target, context in pairs:

            v = W[target]

            # positive
            u = W_context[context]
            score = np.dot(v, u)
            pred = 1 / (1 + np.exp(-score))
            error = 1 - pred

            W[target] += lr * error * u
            W_context[context] += lr * error * v
            loss += -np.log(pred + 1e-9)

            # negative samples
            negatives = get_negative_samples(vocab_size, context, neg_samples)

            for neg in negatives:
                u_neg = W_context[neg]
                score_neg = np.dot(v, u_neg)
                pred_neg = 1 / (1 + np.exp(-score_neg))

                error_neg = 0 - pred_neg

                W[target] += lr * error_neg * u_neg
                W_context[neg] += lr * error_neg * v

                loss += -np.log(1 - pred_neg + 1e-9)

        avg_loss = loss / len(pairs)
        print(f"[Skip-gram] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return W


# -------------------------------
# CBOW training
# -------------------------------
def train_cbow(vocab_size, pairs, dim=100, epochs=25, lr=0.01, neg_samples=5):

    W = np.random.randn(vocab_size, dim)
    W_context = np.random.randn(vocab_size, dim)

    for epoch in range(epochs):
        loss = 0

        for context, target in pairs:

            v = np.mean([W[i] for i in context], axis=0)

            # positive
            u = W_context[target]
            score = np.dot(v, u)
            pred = 1 / (1 + np.exp(-score))
            error = 1 - pred

            for i in context:
                W[i] += lr * error * u / len(context)

            W_context[target] += lr * error * v
            loss += -np.log(pred + 1e-9)

            # negative samples
            negatives = get_negative_samples(vocab_size, target, neg_samples)

            for neg in negatives:
                u_neg = W_context[neg]
                score_neg = np.dot(v, u_neg)
                pred_neg = 1 / (1 + np.exp(-score_neg))

                error_neg = 0 - pred_neg

                for i in context:
                    W[i] += lr * error_neg * u_neg / len(context)

                W_context[neg] += lr * error_neg * v
                loss += -np.log(1 - pred_neg + 1e-9)

        avg_loss = loss / len(pairs)
        print(f"[CBOW] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return W


# -------------------------------
# Cosine similarity
# -------------------------------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# -------------------------------
# Nearest words
# -------------------------------
def get_nearest(word, W, word2idx, idx2word):

    if word not in word2idx:
        print(f"{word} not found")
        return

    vec = W[word2idx[word]]

    sims = []
    for i in range(len(W)):
        sim = cosine_similarity(vec, W[i])
        sims.append((idx2word[i], sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    print(f"\nNearest words for '{word}':")
    for w, s in sims[1:6]:
        print(w, round(s, 3))


# -------------------------------
# Analogy
# -------------------------------
def analogy(a, b, c, W, word2idx, idx2word):

    if a not in word2idx or b not in word2idx or c not in word2idx:
        print("Word missing")
        return

    vec = W[word2idx[b]] - W[word2idx[a]] + W[word2idx[c]]

    sims = []
    for i in range(len(W)):
        sim = cosine_similarity(vec, W[i])
        sims.append((idx2word[i], sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    print(f"\nAnalogy: {a} : {b} :: {c} : ?")
    for w, s in sims[:3]:
        print(w, round(s, 3))


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    corpus = load_corpus()
    word2idx, idx2word = build_vocab(corpus)

    print("Vocabulary size:", len(word2idx))

    sg_pairs = generate_skipgram_pairs(corpus, word2idx)
    cbow_pairs = generate_cbow_pairs(corpus, word2idx)

    print("\nTraining Skip-gram...")
    sg_embeddings = train_skipgram(len(word2idx), sg_pairs)

    print("\nTraining CBOW...")
    cbow_embeddings = train_cbow(len(word2idx), cbow_pairs)

    # -------------------------------
    # Analysis (Skip-gram)
    # -------------------------------
    print("\n--- Skip-gram Results ---")
    get_nearest("research", sg_embeddings, word2idx, idx2word)
    get_nearest("students", sg_embeddings, word2idx, idx2word)
    get_nearest("phd", sg_embeddings, word2idx, idx2word)
    get_nearest("learning", sg_embeddings, word2idx, idx2word)
    get_nearest("exams", sg_embeddings, word2idx, idx2word)

    analogy("students", "research", "phd", sg_embeddings, word2idx, idx2word)
    analogy("courses", "exams", "students", sg_embeddings, word2idx, idx2word)
    import numpy as np
import random

# -------------------------------
# Stopwords
# -------------------------------
stopwords = set([
    "the", "is", "and", "are", "to", "of", "in", "for",
    "on", "with", "as", "by", "an", "be", "this", "that",
    "it", "from", "or", "at", "was", "but",
    "may", "can", "will", "such", "also"
])

# -------------------------------
# Load corpus
# -------------------------------
def load_corpus():
    corpus = []

    with open("../Data/cleaned/corpus.txt", "r") as f:
        for line in f:
            words = line.strip().split()
            words = [w for w in words if w not in stopwords]
            corpus.append(words)

    return corpus


# -------------------------------
# Build vocab
# -------------------------------
def build_vocab(corpus):
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    return word2idx, idx2word


# -------------------------------
# Skip-gram pairs
# -------------------------------
def generate_skipgram_pairs(corpus, word2idx, window=4):
    pairs = []

    for sentence in corpus:
        for i, word in enumerate(sentence):
            target = word2idx[word]

            for j in range(max(0, i-window), min(len(sentence), i+window+1)):
                if i != j:
                    context = word2idx[sentence[j]]
                    pairs.append((target, context))

    return pairs


# -------------------------------
# CBOW pairs
# -------------------------------
def generate_cbow_pairs(corpus, word2idx, window=4):
    pairs = []

    for sentence in corpus:
        for i in range(len(sentence)):
            context = []

            for j in range(max(0, i-window), min(len(sentence), i+window+1)):
                if i != j:
                    context.append(word2idx[sentence[j]])

            target = word2idx[sentence[i]]
            pairs.append((context, target))

    return pairs


# -------------------------------
# Negative sampling
# -------------------------------
def get_negative_samples(vocab_size, positive, k):
    negatives = []
    while len(negatives) < k:
        neg = random.randint(0, vocab_size - 1)
        if neg != positive:
            negatives.append(neg)
    return negatives


# -------------------------------
# Skip-gram training
# -------------------------------
def train_skipgram(vocab_size, pairs, dim=100, epochs=25, lr=0.01, neg_samples=5):

    W = np.random.randn(vocab_size, dim)
    W_context = np.random.randn(vocab_size, dim)

    for epoch in range(epochs):
        loss = 0

        for target, context in pairs:

            v = W[target]

            # positive
            u = W_context[context]
            score = np.dot(v, u)
            pred = 1 / (1 + np.exp(-score))
            error = 1 - pred

            W[target] += lr * error * u
            W_context[context] += lr * error * v
            loss += -np.log(pred + 1e-9)

            # negative samples
            negatives = get_negative_samples(vocab_size, context, neg_samples)

            for neg in negatives:
                u_neg = W_context[neg]
                score_neg = np.dot(v, u_neg)
                pred_neg = 1 / (1 + np.exp(-score_neg))

                error_neg = 0 - pred_neg

                W[target] += lr * error_neg * u_neg
                W_context[neg] += lr * error_neg * v

                loss += -np.log(1 - pred_neg + 1e-9)

        avg_loss = loss / len(pairs)
        print(f"[Skip-gram] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return W


# -------------------------------
# CBOW training
# -------------------------------
def train_cbow(vocab_size, pairs, dim=100, epochs=25, lr=0.01, neg_samples=5):

    W = np.random.randn(vocab_size, dim)
    W_context = np.random.randn(vocab_size, dim)

    for epoch in range(epochs):
        loss = 0

        for context, target in pairs:

            v = np.mean([W[i] for i in context], axis=0)

            # positive
            u = W_context[target]
            score = np.dot(v, u)
            pred = 1 / (1 + np.exp(-score))
            error = 1 - pred

            for i in context:
                W[i] += lr * error * u / len(context)

            W_context[target] += lr * error * v
            loss += -np.log(pred + 1e-9)

            # negative samples
            negatives = get_negative_samples(vocab_size, target, neg_samples)

            for neg in negatives:
                u_neg = W_context[neg]
                score_neg = np.dot(v, u_neg)
                pred_neg = 1 / (1 + np.exp(-score_neg))

                error_neg = 0 - pred_neg

                for i in context:
                    W[i] += lr * error_neg * u_neg / len(context)

                W_context[neg] += lr * error_neg * v
                loss += -np.log(1 - pred_neg + 1e-9)

        avg_loss = loss / len(pairs)
        print(f"[CBOW] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return W


# -------------------------------
# Cosine similarity
# -------------------------------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# -------------------------------
# Nearest words
# -------------------------------
def get_nearest(word, W, word2idx, idx2word):

    if word not in word2idx:
        print(f"{word} not found")
        return

    vec = W[word2idx[word]]

    sims = []
    for i in range(len(W)):
        sim = cosine_similarity(vec, W[i])
        sims.append((idx2word[i], sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    print(f"\nNearest words for '{word}':")
    for w, s in sims[1:6]:
        print(w, round(s, 3))


# -------------------------------
# Analogy
# -------------------------------
def analogy(a, b, c, W, word2idx, idx2word):

    if a not in word2idx or b not in word2idx or c not in word2idx:
        print("Word missing")
        return

    vec = W[word2idx[b]] - W[word2idx[a]] + W[word2idx[c]]

    sims = []
    for i in range(len(W)):
        sim = cosine_similarity(vec, W[i])
        sims.append((idx2word[i], sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    print(f"\nAnalogy: {a} : {b} :: {c} : ?")
    for w, s in sims[:3]:
        print(w, round(s, 3))


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    corpus = load_corpus()
    word2idx, idx2word = build_vocab(corpus)

    print("Vocabulary size:", len(word2idx))

    sg_pairs = generate_skipgram_pairs(corpus, word2idx)
    cbow_pairs = generate_cbow_pairs(corpus, word2idx)

    print("\nTraining Skip-gram...")
    sg_embeddings = train_skipgram(len(word2idx), sg_pairs)

    print("\nTraining CBOW...")
    cbow_embeddings = train_cbow(len(word2idx), cbow_pairs)

    # -------------------------------
    # Analysis (Skip-gram)
    # -------------------------------
    print("\n--- Skip-gram Results ---")
    get_nearest("research", sg_embeddings, word2idx, idx2word)
    get_nearest("students", sg_embeddings, word2idx, idx2word)
    get_nearest("phd", sg_embeddings, word2idx, idx2word)
    get_nearest("learning", sg_embeddings, word2idx, idx2word)
    get_nearest("exams", sg_embeddings, word2idx, idx2word)

    analogy("students", "research", "phd", sg_embeddings, word2idx, idx2word)
    analogy("courses", "exams", "students", sg_embeddings, word2idx, idx2word)
    analogy("phd", "research", "students", sg_embeddings, word2idx, idx2word)
    # -------------------------------
    # Analysis (CBOW)
    # -------------------------------
    print("\n--- CBOW Results ---")
    get_nearest("research", cbow_embeddings, word2idx, idx2word)
    get_nearest("students", cbow_embeddings, word2idx, idx2word)
    get_nearest("phd", cbow_embeddings, word2idx, idx2word)
    get_nearest("learning", cbow_embeddings, word2idx, idx2word)
    get_nearest("exams", cbow_embeddings, word2idx, idx2word)

    analogy("students", "research", "phd", sg_embeddings, word2idx, idx2word)
    analogy("courses", "exams", "students", sg_embeddings, word2idx, idx2word)
    analogy("phd", "research", "students", sg_embeddings, word2idx, idx2word)


    
