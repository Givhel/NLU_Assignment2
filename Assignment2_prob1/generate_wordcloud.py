from wordcloud import WordCloud
import matplotlib.pyplot as plt


def load_corpus():
    """
    Load processed corpus from cleaned file
    """
    corpus = []

    with open("../Data/cleaned/corpus.txt", "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(line.strip().split())

    return corpus


def generate_wordcloud(corpus):

    text = " ".join([" ".join(doc) for doc in corpus])

    wc = WordCloud(width=800, height=400).generate(text)

    plt.imshow(wc)
    plt.axis("off")
    plt.title("Word Cloud of IIT Jodhpur Dataset")

    plt.savefig("../Output/wordcloud.png")
    plt.show()


if __name__ == "__main__":

    corpus = load_corpus()
    generate_wordcloud(corpus)