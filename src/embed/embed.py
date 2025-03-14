import pickle
import re
import sys
import unicodedata
from collections import defaultdict

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import BertModel, BertTokenizer

import utils

SRC = sys.argv[1]


ocr_engine = ocr_predictor(
    pretrained=True, det_arch="db_resnet50", reco_arch="crnn_vgg16_bn"
)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
embedder = BertModel.from_pretrained("bert-base-uncased")


def run_ocr(imgp):
    docs = DocumentFile.from_images(imgp)
    res = ocr_engine(docs)
    return res


def clean(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'()\[\]{}\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text


def cluster_bert_embeddings(embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5, metric="euclidean", cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(embeddings)
    return labels


def visualize_clusters(embeddings, labels, method="umap"):
    if method == "umap":
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    else:
        raise ValueError("Unsupported visualization method")

    reduced_embeddings = reducer.fit_transform(embeddings)
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
    )
    plt.colorbar()
    plt.legend()
    plt.title("Cluster Visualization")
    plt.show()


def multi_chunk_embedding(big_chunk):
    chunk_size = 512
    chunks = [
        big_chunk[i : i + chunk_size] for i in range(0, len(big_chunk), chunk_size)
    ]
    embeddings = []
    for chunk in chunks:
        encoded_input = tokenizer(chunk, return_tensors="pt")
        output = embedder(**encoded_input)
        embeddings.append(output.last_hidden_state.mean(dim=1).detach().numpy())
    return embeddings


if __name__ == "__main__":
    imagepaths = utils.find_images(SRC)
    texts = [run_ocr(imgp).render() for imgp in imagepaths]
    print(texts)
    pickle.dump(texts, open("texts.pkl", "wb"))
    # texts = pickle.load(open("texts.pkl", "rb"))

    embeddings = []
    for str_res in texts:
        try:
            encoded_input = tokenizer(str_res, return_tensors="pt")
            if len(encoded_input["input_ids"]) > 512:
                embeddings.extend(multi_chunk_embedding(str_res))
            else:
                output = embedder(**encoded_input)
                embeddings.append(
                    output.last_hidden_state.mean(dim=1).detach().numpy()[0]
                )
        except Exception as e:
            print(f"Error: {e} at {str_res}")

    np.save("embeddings.npy", embeddings)
    # embeddings = np.load("embeddings.npy")

    labels = cluster_bert_embeddings(embeddings)
    txts = defaultdict(list)
    for i, label in enumerate(labels):
        txts[label].append(texts[i])
    for k, v in txts.items():
        print(f"Cluster {k}: {v}")

    visualize_clusters(embeddings, labels)
