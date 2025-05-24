import h5py
import numpy as np
import os
from scipy import spatial
from ..feature_extraction.vgg_feature_extractor import VGGNet
from .visualization import plot_feature_space


def retrieve_images(query_img, features_path, top_n=3):
    # Load features
    with h5py.File(features_path, "r") as h5f:
        feats = h5f["dataset_1"][:]
        imgNames = h5f["dataset_2"][:]

    # Extract query image features
    model = VGGNet()
    query_feat = model.extract_feat(query_img)

    # Compute cosine‐similarity scores
    scores = 1 - np.array([spatial.distance.cosine(query_feat, f) for f in feats])
    rank_ID = np.argsort(scores)[::-1]
    top_ids = rank_ID[:top_n]

    # (Optional) print & visualize
    for i, idx in enumerate(top_ids):
        name = imgNames[idx]
        name = name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else name
        print(f"{i + 1}. {name} => {scores[idx]:.4f}")
    fig = plot_feature_space(
        feats, query_feat, top_ids, imgNames, "VGG16 Feature Space - Retrieval Results"
    )

    # Build results list
    img_dir = "data/all_images"
    results = []
    for idx in top_ids:
        name = imgNames[idx]
        if isinstance(name, (bytes, bytearray)):
            name = name.decode("utf-8")
        results.append(os.path.join(img_dir, name))

    return results, fig
