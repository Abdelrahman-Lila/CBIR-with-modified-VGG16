import h5py
import numpy as np
from scipy import spatial
from ..feature_extraction.vgg_feature_extractor import VGGNet
from .visualization import plot_feature_space

def retrieve_images(query_img_path, features_path, top_n=3):
    # Load features
    with h5py.File(features_path, 'r') as h5f:
        feats = h5f['dataset_1'][:]
        imgNames = h5f['dataset_2'][:]
    
    # Extract query image features
    model = VGGNet()
    query_feat = model.extract_feat(query_img_path)
    
    # Compute similarity scores
    scores = []
    for i in range(feats.shape[0]):
        score = 1 - spatial.distance.cosine(query_feat, feats[i])
        scores.append(score)
    scores = np.array(scores)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    
    # Get top matches
    top_matches = rank_ID[:top_n]
    top_scores = rank_score[:top_n]
    
    # Print results
    print(f"Top {top_n} matches with similarity scores:")
    for i, (image_id, score) in enumerate(zip(top_matches, top_scores)):
        image_name = imgNames[image_id].decode('utf-8') if isinstance(imgNames[image_id], bytes) else imgNames[image_id]
        print(f"{i+1}. Image: {image_name}, Score: {score:.4f}")
    
    # Visualize
    plot_feature_space(feats, query_feat, top_matches, imgNames, 
                      "VGG16 Feature Space - Image Retrieval Results")