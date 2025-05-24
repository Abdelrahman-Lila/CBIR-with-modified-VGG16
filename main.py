import os
from src.feature_extraction.extract_features import extract_features
from src.retrieval.image_retrieval import retrieve_images

def main():
    # Paths
    images_path = "data/all_images"
    query_images_path = "data/query_images"
    features_path = "data/CNNFeatures.h5"
    visualization_path = "feature_space.png"
    
    # Extract features from all images
    if not os.path.exists(features_path):
        print("Extracting features...")
        extract_features(images_path, features_path)
    else:
        print("Features already extracted, skipping extraction.")
    
    # Perform retrieval with a query image
    query_image = os.path.join(query_images_path, "tigerq.jpg")
    if os.path.exists(query_image):
        print(f"\nPerforming retrieval for query image: {query_image}")
        retrieve_images(query_image, features_path, top_n=3)
        print(f"Visualization saved to: {visualization_path}")
    else:
        print(f"Query image {query_image} not found.")

if __name__ == "__main__":
    main()