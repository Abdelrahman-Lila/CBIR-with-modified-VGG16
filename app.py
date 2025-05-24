import streamlit as st
import os
from PIL import Image
import numpy as np
from src.feature_extraction.extract_features import extract_features
from src.retrieval.image_retrieval import retrieve_images
from src.utils.visualization import plot_similarity_graph

def main():
    st.title("Content-Based Image Retrieval (CBIR) System")
    
    st.sidebar.header("Upload Query Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption='Uploaded Image', use_column_width=True)
        
        # Save the uploaded image temporarily
        query_image_path = "temp_query_image.jpg"
        query_image.save(query_image_path)
        
        features_path = "/media/ahmed/Work/Materials/8th Sem/Computer Vision/Labs/streamlit-cbir-app/data/CNNFeatures.h5"
        images_dir = "/media/ahmed/Work/Materials/8th Sem/Computer Vision/Labs/streamlit-cbir-app/data/all_images"

        if not os.path.exists(features_path):
            st.warning("Features not extracted yet. Running feature extraction...")
            with st.spinner("Extracting features..."):
                extract_features(images_dir, features_path)
            st.success("Feature extraction completed. Performing retrieval...")
        else:
            st.success("Features found. Performing retrieval...")

        best_matches, similarity_scores = retrieve_images(query_image_path, features_path, top_n=3)
        
        if best_matches:
            st.subheader("Best Matches:")
            for i, (match, score) in enumerate(zip(best_matches, similarity_scores)):
                st.image(match, caption=f'Match {i+1} - Similarity Score: {score:.2f}', use_column_width=True)
            
            st.subheader("Similarity Graph")
            plot_similarity_graph(similarity_scores)
        else:
            st.error("No matches found.")

if __name__ == "__main__":
    main()