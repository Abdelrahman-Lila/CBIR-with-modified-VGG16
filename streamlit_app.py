import os
import streamlit as st
import numpy as np
from PIL import Image
from src.feature_extraction.vgg_feature_extractor import VGGNet
from src.retrieval.image_retrieval import retrieve_images
from src.feature_extraction.extract_features import extract_features
import h5py


def main():
    st.title("Content-Based Image Retrieval System")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded image temporarily
        img = Image.open(uploaded_file)
        img_path = "data/query_images/uploaded_image.jpg"
        img.save(img_path)

        # Extract features from all images if not already done
        features_path = "data/CNNFeatures.h5"
        st.write("Extracting features from all images...")
        extract_features("data/all_images", features_path)
        st.success("Feature extraction complete.")

        # Perform retrieval
        st.write("Retrieving similar images...")
        results, fig = retrieve_images(img_path, features_path, top_n=3)

        # Display uploaded image
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Show best match
        if results:
            best = results[0]
            st.write("Best Match:")
            st.image(best, caption="Best Match", use_container_width=True)

        # Show the 3D feature‐space plot
        st.write("Feature Space Visualization")
        st.pyplot(fig)

        # Top 3 side-by-side
        if results and len(results) > 1:
            st.write("Top 3 Matches:")
            cols = st.columns(len(results))
            for i, path in enumerate(results):
                with cols[i]:
                    st.image(path, caption=f"Match {i + 1}", use_container_width=True)


if __name__ == "__main__":
    main()
