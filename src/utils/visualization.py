def plot_similarity_graph(similarity_scores, best_match_image):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(similarity_scores)), similarity_scores, color='blue')
    plt.xlabel('Image Index')
    plt.ylabel('Similarity Score')
    plt.title('Image Similarity Scores')
    plt.xticks(range(len(similarity_scores)), [f'Image {i}' for i in range(len(similarity_scores))])
    plt.axhline(y=max(similarity_scores), color='r', linestyle='--', label='Best Match')
    plt.legend()
    plt.grid()

    plt.savefig('similarity_graph.png')
    plt.close()

def display_best_match(best_match_image):
    from PIL import Image
    import streamlit as st

    image = Image.open(best_match_image)
    st.image(image, caption='Best Match', use_column_width=True)