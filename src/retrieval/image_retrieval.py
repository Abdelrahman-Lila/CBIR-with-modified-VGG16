def retrieve_images(query_image_path, features_path, top_n=3):
    import numpy as np
    import h5py
    from sklearn.metrics.pairwise import cosine_similarity
    from PIL import Image

    # Load the features
    with h5py.File(features_path, 'r') as f:
        features = f['features'][:]
        image_paths = list(f['image_paths'])

    # Load the query image and extract its features
    query_image = Image.open(query_image_path)
    query_image = query_image.resize((224, 224))  # Resize to match the model input
    query_image = np.array(query_image) / 255.0  # Normalize
    query_image = query_image.flatten()  # Flatten the image

    # Compute cosine similarity between the query image and all features
    similarities = cosine_similarity([query_image], features)[0]

    # Get the top N similar images
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_similarities = similarities[top_indices]
    top_image_paths = [image_paths[i] for i in top_indices]

    return top_image_paths, top_similarities