# CBIR Streamlit Application

This project implements a Content-Based Image Retrieval (CBIR) system using a modified VGG16 model for feature extraction and similarity retrieval. The application is built using Streamlit, allowing users to upload images and visualize the results interactively.

## Project Structure

```
cbir-streamlit-app
├── data
│   ├── all_images          # Directory containing images for feature extraction
│   └── query_images        # Directory for user-uploaded query images
├── src
│   ├── feature_extraction
│   │   ├── extract_features.py  # Extracts features from images
│   │   └── vgg_feature_extractor.py  # Defines the VGGNet class for feature extraction
│   └── retrieval
│       ├── image_retrieval.py  # Retrieves similar images and computes similarity scores
│       └── visualization.py      # Visualizes the feature space in 3D
├── streamlit_app.py             # Main Streamlit application file
├── requirements.txt              # Lists project dependencies
└── README.md                     # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd cbir-streamlit-app
   ```

2. **Install dependencies**:
   It is recommended to create a virtual environment before installing the dependencies.
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the data**:
   - Place your images in the `data/all_images` directory for feature extraction.
   - Upload your query images to the `data/query_images` directory.

## Usage

1. **Run the Streamlit application**:
   ```
   streamlit run streamlit_app.py
   ```

2. **Upload an image**:
   - Use the provided interface to upload a query image.

3. **View results**:
   - The application will extract features from the uploaded image, retrieve similar images from the database, and display a 3D graph of the feature space along with the best match.

## Dependencies

- Streamlit
- NumPy
- Matplotlib
- TensorFlow
- h5py
- scikit-learn

## License

This project is licensed under the MIT License. See the LICENSE file for details.