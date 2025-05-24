# Streamlit CBIR Application

This project is a Content-Based Image Retrieval (CBIR) application built using Streamlit. It allows users to upload a query image and retrieves similar images from a dataset, displaying the results along with a similarity graph.

## Project Structure

```
streamlit-cbir-app
├── src
│   ├── feature_extraction
│   │   └── extract_features.py
│   ├── retrieval
│   │   └── image_retrieval.py
│   └── utils
│       └── visualization.py
├── data
│   ├── all_images
│   └── query_images
├── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-cbir-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your images in the `data/all_images` directory for the dataset and any query images in the `data/query_images` directory.

2. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to `http://localhost:8501` to access the application.

4. Upload a query image and view the retrieved similar images along with the similarity graph.

## Features

- Upload a query image for searching.
- Retrieve and display the most similar images from the dataset.
- Visualize the similarity between the query image and the retrieved images.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.