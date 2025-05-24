def extract_features(images_path, features_path):
    import os
    import numpy as np
    from keras.applications.vgg16 import VGG16, preprocess_input
    from keras.preprocessing.image import load_img, img_to_array
    from keras.models import Model
    import h5py

    # Load VGG16 model + higher level layers
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)

    features = []
    image_names = []

    # Loop through all images in the dataset
    for img_name in os.listdir(images_path):
        img_path = os.path.join(images_path, img_name)
        if os.path.isfile(img_path):
            # Load and preprocess the image
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Extract features
            feature = model.predict(img_array)
            features.append(feature.flatten())
            image_names.append(img_name)

    # Save features to an HDF5 file
    with h5py.File(features_path, 'w') as f:
        f.create_dataset('features', data=np.array(features))
        f.create_dataset('image_names', data=np.array(image_names, dtype='S'))  # Store names as bytes

    print(f"Extracted features for {len(features)} images and saved to {features_path}.")