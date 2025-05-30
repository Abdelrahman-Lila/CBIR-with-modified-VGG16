import os
import h5py
import numpy as np
from .vgg_feature_extractor import VGGNet

def extract_features(images_path, output_path):
    img_list = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    print("Start feature extraction")
    
    model = VGGNet()
    feats = []
    names = []
    
    for im in img_list:
        print(f"Extracting features from image - {im}")
        X = model.extract_feat(im)
        feats.append(X)
        names.append(os.path.basename(im))
    
    feats = np.array(feats)
    
    print(f"Writing feature extraction results to {output_path}")
    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('dataset_1', data=feats)
        h5f.create_dataset('dataset_2', data=np.bytes_(names))