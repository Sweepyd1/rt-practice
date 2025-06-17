import cv2 

import os

import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
import hdbscan  
from glob import glob


input_folder = './data/grains'

output_base_folder = 'saved'


methods = ['kmeans', 'meanshift', 'dbscan', 'hdbscan', 'optics', 'gmm']
for m in methods:
    os.makedirs(os.path.join(output_base_folder, m), exist_ok=True)

def preprocess_image(img):
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    pixel_values = img_lab.reshape((-1, 3))

    pixel_values = pixel_values.astype(np.float32)
    pixel_values /= 255.0
    return pixel_values

def save_clustered_image(labels, img_shape, output_path):
    labels_img = labels.reshape(img_shape[:2])
    labels_norm = cv2.normalize(labels_img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    labels_norm = labels_norm.astype(np.uint8)
    colored = cv2.applyColorMap(labels_norm, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, colored)

def cluster_kmeans(data, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return labels

def cluster_meanshift(data):
    model = MeanShift()
    labels = model.fit_predict(data)
    return labels

def cluster_dbscan(data, eps=0.05, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    return labels

def cluster_hdbscan(data, min_cluster_size=10):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(data)
    return labels

def cluster_optics(data, max_eps=0.1, min_samples=10):
    model = OPTICS(max_eps=max_eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    return labels

def cluster_gmm(data, n_components=3):
    model = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=42)
    labels = model.fit_predict(data)
    return labels


image_paths = glob(os.path.join(input_folder, '*.*'))  
for img_path in image_paths:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Не удалось загрузить {img_path}")
        continue

    data = preprocess_image(img)

    try: 
        labels = cluster_kmeans(data, n_clusters=3)
        save_clustered_image(labels, img.shape, os.path.join(output_base_folder, 'kmeans', f'{img_name}_kmeans.png'))

        labels = cluster_meanshift(data)
        save_clustered_image(labels, img.shape, os.path.join(output_base_folder, 'meanshift', f'{img_name}_meanshift.png'))

        labels = cluster_dbscan(data)
        save_clustered_image(labels, img.shape, os.path.join(output_base_folder, 'dbscan', f'{img_name}_dbscan.png'))

        labels = cluster_hdbscan(data)
        save_clustered_image(labels, img.shape, os.path.join(output_base_folder, 'hdbscan', f'{img_name}_hdbscan.png'))

        labels = cluster_optics(data)
        save_clustered_image(labels, img.shape, os.path.join(output_base_folder, 'optics', f'{img_name}_optics.png'))

        labels = cluster_gmm(data, n_components=3)
        save_clustered_image(labels, img.shape, os.path.join(output_base_folder, 'gmm', f'{img_name}_gmm.png'))
    except Exception as e:
        print(e)
    finally:
        print("все!")
