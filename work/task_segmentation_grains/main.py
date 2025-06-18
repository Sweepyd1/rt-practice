import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, MeanShift, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
import hdbscan
import gc
from sklearn.cluster import  estimate_bandwidth


input_folder = 'data/grains'
output_base_folder = 'saved'
methods = ['kmeans', 'meanshift', 'dbscan', 'hdbscan', 'optics', 'gmm']

for method in methods:
    os.makedirs(os.path.join(output_base_folder, method), exist_ok=True)

def resize_image(image, max_size=200):
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return image


def image_to_feature_vector(image):
    return image.reshape((-1, 3))

def cluster_and_save(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Не удалось прочитать {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Уменьшаем размер для экономии памяти
    img_rgb_small = resize_image(img_rgb, max_size=200)
    pixels = image_to_feature_vector(img_rgb_small).astype(np.float32)

    clustered_images = {}


    mbk = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=1000)
    labels = mbk.fit_predict(pixels)
    clustered_images['kmeans'] = labels


    dbscan = DBSCAN(eps=10, min_samples=5)
    labels = dbscan.fit_predict(pixels)
    clustered_images['dbscan'] = labels


    hdb = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = hdb.fit_predict(pixels)
    clustered_images['hdbscan'] = labels


    optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
    labels = optics.fit_predict(pixels)
    clustered_images['optics'] = labels

    gmm = GaussianMixture(n_components=4, random_state=42)
    labels = gmm.fit_predict(pixels)
    clustered_images['gmm'] = labels

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    h, w = img_rgb_small.shape[:2]

    for method_name, labels in clustered_images.items():
        unique_labels = np.unique(labels)
        black = [0, 0, 0]
        random_colors = [list(np.random.randint(0, 256, size=3)) for _ in range(len(unique_labels) - 1)]
        colors = np.array([black] + random_colors, dtype=np.uint8)
        label_indices = np.array([np.where(unique_labels == l)[0][0] for l in labels])
        segmented_img = colors[label_indices].reshape((h, w, 3))
        segmented_bgr = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(output_base_folder, method_name, f"{base_name}_{method_name}.png")
        cv2.imwrite(save_path, segmented_bgr)
        print(f"Сохранено: {save_path}")

    del pixels, clustered_images, labels
    gc.collect()

def main():
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            full_path = os.path.join(input_folder, filename)
            print(f"Обрабатываем {filename}...")
            cluster_and_save(full_path)
            

if __name__ == "__main__":
    main()
