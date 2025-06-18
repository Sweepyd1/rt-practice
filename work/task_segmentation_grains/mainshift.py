import os
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

def meanshift_segmentation(image_path, save_folder='saved/meanshift', resize_max=200):
    os.makedirs(save_folder, exist_ok=True)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Не удалось прочитать {image_path}")
        return
    h, w = img_bgr.shape[:2]
    scale = resize_max / max(h, w) if max(h, w) > resize_max else 1
    img_small = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)


    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)

    n_samples = min(500, pixels.shape[0])
    sample_pixels = pixels[np.random.choice(pixels.shape[0], n_samples, replace=False)]

    bandwidth = estimate_bandwidth(sample_pixels, quantile=0.2)
    if bandwidth <= 0:
        bandwidth = 30  

 
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pixels)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    n_clusters = len(np.unique(labels))
    print(f"MeanShift: найдено кластеров: {n_clusters}")


    colors = np.random.randint(0, 255, size=(n_clusters, 3), dtype=np.uint8)
    segmented_img = colors[labels].reshape(img_rgb.shape)


    save_path = os.path.join(save_folder, os.path.basename(image_path))
    segmented_bgr = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, segmented_bgr)
    print(f"Сегментированное изображение сохранено: {save_path}")


if __name__ == "__main__":
    image_folder = "data/grains"
    for image_name in os.listdir(image_folder):
        full_path = os.path.join(image_folder, image_name)
        print(full_path)
        meanshift_segmentation(full_path)
