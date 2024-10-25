import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

def fuzzy_c_means(image, num_clusters=2, m=2, max_iter=100, error=1e-5):
    if len(image.shape) == 2:  # Grayscale image
        pixels = image.flatten().astype(float)
        channels = 1
    else:  # RGB image
        pixels = image.reshape(-1, 3).astype(float)
        channels = 3

    N = len(pixels)

    # Initalisation des clusters
    centers = np.random.rand(num_clusters, channels) * 255
    U = np.random.dirichlet(np.ones(num_clusters), N)  

    # Fuzzy C-means loop
    for iteration in range(max_iter):
        U_old = U.copy()

        # Calcul des distances
        distances = np.zeros((N, num_clusters))
        for j in range(num_clusters):
            if channels == 1: # difference absolue pour nv gris
                distances[:, j] = np.abs(pixels - centers[j]) 
            else: # distance euclidienne pour RGB
                distances[:, j] = np.linalg.norm(pixels - centers[j], axis=1)

        # Mise à jour de la matrice
        U = 1 / (distances**(2 / (m - 1)) + 1e-8)
        U = U / np.sum(U, axis=1, keepdims=True)

        # Mise à jour des centroids
        for j in range(num_clusters):
            centers[j] = np.dot(U[:, j]**m, pixels) / np.sum(U[:, j]**m)

        ### Afficher les images segmentées à chaque itérations
        #print(f"Iteration {iteration + 1}")
        #visualize_clusters(U, centers, image.shape, channels)

        # Critère d'arret
        if np.linalg.norm(U - U_old) < error:
            break

    return U, centers

# Charger une image
def load_image(image_path):
    image = io.imread(image_path)
    return image

# Afficher les images segmentées
def visualize_clusters(U, centers, image_shape, channels):
    # images nv gris
    if channels == 1:
        cluster_map = np.argmax(U, axis=1)  
        segmented_image = centers[cluster_map].reshape(image_shape).astype(np.uint8)

        plt.imshow(segmented_image, cmap='gray')
        plt.title("Image segmentée par Fuzzy C-means (Grayscale)")
    else:
        # images RGB
        segmented_image = np.dot(U, centers).reshape(image_shape).astype(np.uint8)

        plt.imshow(segmented_image)
        plt.title("Image segmentée par Fuzzy C-means (RGB)")
    plt.axis('off')
    plt.show()

# Heatmaps
def plot_heatmaps(U, image_shape, num_clusters):
    for cluster_index in range(num_clusters):
        membership_map = U[:, cluster_index].reshape(image_shape[0], image_shape[1])
        plt.imshow(membership_map, cmap='hot')
        plt.colorbar()
        plt.title(f"Carte de chaleur pour le cluster {cluster_index}")
        plt.show()

image_path = 'milky-way.jpg'
image = load_image(image_path)
image_shape = image.shape

num_clusters = 3
U, centers = fuzzy_c_means(image, num_clusters=num_clusters)

plot_heatmaps(U, image_shape, num_clusters)
