import numpy as np
from data_util import get_iris_data
from scipy.stats import mode

class Kmeans:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, data):
        # Define the number of features and clusters
        num_samples = data.shape[0]
        num_features = data.shape[1]
        num_clusters = self.k
        
        # Initialize centroids randomly from the data points
        random_indices = np.random.choice(num_samples, num_clusters, replace=False)
        centroids = data[random_indices]
        labels = np.zeros(num_samples, dtype=int)
        
        for _ in range(self.max_iters):
            # Iterate through all points and calculate the cluster assignment
            for i in range(num_samples):
                self.assign_cluster(data[i], centroids, labels, i)

            # Recompute centroids
            new_centroids = np.zeros((num_clusters, num_features))
            for i in range(num_clusters):
                points_in_cluster = data[labels == i]
                if len(points_in_cluster) > 0:
                    new_centroids[i] = points_in_cluster.mean(axis=0)

            # Check for convergence
            if np.allclose(centroids, new_centroids, atol=self.tol):
                break
            centroids = new_centroids

        self.centroids = centroids
        return labels

    def euclidian_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def assign_cluster(self, point, centroids, labels, index):
        distances_arr = []

        # Loop over each cluster centroid
        for ctrd in centroids:
            # Compute the distance from the point to the current centroid
            distance = self.euclidian_distance(point, ctrd)
            distances_arr.append(distance)

        # Find the index of the closest centroid (i.e., the one with the minimum distance)
        closest_cluster_index = np.argmin(distances_arr)
        
        # Assign the point to the closest cluster
        labels[index] = closest_cluster_index

    def cal_accuracy(self, label_pre, label_true):
        if label_pre.ndim == 2:
            label_pre = label_pre.flatten()

        if label_true.ndim == 2:
            label_true = label_true.flatten()

        print('Predicted labels: ', label_pre)
        print('True labels: ', label_true)

        return np.mean(label_pre == label_true)

def iris_classification():
    # Load data from file
    train_x, train_y, test_x, test_y = get_iris_data()
    train_y = train_y - 1
    num_class = len(np.unique(train_y))

    # Train a KMeans model
    kmeans = Kmeans(num_class)
    labels = kmeans.fit(train_x)

    # Map predicted labels to closest true labels
    mapped_labels = np.zeros_like(labels)
    for i in range(num_class):
        mask = (labels == i)
        mapped_labels[mask] = mode(train_y[mask])[0]

    # Calculate accuracy
    accuracy = kmeans.cal_accuracy(mapped_labels, train_y)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    iris_classification()