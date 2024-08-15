import numpy as np
from data_util import get_iris_data, load_iris_data, kfold_split

NUM_CLUSTER = 2

class GMM:
    def __init__(self, id, k):
        self.id = id
        self.k = k
        self.pi = []
        self.mu = []
        self.sigma = []

    def fit(self, data):
        # Define the number of features and clusters
        num_samples = data.shape[0]
        num_features = data.shape[1]
        num_clusters = self.k

        # Declare variables
        pi = np.ones(num_clusters) / num_clusters
        mu = np.zeros((num_clusters, num_features))
        labels = np.zeros(num_samples)
        sigma = []
        for i in range(num_clusters):
            sigma.append(np.zeros((num_features, num_features)))

        # Initialize clusters & labels
        cluster_start = np.array_split(data, num_clusters)
        for i in range(num_clusters):
            mu[i] = np.mean(cluster_start[i], axis=0)
            sigma[i] = np.cov(cluster_start[i].T)

        # Calculate log likelihood for comparison after the iteration
        log_l_prev = self.log_likelihood(data, mu, sigma, pi)

        # Loop until convergence
        x = True
        while x:
            # Assign cluster to each point
            for i in range(num_samples):
                labels[i] = self.assign_cluster(data[i], mu, sigma, pi)

            self.optimize_mu(data, mu, sigma, pi)
            self.optimize_sigma(data, mu, sigma, pi)
            self.optimize_pi(data, mu, sigma, pi)

            log_l_post = self.log_likelihood(data, mu, sigma, pi)
            print(self.id, '- Log likelihood: ', log_l_post)

            if np.abs(log_l_post - log_l_prev) < 1e-5:
                x = False
                self.pi = pi
                self.mu = mu
                self.sigma = sigma
            else:
                log_l_prev = log_l_post

    def multivariate_gaussian(self, point, mu, sigma, pi):
        diff = point - mu
        d = len(point)

        norm_factor = (2 * np.pi) ** (-d / 2)
        det_sigma = np.linalg.det(sigma) ** (-1 / 2)
        exponent = -0.5 * diff.T @ np.linalg.inv(sigma) @ diff
        result = pi * norm_factor * det_sigma * np.exp(exponent)
        
        return result
    
    def log_likelihood(self, data, mu, sigma, pi):
        num_samples = data.shape[0]
        num_clusters = self.k
        log_likelihood = 0

        for i in range(num_samples):
            likelihood_sum = 0
            for k in range(num_clusters):
                likelihood_sum += self.multivariate_gaussian(data[i], mu[k], sigma[k], pi[k])
            log_likelihood += np.log(likelihood_sum)

        return log_likelihood
    
    def r_score(self, point, mu, sigma, pi, index):
        num_clusters = self.k
        total_gaussian_sum = 0

        gaussian_value = self.multivariate_gaussian(point, mu[index], sigma[index], pi[index])
        for j in range(num_clusters):
            total_gaussian_sum += self.multivariate_gaussian(point, mu[j], sigma[j], pi[j])
        responsibility = gaussian_value / total_gaussian_sum

        return responsibility
    
    def assign_cluster(self, point, mu, sigma, pi):
        num_clusters = self.k
        max_resp = float('-inf')
        cluster = 0

        for i in range(num_clusters):
            resp = self.r_score(point, mu, sigma, pi, i)
            if resp > max_resp:
                max_resp = resp
                cluster = i
        
        return cluster
    
    def optimize_mu(self, data, mu, sigma, pi):
        num_samples = data.shape[0]
        num_clusters = self.k

        for k in range(num_clusters):
            numerator = 0
            denominator = 0
            for i in range(num_samples):
                resp = self.r_score(data[i], mu, sigma, pi, k)
                numerator += resp * data[i]
                denominator += resp
            mu[k] = numerator / denominator

    def optimize_sigma(self, data, mu, sigma, pi):
        num_samples = data.shape[0]
        num_clusters = self.k

        for k in range(num_clusters):
            numerator = 0
            denominator = 0
            for i in range(num_samples):
                resp = self.r_score(data[i], mu, sigma, pi, k)
                diff = (data[i] - mu[k])[:, np.newaxis]
                numerator += resp * (diff @ diff.T)
                denominator += resp
            sigma[k] = numerator / denominator

    def optimize_pi(self, data, mu, sigma, pi):
        num_samples = data.shape[0]
        num_clusters = self.k
        denominator = num_samples

        for k in range(num_clusters):
            numerator = 0
            for i in range(num_samples):
                numerator += self.r_score(data[i], mu, sigma, pi, k)
            pi[k] = numerator / denominator

def cal_accuracy(label_pre, label_true):
        print('Predicted labels: ', label_pre)
        print('True labels: ', label_true)

        return np.mean(label_pre == label_true)

def iris_classification():
    # Load data from file
    # Make sure that iris.dat is in data/
    train_x, train_y, test_x, test_y = get_iris_data()
    train_y = train_y - 1
    test_y = test_y - 1
    num_class = (np.unique(train_y)).shape[0]

    # Split training data into 3 classes
    train_y_flat = train_y.flatten()
    test_y_flat = test_y.flatten()
    train_x_0 = train_x[train_y_flat == 0]
    train_x_1 = train_x[train_y_flat == 1]
    train_x_2 = train_x[train_y_flat == 2]

    # Create GMM models for each class
    gmm_model_0 = GMM(id=0, k=2)
    gmm_model_1 = GMM(id=1, k=2)
    gmm_model_2 = GMM(id=2, k=2)

    gmm_model_0.fit(train_x_0)
    gmm_model_1.fit(train_x_1)
    gmm_model_2.fit(train_x_2)

    # Predict class for each test sample
    test_labels = []
    for point in test_x:
        resp_0 = np.sum([gmm_model_0.multivariate_gaussian(point, mu, sigma, pi) for mu, sigma, pi in zip(gmm_model_0.mu, gmm_model_0.sigma, gmm_model_0.pi)])
        resp_1 = np.sum([gmm_model_1.multivariate_gaussian(point, mu, sigma, pi) for mu, sigma, pi in zip(gmm_model_1.mu, gmm_model_1.sigma, gmm_model_1.pi)])
        resp_2 = np.sum([gmm_model_2.multivariate_gaussian(point, mu, sigma, pi) for mu, sigma, pi in zip(gmm_model_2.mu, gmm_model_2.sigma, gmm_model_2.pi)])

        max_resp = max(resp_0, resp_1, resp_2)
        if max_resp == resp_0:
            test_labels.append(0)
        elif max_resp == resp_1:
            test_labels.append(1)
        else:
            test_labels.append(2)

    # Calculate accuracy
    test_labels = np.array(test_labels)
    accuracy = cal_accuracy(test_labels, test_y_flat)
    print('Accuracy: ', accuracy)

if __name__ == '__main__':
    iris_classification()