import numpy as np
from data_util import process_input_1, load_iris_data, kfold_split

NUM_SPLIT = 5
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
        sigma = []
        for i in range(num_clusters):
            sigma.append(np.zeros((num_features, num_features)))
        resp_maxtrix = np.zeros((num_samples, num_clusters))
        total_gamma = np.zeros(num_clusters)

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
            # E-step: Evaluate the responsibilities
            for i in range(num_samples):
                for k in range(num_clusters):
                    resp_maxtrix[i][k] = self.r_score(data[i], mu, sigma, pi, k)

            # M-step: Update the parameters
            total_gamma = np.sum(resp_maxtrix, axis=0)
            for k in range(num_clusters):
                N_k = total_gamma[k]
                sum_mu = 0
                sum_sigma = 0
                for i in range(num_samples):
                    sum_mu += resp_maxtrix[i][k] * data[i]
                mu[k] = sum_mu / N_k

                for i in range(num_samples):
                    diff = (data[i] - mu[k])[:, np.newaxis]
                    sum_sigma += resp_maxtrix[i][k] * (diff @ diff.T)
                sigma[k] = sum_sigma / N_k

                pi[k] = N_k / num_samples

            log_l_post = self.log_likelihood(data, mu, sigma, pi)
            # print(self.id, '- Log likelihood: ', log_l_post)

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
        reg_val = 1e-6

        norm_factor = (2 * np.pi) ** (-d / 2)
        det_sigma = np.linalg.det(sigma + reg_val * np.eye(d)) ** (-1 / 2)
        exponent = -0.5 * diff.T @ np.linalg.inv(sigma + reg_val * np.eye(d)) @ diff
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

def calculate_confusion_matrix(label_pre, label_true, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(label_true, label_pre):
        confusion_matrix[true][pred] += 1
    
    return confusion_matrix

def cal_accuracy(label_pre, label_true, num_classes):
        confusion_matrix = calculate_confusion_matrix(label_pre, label_true, num_classes)
        print('Confusion matrix: ')
        print(confusion_matrix)

        return np.mean(label_pre == label_true)

def iris_classification():
    # Load iris data
    X, Y, num_classes = load_iris_data()
    total_accuracy = 0

    # K-fold cross validation
    for train_x, train_y, test_x, test_y in kfold_split(X, Y, NUM_SPLIT):
        # Create GMM models for each class
        gmm_models = []
        for clss in range(num_classes):
            indices = np.where(train_y == clss)
            gmm_model = GMM(id=clss, k=NUM_CLUSTER)
            gmm_model.fit(train_x[indices])
            gmm_models.append(gmm_model)

        # Predict class for each test sample
        test_labels = []
        for point in test_x:
            resp_values = []
            for gmm in gmm_models:
                resp_sum = 0
                for k in range(gmm.k):
                    resp_sum += gmm.multivariate_gaussian(point, gmm.mu[k], gmm.sigma[k], gmm.pi[k])
                resp_values.append(resp_sum)
            
            max_index = np.argmax(resp_values)
            test_labels.append(max_index)

        # Calculate accuracy
        test_labels = np.array(test_labels)
        accuracy = cal_accuracy(test_labels, test_y, num_classes)
        print(f'Accuracy of this fold: {accuracy * 100:.6f}%')

        total_accuracy += accuracy
    print(f'Average accuracy: {total_accuracy / NUM_SPLIT * 100:.6f}%')

if __name__ == '__main__':
    iris_classification()