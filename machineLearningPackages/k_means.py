
import math
import random
import numpy as np
import collections

def euler_distance(point1: list, point2: list) -> float:
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


class K_means(object):
    def __init__(self, k: int, max_iter=10):
        self.k = k
        self.max_iter = 10      
        self.data_set = None   
        self.labels = None      

    def init_centroids(self) -> list:
        point_num = np.shape(self.data_set)[0]
        random_index = random.sample(list(range(point_num)), self.k)
        centroids = [self.data_set[i] for i in random_index]
        return centroids

    def fit(self, data_set):
        self.data_set = data_set
        point_num = np.shape(data_set)[0]
        self.labels = [ -1 ] * point_num            
        centroids = self.init_centroids()           
        old_centroids = []                         
        step = 0                                   
        while not self.should_stop(old_centroids, centroids, step):
            old_centroids = np.copy(centroids)
            step += 1
            for i, point in enumerate(data_set):
                self.labels[i] = self.get_closest_index(point, centroids)
            centroids = self.update_centroids()

    def get_closest_index(self, point, centroids):
        min_dist = math.inf 
        label = -1
        for i, centroid in enumerate(centroids):
            dist = euler_distance(centroid, point)
            if dist < min_dist:
                min_dist = dist
                label = i
        return label

    def update_centroids(self):
        collect = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            collect[label].append(self.data_set[i])

        centroids = []
        for i in range(self.k):
            centroids.append(np.mean(collect[i], axis=0))
        return centroids

    def should_stop(self, old_centroids, centroids, step) -> bool:

        if step > self.max_iter:
            return True
        return np.array_equal(old_centroids, centroids)


def compare_test():
	from sklearn import cluster, datasets
	iris = datasets.load_iris()
	my_k = K_means(4)
	my_k.fit(iris.data)
	print(np.array(my_k.labels))

	sk = cluster.KMeans(4)
	sk.fit(iris.data)
	print(sk.labels_)


if __name__ == '__main__':
	compare_test()