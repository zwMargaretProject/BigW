import math
import collections
import numpy as np 

def euler_distance(point1: list, point2: list):
	'''
	Calculate relative distance of two points.
	'''
	distance = 0.0
	for x, y in zip(point1, point2):
		distance += math.pow(a - b, 2)
	return math.sqrt(distance)


class KNeighborsClass(object):
	def __init__(self, n_neighbors=5):
		self.n_neighbors = n_neighbors

	def fit(self, data_set, labels):
		self.data_set = data_set
		self.labels = labels

	def predict(self, test_row):
        dist = []
        for v in self.data_set:
            dist.append(euler_distance(v, test_row))
        dist = np.array(dist)
        sorted_dist_index = np.argsort(dist) 

        class_list = [ self.labels[sorted_dist_index[i] ] for i in range(self.n_neighbors)]
        result_dict = collections.Counter(class_list)   
        ret = sorted(result_dict.items(), key=lambda x: x[1], reverse=True) 
        return ret[0][0]


#################################

def direct_use():
    from sklearn import datasets
    iris = datasets.load_iris()
    knn = KNeighborsClass(n_neighbors=5)
    knn.fit(iris.data, iris.target)
    predict = knn.predict([0.1, 0.2, 0.3, 0.4])
    print(predict)  

def import_external_package():
	from sklearn import datasets
	from sklearn import neighbors

	iris = datasets.load_iris()
	print(iris.shape)   # (n_samples, n_features)

	knn = neighbors.KNeighborsClass(n_neighbors=5)
	knn.fit(iris.data, iris.target)
	predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
	print(predict)    # output: [0]


if __name__ == '__main__':
	direct_use()
	import_external_package()
