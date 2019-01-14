import math
import collections

def entropy(rows: list) -> float:
    """
    计算数组的熵
    """
    result = collections.Counter()
    result.update(rows)
    rows_len = len(rows)
    assert rows_len   # 数组长度不能为0
    # 开始计算熵值
    ent = 0.0
    for r in result.values():
        p = float(r) / rows_len
        ent -= p * math.log2(p)
    return ent

def condition_entropy(future_list: list, result_list: list) -> float:
    """
    计算条件熵
    """
    entropy_dict = collections.defaultdict(list)  # {0:[], 1:[]}
    for future, value in zip(future_list, result_list):
        entropy_dict[future].append(value)
    # 计算条件熵
    ent = 0.0
    future_len = len(future_list)  # 数据个数
    for value in entropy_dict.values():
        p = len(value) / future_len * entropy(value)
        ent += p

    return ent

def gain(future_list: list, result_list: list) -> float:
    """
    获取某特征的信息增益
    """
    info = entropy(result_list)
    info_condition = condition_entropy(future_list, result_list)
    return info - info_condition

def if_split_end(result_list: list) -> bool:
    """
    递归的结束条件，每个分支的结果集都是相同的分类
    """
    result = collections.Counter(result_list)
    return len(result) == 1

def choose_best_future(data_set: list, labels: list, ignore_index: list) -> int:
	"""
	从特征向量中筛选出最好的特征，返回它的特征索引
	"""
	result_dict = {}  # { 索引: 信息增益值 }
	future_num = len(data_set[0])
	for i in range(future_num):
	    if i in ignore_index: # 如果已经计算过了
	        continue
	    future_list = [x[i] for x in data_set]
	    result_dict[i] = gain(future_list, labels) # 获取信息增益
	# 排序后选择第一个
	ret = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
	return ret[0][0]

class DecisionNode(object):
    def __init__(self, col=-1, data_set=None, labels=None, results=None, tb=None, fb=None):
        self.has_calc_index = []    
        self.col = col              
        self.data_set = data_set    
        self.labels = labels        
        self.results = results      
        self.tb = tb               
        self.fb = fb               


class DecisionTreeClass():
    def __init__(self):
        self.future_num = 0      # 特征
        self.tree_root = None    # 决策树根节点

    def build_tree(self, node: DecisionNode):
	    # 递归条件结束
	    if if_split_end(node.labels):
	        node.results = node.labels[0] # 表明是叶子节点
	        return
	    #print(node.data_set)
	    # 不是叶子节点，开始创建分支
	    best_index = choose_best_future(node.data_set, node.labels, node.has_calc_index)
	    node.col = best_index

	    # 根据信息增益最大进行划分
	    # 左子树
	    tb_index = [i for i, value in enumerate(node.data_set) if value[best_index]]
	    tb_data_set     = [node.data_set[x] for x in tb_index]
	    tb_data_labels  = [node.labels[x] for x in tb_index]
	    tb_node = DecisionNode(data_set=tb_data_set, labels=tb_data_labels)
	    tb_node.has_calc_index = list(node.has_calc_index)
	    tb_node.has_calc_index.append(best_index)
	    node.tb = tb_node

	    # 右子树
	    fb_index = [i for i, value in enumerate(node.data_set) if not value[best_index]]
	    fb_data_set = [node.data_set[x] for x in fb_index]
	    fb_data_labels = [node.labels[x] for x in fb_index]
	    fb_node = DecisionNode(data_set=fb_data_set, labels=fb_data_labels)
	    fb_node.has_calc_index = list(node.has_calc_index)
	    fb_node.has_calc_index.append(best_index)
	    node.fb = fb_node

	    # 递归创建子树
	    if tb_index:
	        self.build_tree(node.tb)
	    if fb_index:
	        self.build_tree(node.fb)



    def fit(self, x: list, y: list):
	    """
	    x是训练集，二维数组。y是结果集，一维数组
	    """
	    self.future_num = len(x[0])
	    self.tree_root = DecisionNode(data_set=x, labels=y)
	    self.build_tree(self.tree_root)
	    self.clear_tree_example_data(self.tree_root)


	def clear_tree_example_data(self, node: DecisionNode):
	    """
	    清理tree的训练数据
	    """
	    del node.has_calc_index
	    del node.labels
	    del node.data_set
	    if node.tb:
	        self.clear_tree_example_data(node.tb)
	    if node.fb:
	        self.clear_tree_example_data(node.fb)



	def _predict(self, data_test: list, node: DecisionNode):
	    if node.results:
	        return node.results
	    col = node.col
	    if data_test[col]:
	        return self._predict(data_test, node.tb)
	    else:
	        return self._predict(data_test, node.fb)

	def predict(self, data_test):
	    """
	    预测
	    """
	    return self._predict(data_test, self.tree_root)

def internal_test:
    dummy_x = [
        [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, ],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, ],
        [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, ],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, ],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, ],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, ],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, ],
        [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, ],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, ],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, ],
        [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, ],
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, ],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, ],
        [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, ],
    ]
    dummy_y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

    tree = DecisionTreeClass()
    tree.fit(dummy_x, dummy_y)

    test_row = [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, ]
    print(tree.predict(test_row))  # output: 1


def external_package():
	# coding: utf-8 import csv
	from sklearn.feature_extraction import DictVectorizer
	from sklearn import preprocessing
	from sklearn import tree

	data_file = open("computer_buy.csv", "r")
	reader = csv.reader(data_file)
	headers = next(reader)

	future_list = []
	label_list = []

	for row in reader:
	    label_list.append(row[-1])
	    row_dict = {}
	    for i in range(1, len(row) -1):
	        row_dict[ headers[i] ] = row[i]
	    future_list.append(row_dict)
	data_file.close()

	# 向量化 x
	vec = DictVectorizer()
	dummy_x = vec.fit_transform(future_list).toarray()

	print("dummy_x:", dummy_x)
	print("vec.get_feature_names()", vec.get_feature_names())

	# 向量化 y
	lb = preprocessing.LabelBinarizer()
	dummy_y = lb.fit_transform(label_list)

	# 构造决策树
	clf = tree.DecisionTreeClassifier(criterion='entropy')
	clf.fit(dummy_x, dummy_y)

	print("clf: ", clf)

	# 保存模型 with open("result.dot", "w") as f:
	    tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

	# 测试数据
	first_row = dummy_x[0, :]
	new_row = list(first_row)
	new_row[0] = 1
	new_row[2] = 0

	predict = clf.predict([new_row])
	print("predict:", predict)    



if __name__ == '__main':
	internal_test()
	external_package()