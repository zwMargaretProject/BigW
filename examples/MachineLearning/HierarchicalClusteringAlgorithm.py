# Python code for question C201 and HFC 2017

import pandas as pd
import numpy as np

class Node(object):
    def __init__(self,index_list,position=None,count=1):
        self.count = count
        self.index_list = index_list  
        self.position = position

class EqualSizedClustering(object):
    def __init__(self,corr_matrix,k,node_size=None):
        self.k = k
        self.corr_matrix = corr_matrix
        self._len = np.shape(self.corr_matrix)[0]
        if node_size is None:
            self.node_size = int(self._len/self.k)
        else:
            self.node_size = node_size
        self.nodes = []

        self.all_index = [i for i in range(self._len)]
        self.df = None

    def initial_fit(self):
        nodes = []
        for i in self.all_index:
            nodes.append(Node(index_list=[i],position=i,count=1))
        position = -1
        distances = {}

        new_nodes = []

        while True:
            max_dist = 0
            nodes_len = len(nodes)

            for i in range(nodes_len-1):
                for j in range(i+1,nodes_len):
                    node_a,node_b = nodes[i],nodes[j]
                    position_pair = (node_a.position,node_b.position)
                    if position_pair not in distances:
                        distances[position_pair] = self.node_distance(node_a,node_b)
                    d = distances[position_pair]
                    if max_dist < d:
                        max_dist = d
                        max_i = i
                        max_j = j
                        max_node_a = node_a
                        max_node_b = node_b
                        
            node_a = max_node_a
            node_b = max_node_b

            new_index_list = node_a.index_list + node_b.index_list
            new_count = node_a.count + node_b.count
            new_node = Node(index_list=new_index_list,count=new_count,position=position) 
            if len(nodes) > 2:
                nodes.pop(max_j)
                nodes.pop(max_i)
            else:
                node_a = nodes[0]
                node_b = nodes[1]
                new_index_list = node_a.index_list + node_b.index_list
                new_count = node_a.count + node_b.count
                new_node = Node(index_list=new_index_list,count=new_count,position=position) 
                new_nodes.append(new_node)
                break
            
            if new_node.count >= self.node_size:
                new_nodes.append(new_node)
            else:             
                nodes.append(new_node)
            position -= 1

            if len(nodes)+len(new_nodes) == self.k :
                if len(nodes) > 0:
                    new_nodes.extend(nodes)
                break

            
        self.nodes = new_nodes

    def adjust_size(self):
        exact_size = [node for node in self.nodes if node.count == self.node_size]
        large_size = [node for node in self.nodes if node.count > self.node_size]
        small_size = [node for node in self.nodes if node.count < self.node_size]
        from_large_to_small = []
        if large_size != []:
            for node in large_size:
                n = node.count - self.node_size
                remove_index = self.remove_smallest_index(node,n,
                                                          distance=False,all_index=self.all_index)
                #remove_index = self.remove_smallest_index(node,n)
                from_large_to_small.extend(remove_index)
                for index in remove_index:
                    node.index_list.remove(index)
                exact_size.append(node)             
            for node in small_size:
                n = self.node_size - node.count
                temp_all_index = [index for index in self.all_index if index not in from_large_to_small]
                add_index = self.add_largest_index(node,from_large_to_small,n,distance=False,all_index=temp_all_index)
                
                #add_index = self.add_largest_index(node,from_large_to_small,n)
                node.index_list.extend(add_index)
                exact_size.append(node)
                for index in add_index:
                    from_large_to_small.remove(index)
        self.nodes = exact_size

        
    def setDataDf(self,df):
        self.df = df   
        
    def ICS(self,node):
        node_corr_matrix = self.corr_matrix[node.index_list][:,node.index_list]
        return node_corr_matrix.sum()

    def TICS(self):
        TICS = 0
        for node in self.nodes:
            TICS += self.ICS(node)
        return TICS

    def AR(self,node):
        temp = self.df[node.index_list]
        return temp.mean(axis=1)
    
    def CARS(self):
        output = pd.DataFrame(index=self.df.index)
        id = 0
        for node in self.nodes:
            id += 1
            output[id] = self.AR(node)
        temp = np.array(output.corr())
        cars = temp.sum()
        return cars
        
    def node_distance(self,node_a,node_b):
        temp = self.corr_matrix[node_b.index_list][:,node_a.index_list]        
        return temp.mean()


    def output_results(self,output_filename):
        label = 0
        index_list,label_list = [],[]
        for node in self.nodes:
            label += 1
            for index in node.index_list:
                index_list.append(index)
                label_list.append(label)
        output = pd.DataFrame({'Stock':pd.Series(index_list),'Label':pd.Series(label_list)})
        output.to_csv(output_filename)
        
    def current_target_value(self):
        cars = self.CARS()
        tics = self.TICS()
        value = cars/tics
        return cars,tics,value
    
    def remove_smallest_index(self,node,remove_num,distance=True,all_index=None):
        index_list = node.index_list
        remove_index = []
        contributions = np.zeros(len(index_list))
        for pos in range(len(index_list)):
            index = index_list[pos]
            temp_node = Node(index_list=[index])
            if distance:
                contributions[pos]= self.node_distance(node,temp_node)
            else:
                contributions[pos] = self.relative_contributions(node,index,all_index)
        for select_time in range(remove_num):
            pos = np.where(contributions==contributions.min())[0][0]
            remove_index.append(index_list[pos])
            contributions[pos] = 1000
        return remove_index
    
    def deleteIndex(self,del_index,index_list):
        new_list = []

        for index in index_list:
            if index != del_index:
                new_list.append(index)
        return new_list
    
    def relative_contributions(self,node,index,all_index):
        index_list = self.deleteIndex(index,node.index_list)
        return self.corr_matrix[[index]][:,index_list].sum()/self.corr_matrix[[index]][:,all_index].sum()
        
    def add_largest_index(self,node,add_index_candidate,add_num,distance=True,all_index=None):
        add_index = []
        contributions = np.zeros(len(add_index_candidate)) 
        for pos in range(len(add_index_candidate)):
            index = add_index_candidate[pos]
            temp_node = Node(index_list=[index])
            if distance:
                contributions[pos]= self.node_distance(node,temp_node)
            else:
                contributions[pos] = self.relative_contributions(node,index,all_index)
        for select_time in range(add_num):
            pos = np.where(contributions==contributions.max())[0][0]
            add_index.append(add_index_candidate[pos])
            contributions[pos] = -1000
        return add_index
    
def test_c201():
    corr = pd.read_csv('D:/c102/c102_corr.csv')
    corr_matrix = np.array(corr)
    print corr_matrix.shape
    eqc = EqualSizedClustering(corr_matrix=corr_matrix,k=4,node_size=125)
    eqc.initial_fit()
    
    df = pd.read_csv('D:/c102/c102_data.csv')
    df = df.set_index('Date')

    df.columns = [i for i in range(503)]
    
    eqc.setDataDf(df)
    print eqc.current_target_value()
    eqc.output_results('D:/c102/c102_labels.csv')
    

def test_HFC():
    df = pd.read_csv('D:/54_hfc_20170614_comp.csv')
    df.columns = [i for i in range(2000)]
    corr = data.corr()
    corr_matrix = np.array(corr)
    eqc = EqualSizedClustering(corr_matrix=corr_matrix,k=10)
    eqc.initial_fit()   
    eqc.setDataDf(df)
    print eqc.current_target_value()
    eqc.adjust_size()
    print eqc.current_target_value() 
    eqc.output_results('D:/labels.csv')
       
if __name__== '__main__':
    test_c201()
    test_HFC()

