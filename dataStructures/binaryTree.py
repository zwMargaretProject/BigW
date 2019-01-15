
class BinaryTreeNode(object):
    def __init__(self, data=None, left=None, right=None):
        self.data, self.left, self.right = data, left, right


class BinaryTree(object):
    def __init__(self, root=None):
        self.root = root
    
    @classmethod
    def build_from(cls, node_list):
        node_dict = {}
        for node_data in node_list:
            data = node_data['data']
            node_dict[data] = BinaryTreeNode(data=data)
        
        for node_data in node_list:
            data = node_data['data']
            node = node_dict[data]
            if node_data['is_root'] is True:
                root = node
            node.left = node_dict.get(node_data['left'])
            node.right = node_dict.get(node_data['right'])
        
        return cls(root)
    
    def preorder_trav(self, subtree):
        if subtree is not None:
            print(subtree.data)
            preorder_trav(subtree.left)
            preorder_trav(subtree.right)

    def reverse(self, subtree):
        if subtree is not None:
            subtree.left, subtree.right = subtree.right, subtree.left
            reverse(subtree.left)
            reverse(subtree.right)


def main():
    node_list = [{'data':1, 'left':2, 'right':3, 'is_root':True},
                 {'data':2, 'left':4, 'right':5, 'is_root':False},
                 {'data':3, 'left':6, 'right':7, 'is_root':False}]
    binary_tree = BinaryTree.build_from(node_list)
    binary_tree.preorder_trav(self, binary_tree.root)
    binary_tree.reverse(self, binary_tree.root)


if __name__ =='__main__':
    main()
