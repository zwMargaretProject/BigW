
class BinarySearchTreeNode(object):
    def __init__(self, key, value, left=None, right=None):
        self.key, self.value = key, value
        self.left, self.right = left, right


class BinarySearchTree(object):
    def __init__(self, root=None):
        self.root = root
    
    @classmethod
    def build_from(cls, node_list):
        cls.size = 0
        key_to_node_dict = {}
        
        for node_dict in node_list:
            key = node_dict['key']
            key_to_node_dict[key] = BinarySearchTreeNode(key=key, value=key)
        
        for node_dict in node_list:
            key = node_dict['key']
            node = key_to_node_dict[key]

            if node_dict['is_root'] is True:
                root = node
            
            node.left = key_to_node_dict.get(node_dict['left'])
            node.right = key_to_node_dict.get(node_dict['right'])
            cls.size += 1
        return cls(root)
    
    def _bst_search(self, subtree, key):
        if subtree is None:
            return None
        elif key < subtree.key:
            return self._bst_search(subtree.left, key)
        elif key > subtree.right:
            return self._bst_search(subtree.right, key)
        else:
            return subtree
    
    def get(self, key, default=None):
        node = self._bst_search(self.root, key)
        if node is None:
            return default
        return node.value
    
    def __contains__(self, key):
        return self._bst_search(self.root, key) is not None
    
    def _bst_min_node(self, subtree):
        if subtree is None:
            return None
        elif subtree.left is None:
            return subtree
        else:
            return self._bst_min_node(subtree.left)
    
    def bst_min(self):
        node = self._bst_min_node(self.root)
        return node.value if node else None
    
    def _bst_insert(self, subtree, key, value):
        if subtree is None:
            subtree = BinarySearchTreeNode(key, value)
        elif key < subtree.key:
            subtree.left = self._bst_insert(subtree.left, key, value)
        elif key > subtree.key:
            subtree.right = self._bst_insert(subtree.right, key, value)
        return subtree
    
    def add(self, key, value):
        node = self._bst_search(self, key, value)
        if node is not None:
            node.value = value
            return False
        else:
            self.root = self._bst_insert(self.root, key, value)
            self.size += 1
            return True
    
    def _bst_remove(self, subtree, key):
        if subtree is None:
            return None
        elif key < subtree.key:
            subtree.left = self._bst_remove(subtree.left, key)
            return subtree
        elif key > subtree.key:
            subtree.right = self._bst_remove(subtree.right, key)
            return subtree
        else:
            if subtree.left is None and subtree.right is None:
                return None
            elif subtree.left is None or subtree.right is None:
                if subtree.left is None:
                    return subtree.left
                else:
                    return subtree.right
            else:
                successor_node = self._bst_min_node(subtree.right)
                subtree.key, subtree.value = successor_node.key, subtree.value
                subtree.right = self._bst_remove(subtree.right, successor_node.key)
                return subtree
    
    def remove(self.key):
        assert key in self
        self.size -= 1
        return self._bst_remove(self.root, key)
        
