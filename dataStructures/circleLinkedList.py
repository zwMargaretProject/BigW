
class Node(object):
    def __init__(self, value=None, prev=None, next=None):
        self.value, self.prev, self.next = value, prev, next

class CircleLinkedList(object):
    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        node = Node()
        node.prev, node.next = node
        self.root = node
        self.length = 0
    
    def headnode(self):
        return self.root.next
    
    def tailnode(self):
        return self.root.prev
    
    def __len__(self):
        return self.length
    
    def append(self, value):
        if self.maxsize is not None and self.length >= self.maxsize:
            raise Exception("Circle Linked List is full with max size {}".format(self.maxsize))
        tailnode = self.tailnode()
        node = Node(value=value)
        tailnode.next = node
        node.prev = tailnode
        node.next = self.root
        self.root.prev = node
        self.length += 1
    
    def appendleft(self, value):
        if self.maxsize is not None and self.length >= self.maxsize:
            raise Exception("Circle Linked List is full with max size {}".format(self.maxsize))
        node = Node(value=value)
        if self.root.next is self.root:
            node.prev = self.root
            node.next = self.root
            self.root.next = node
            self.root.prev = node
        else:
            headnode = self.headnode()
            self.root.next = node
            node.prev = self.root
            node.next = headnode
            headnode.prev = node
        self.length += 1
    
    def _iter_node(self):
        if self.root.next is self.root:
            return
        currnode = self.headnode()
        while currnode.next is not self.root:
            yield currnode
            currnode = currnode.next
        yield currnode
    
    def __iter__(self):
        for node in self._iter_node():
            yield node.value
    
    def _reverse_iter_node(self):
        if self.root.prev is self.root:
            return
        currnode = self.tailnode()
        while currnode.prev is not self.root:
            yield currnode
            currnode = currnode.prev
        yield currnode
    
    def __reverse__(self):
        for node in self._reverse_iter_node():
            yield node.value
    
    def pop(self):
        if self.length == 0:
            return
        tailnode = self.tailnode()
        tailnode.prev.next = self.root
        self.root.prev = tailnode.prev
        del tailnode
        self.length -= 1
        return tailnode.value
    
    def popleft(self):
        if self.length == 0:
            return
        headnode = self.headnode()
        self.root.next = headnode.next
        headnode.next.prev = self.root
        del headnode
        self.length -= 1
        return headnode.value
        

            

            
            
