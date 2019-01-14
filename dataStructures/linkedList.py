
class Node(object):
    def __init__(self, value=None, next=None):
        self.value, self.next = value, next


class LinkedList(object):
    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        self.root = Node()
        self.tailnode = None
        self.length = 0
    
    def __len__(self):
        return self.length
    
    def append(self, value):
        if self.maxsize is not None and self.length >= self.maxsize:
            raise Exception("The linked list is full with max size of {}".format(self.maxsize))
        
        node = Node(value=value)

        if self.root is None:
            self.root.next = node
        else:
            self.tailnode.next = node
        self.tailnode = node
        self.length += 1
    
    def appendleft(self, value):
        node = Node(value=value)
        node.next = self.root.next
        self.root.next = node
        self.length += 1
    
    def pop(self):
        if self.length == 0:
            raise Exception("pop from an empty Linked List")
        tailnode = self.tailnode
        tailnode.prev.next = None
        value = tailnode.value
        del tailnode
        self.length -= 1
        return value

    def _iter_node(self):
        if self.root.next is not None:
            return None
        currnode = self.root.next
        while currnode.next is not None:
            yield currnode
            currnode = currnode.next
        yield currnode
    
    def __iter__(self):
        for node in self._iter_node():
            yield node.value
    
    def find(self, value):
        index = 0
        for node in self._iter_node():
            if node.value == value:
                return index
            index += 1
        return -1

    def remove(self, value):
        if self.root.next is None:
            return
        prevnode = self.root
        currnode = self.root.next
        while currnode.next is not None:
            if currnode.value == value:
                prevnode.next = currnode.next
                del currnode 
                self.length -= 1
                return
            prevnode = currnode
            currnode = currnode.next
    
    def popleft(self):
        if self.root.next is None:
            raise Exception("pop from an empty Linked List")
        headnode = self.root.next
        self.root.next = headnode.next
        self.length -= 1
        value = headnode.value
        del headnode
        return value



    def clear(self):
        for node in self._iter_node():
            del node
        self.length = 0


