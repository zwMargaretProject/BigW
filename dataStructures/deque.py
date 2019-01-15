from circalDoubleLinkedList import CircalDoubleLinkedList

class Deque(CircalDoubleLinkedList):
    def pop(self):
        if len(self) == 0:
            raise Exception('Empty')
        tailnode = self.tailnode()
        value = tailnode.value
        self.remove(tailnode)
        return value
    
    def popleft(self):
        if len(self) == 0:
            raise Exception('Empty')
        headnode = self.headnode()
        value = headnode.value
        self.remove(headnode)
        return value