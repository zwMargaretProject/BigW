from collections import LinkedList

class FullError(Exception):
    pass

class EmptyError(Exception):
    pass

class Queue(object):
    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        self._item_linked_list = LinkedList()
    
    def __len__(self):
        return len(self._item_linked_list)
    
    def push(self, value):
        if self.maxsize is not None and self.maxsize <= len(self):
            raise FullError("Queue is full with max size {}".format(self.maxsize))
        self._item_linked_list.append(value)
    
    def pop(self):
        if len(self) == 0:
            raise EmptyError("Queue is empty")
        return self._item_linked_list.popleft()


class ArrayQueue(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.array = Array(maxsize)
        self.head = 0
        self.tail = 0
    
    def __len__(self):
        return self.head

    def push(self, value):
        if len(self) >= self.maxsize:
            raise FullError("Queue is full with max size {}".format(self.maxsize))
        self.array[self.head] = value
        self.head += 1
    
    def pop(self):
        if len(self) == 0:
            raise EmptyError("Queue is empty")
        value = self.array[self.tail % self.maxsize]
        self.tail += 1
    