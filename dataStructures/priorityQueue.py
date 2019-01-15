from maxHeap import MaxHeap

class PriorityQueue(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._maxheap = MaxHeap(maxsize)
    
    def push(self, priority, value):
        entry = (priority, value)
        self._maxheap.add(entry)
    
    def pop(self, with_priority=False):
        entry = self._maxheap.extract()
        if with_priority:
            return entry
        else:
            return entry[1]
    
    def is_empty(self):
        return len(self._maxheap) == 0