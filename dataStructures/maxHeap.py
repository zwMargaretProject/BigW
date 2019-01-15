
from collections import Array

class MaxHeap(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._elements = Array(maxsize)
        self._count = 0
    
    def __len__(self):
        return self._count
    
    def add(self, value):
        if self.maxsize <= len(self):
            raise Exception('Max heap is full.')
        
        self._elements[self._count] = value
        self._count += 1
        self._shiftup(self._count - 1)
    
    def _shiftup(self, index):
        if index > 0:
            parent = int((index - 1) / 2)

            if self._elements[index] > self._elements[parent]:
                self._elements[index], self._elements[parent] = self._elements[parent], self._elements[index]
                self._shiftup(parent)
    
    def extract(self):
        if self._count <= 0:
            raise Exception('Empty')
        
        value = self._elements[0]
        self._count -= 1
        self._elements[0] = self._elements[self._count]
        self._shiftdown(0)
        return value
    
    def _shiftdown(self, index):
        left = 2 * index + 1
        right = 2 * index + 2
        largest = index 

        if (left < self._count and self._elements[left] >= self._elements[largest] and self._elements[left] > self._elements[right]):
            largest = left
        
        elif right < self._count and self._elements[right] >= self._elements[largest]:
            largest = right
        
        if largest != index:
            self._elements[index], self._elements[largest] = self._elements[largest], self._elements[index]
            self._shiftdown(largest)
