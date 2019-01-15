
from hashTable import HashTable

class setADT(object):
    def add(self, key):
        return super(setADT, self).add(key, True)
    
    def __and__(self, other_set):
        new_set = setADT()
        for element_a in self:
            if element_a in other_set:
                new_set.add(element_a)
        return new_set
    
    def __sub__(self, other_set):
        new_set = setADT()
        for element_a in self:
            if element_a not in other_set:
                new_set.add(element_a)
        return new_set
    
    def __or__(self, other_set):
        new_set = setADT()
        for element_a in self:
            new_set.add(element_a)
        # HashTable can automatically deal with repeated numbers.
        for element_b in other_set:
            new_set.add(element_b)
        return new_set