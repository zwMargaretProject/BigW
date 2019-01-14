from collections import Array

class Slot(object):
    def __init__(self, key, value):
        self.key, self.value = key, value
    

class HashTable(object):
    UNUSED = None
    EMPTY = Slot(None, None)

    def __init__(self):
        self._table = Array(8, init=HashTable.UNUSED)
        self.length = 0
    
    @property
    def _load_factor(self):
        return self.length / (len(self._table))
    
    def __len__(self):
        return self.length
    
    def _hash(self, key):
        return abs(hash(key) % len(self._table))
    
    def _find_key(self, key):
        index = self._hash(key)
        _len = len(self._table)

        while self._table[index] is not HashTable.UNUSED:
            if self._table[index] is HashTable.EMPTY:
                index = (index * 5 + 1) % _len
                continue
            
            elif self._table[index].key == key:
                return index
            
            else:
                index = (index * 5 + 1) % _len

        return None
    
    def _find_slot_to_insert(self, key):
        index = self._hash(key)
        _len = len(self._table)

        while not self._slot_can_be_inserted(index):
            index = (index * 5 + 1) % _len
        return index
    
    def _slot_can_be_inserted(self, index):
        return (self._table[index] is HashTable.EMPTY or self._table[index] is HashTable.UNUSED)
    
    def __contains__(self, key):
        index = self._find_key(key)
        return index is not None
    
    def add(self, key, value):
        if key in self:
            index = self._find_key(key)
            self._table[index] = value
            return False
        else:
            index = self._find_slot_to_insert(key)
            self._table[index] = Slot(key, value)
            self.length += 1
            if self._load_factor >= 0.8:
                self._rehash()
            return True
    
    def _rehash(self):
        old_table = self._table
        newsize = len(old_table) * 2
        self._table = Array(newsize, HashTable.UNUSED)
        self.length = 0

        for slot in old_table:
            if slot is not HashTable.UNUSED and slot is not HashTable.EMPTY:
                index = self._find_slot_to_insert(slot.key)
                self._table[index] = slot.value
                self.length += 1
    
    def get(self, key, default=None):
        index = self._find_key(key)
        if index is None:
            return default
        return self._table[index].value
    
    def remove(self, key):
        index = self._find_key(key)
        if index is None:
            raise KeyError()
        value = self._table[index].value
        self.length -= 1
        self._table[index] = HashTable.EMPTY
        return value
    
    def __iter__(self):
        for slot in self._table:
            if slot not in (HashTable.EMPTY, HashTable.UNUSED):
                yield slot
