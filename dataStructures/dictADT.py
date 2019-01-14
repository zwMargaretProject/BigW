from hashTable import HashTable

class DictADT(HashTable):
    def __setitem__(self, key, value):
        self.add(key, value)
    
    def __getitem__(self, key):
        if key not in self:
            raise KeyError()
        else:
            return self.get(key)
    
    def _iter_slot(self):
        for slot in self._table:
            if slot not in (HashTable.UNUSED, HashTable.EMPTY):
                yield slot
    
    def items(self):
        for slot in self._iter_slot():
            yield (slot.key, slot.value)
    
    def keys(self):
        for slot in self._iter_slot():
            yield slot.key
    
    def values(self):
        for slot in self._iter_slot():
            yield slot.value
