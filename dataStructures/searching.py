
def binary_search(sorted_array, value):
    if not sorted_array:
        return 
    n = len(sorted_array)
    beg = 0
    end = n - 1
    while beg < end:
        mid = int((beg + end) / 2)
        if sorted_array[mid] == value:
            return mid
        elif sorted_array[mid] > value:
            end = mid - 1
        else:
            beg = mid + 1
    return None
