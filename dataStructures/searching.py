
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


def linear_search(iterable, target):
    for index, value in enumerate(iterable):
        if value == target:
            return index
    return None


def linear_search_2(iterable, predicate=lambda x: x % 8):
    for index, value in enumerate(iterable):
        if predicate(value):
            return index
    return None


def linear_search_recursive(iterable, target):
    if len(iterable) == 0:
        return None

    index = len(iterable) - 1
    if iterable[index] == target:
        return index
    else:
        return linear_search_recursive(iterable[:index], target)


