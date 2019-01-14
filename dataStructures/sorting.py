
def bubble_sort(seq):
    if len(seq) <= 1:
        return seq
    n = len(seq)
    for i in range(n-1):
        for j in range(n-1-i):
            if seq[j] > seq[j+1]:
                seq[j], seq[j+1] = seq[j+1], seq[j]
    
def quick_sort(seq):
    if len(seq) <= 1:
        return seq
    else:
        pivot_index = 0
        value = seq[pivot_index]
        less_part = [i for i in seq[pivot_index:] if i <= value]
        larger_part = [i for i in seq[pivot_index:] if i > value]
        return quick_sort(less_part) + [value] + quick_sort(larger_part)

def divide_and_merge_sort(seq):
    if len(seq) <= 1:
        return seq
    else:
