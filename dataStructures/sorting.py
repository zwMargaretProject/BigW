
def bubble_sort(seq):
    n = len(seq)
    for i in range(n-1):
        for j in range(n-1-i):
            if seq[j] > seq[j+1]:
                seq[j], seq[j+1] = seq[j+1], seq[j]


def select_sort(seq):
    n = len(seq)
    for i in range(n-1):
        min_index = i
        for j in range(i+1, n):
            if seq[j] < seq[min_index]:
                min_index = j
        if min_index != i:
            seq[i], seq[min_index] = seq[min_index], seq[i]


def insertion_sort(seq):
    n = len(seq)
    for i in range(1, n):
        value = seq[i]
        pos = i
        while pos > 0 and value < seq[pos-1]:
            seq[pos] = seq[pos-1]
            pos -= 1
        seq[pos] = value


def quick_sort(seq):
    if len(seq) <= 1:
        return seq
    else:
        pivot_index = 0
        value = seq[pivot_index]
        less_part = [i for i in seq[pivot_index:] if i <= value]
        larger_part = [i for i in seq[pivot_index:] if i > value]
        return quick_sort(less_part) + [value] + quick_sort(larger_part)


#### merge sort
def merge_sort(seq):
    if len(seq) <= 1:
        return seq
    else:
        mid = int(len(seq)/2)
        left_half = merge_sort(seq[:mid])
        right_half = merge_sort(seq[mid:])
        
        new_seq = merge_sorted_list(left_half, right_half)
        return new_seq


def merge_sorted_list(sorted_a, sorted_b):
    len_a, len_b = len(sorted_a), len(sorted_b)
    a, b = 0, 0
    new_sorted_seq = []

    while a < len_a and b < len_b:
        if sorted_a[a] < sorted_b[b]:
            new_sorted_seq.append(sorted_a[a])
            a += 1
        else:
            new_sorted_seq.append(sorted_b[b])
            b += 1
    while a < len_a:
        new_sorted_seq.append(sorted_a[a])
        a += 1
    while b < len_b:
        new_sorted_seq.append(sorted_b[b])
        b += 1
    return new_sorted_seq
        
