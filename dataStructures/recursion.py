
# 1
def fact(n):
    if n == 0:
        return 1
    else:
        return n * fact(n-1)


# 2
# Use stack to simulate recursion.

from stack import Stack

def print_num_use_stack(n):
    s = Stack()
    while n > 0:
        s.push(n)
        n -= 1
    while not s.is_empty():
        print(s.pop())