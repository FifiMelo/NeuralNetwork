import time
def add_dots(start, stop):
    if stop > start + 1:
        A = start + (stop - start)//3
        B = start + 2 * (stop - start)//3
        add_dots(start, A)
        t.append(A)
        t.append(B)
        add_dots(B, stop)
start_time = time.time()
N = 23
t = [0]
add_dots(0,3**N)
t.append(3**N)
print(time.time() - start_time)