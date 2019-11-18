import sys

n = int(sys.argv[1])
m = 0
insert_hash = [1] + [0]*n + [1]
close_hash = [0] * (n+1)

def dfs(k):
    global n, m, insert_hash, close_hash

    if k == 2*n+1:
        m += 1
        return

    for i in range(1, n+1):
        if not insert_hash[i]:
            insert_hash[i] = 1
            dfs(k+1)
            insert_hash[i] = 0

    for i in range(n+1):
        if (not close_hash[i]) and insert_hash[i] and insert_hash[i+1]:
            close_hash[i] = 1
            dfs(k+1)
            close_hash[i] = 0

dfs(0)
print(m)
