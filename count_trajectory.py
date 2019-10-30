
N = 10
s = [[] for n in range(N+1)]
s[0] = [1]
for n in range(1, N+1):
    s[n] = [0] * (2*n+1)
    for k in range(0, 2*n+1):
        for i in range(0, 2*n-1):   # n-trajectory, close slot n-(n+1) at step k
            s[n][k] += s[n-1][i] * min(k, i+1)
    print(s[n])
