def state_probability(pi, A, s, T):
    N = len(pi) # number of states
    probs = []
    for i in range(N):
        # What is p(Q0 = i)?
        probs.append(pi[i])
    for t in range(1, T+1):
        new_probs = [] # State probabilities at next time step
        for j in range(N):
            p = 0 # p(Qt = j)
            for i in range(N):
                p += probs[i] * A[i][j]
            new_probs.append(p)
        probs = new_probs
    # Return corresponding element of probs
    return probs[s]


if __name__ == '__main__':
    # Example from the tutorial / notes
    pi = [0.6, 0.3, 0.1]
    A = [[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [0.2, 0.3, 0.5]]
    s = 2
    T = 2
    print(state_probability(pi, A, s, T)) # Should output 0.195
