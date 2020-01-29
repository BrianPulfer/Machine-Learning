import numpy as np

x = [[-9, -5, 5],
        [4, -7, -11],
        [7, 6, -1],
        [-9, -5, 4],
        [-5, -6, -1],
        [-4, -4, -8],
        [5, 7, -9],
        [2, -4, 3],
        [-6, 1, 7],
        [-10, 6, -7]]

y = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]

w = [-0.1, -0.3, 0.2]
b = 2

LEARNING_RATE = 0.02


def predict(x):
    return w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b


def main():
    de_dws = [0, 0, 0]

    steps = 10

    for step in range(steps):
        print("\nWeights at step "+str(step)+": "+str(w[0])+" "+str(w[1])+" "+str(w[2]))

        total_error = 0

        for i in range(len(x)):
            prediction = predict(x[i])
            actual = y[i]

            error = ((actual - prediction)*(actual - prediction)/2)
            total_error = total_error + error

            de_dy = (prediction - actual)

            for j in range(len(de_dws)):
                dy_dw = x[i][j]
                de_dws[j] = de_dws[j] + (de_dy * dy_dw) / len(x)

        # WEIGHTS UPDATE
        for i in range(len(w)):
            w[i] = w[i] - LEARNING_RATE * de_dws[i]

        total_error = total_error / len(x)
        print('Total error at step '+str(step)+': '+str(total_error))


if __name__ == '__main__':
    main()
