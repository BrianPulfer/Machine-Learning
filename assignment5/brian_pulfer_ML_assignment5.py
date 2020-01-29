import math
import numpy as np
import matplotlib.pyplot as plt


def gaussian(mu, sigma, x):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * (math.exp(-1 / 2 * ((x - mu) / sigma) ** 2))


def sphere(x):
    ret_val = 0
    for i in range(len(x)):
        ret_val = ret_val + x[i] ** 2
    return ret_val


def rastrigin(x, A=10):
    n = len(x)
    ret_val = A * n

    for i in range(len(x)):
        ret_val = ret_val + (x[i] ** 2 - A * math.cos(2 * math.pi * x[i]))
    return ret_val


def plot_contours_sphere():
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)

    x1, x2 = np.meshgrid(x1, x2)

    sphere_val = np.sqrt(x1 ** 2 + x2 ** 2)

    sphere_fig, sphere_ax = plt.subplots(1, 1)
    sphere_cp = sphere_ax.contourf(x1, x2, sphere_val)
    sphere_fig.colorbar(sphere_cp)
    plt.show()


def plot_contours_rastrigin():
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)

    x1, x2 = np.meshgrid(x1, x2)

    rastrigin_val = 20 + x1 ** 2 - 10 * np.cos(2 * math.pi * x1) + x2 ** 2 - 10 * np.cos(2 * math.pi * x2)

    rastrigin_fig, rastrigin_ax = plt.subplots(1, 1)
    rastrigin_cp = rastrigin_ax.contourf(x1, x2, rastrigin_val)
    rastrigin_fig.colorbar(rastrigin_cp)
    plt.show()


def sample_points():
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)

    points = []
    y_sphere, y_rastr = [], []

    for i in range(len(x1)):
        for j in range(len(x2)):
            point = [x1[i], x2[j]]

            points.append(point)
            y_sphere.append(sphere(point))
            y_rastr.append(rastrigin(point))

    min_sphere_point = points[np.array(y_sphere).argmin()]
    min_rastrigin_point = points[np.array(y_rastr).argmin()]
    print("Global optimum for sphere seems to be around: ", min_sphere_point)
    print("Global optimum for rastrigin seems to be around: ", min_rastrigin_point)


def evaluate_individuals(individuals, function):
    evaluations = []
    for individual in individuals:
        evaluations.append(function(individual))
    return evaluations


def get_elite_set(percentage, individuals, evaluations):
    nr_elites = int(len(individuals) * percentage)
    inds, evals = np.array(individuals), np.array(evaluations)

    elites = []

    for i in range(nr_elites):
        elite_index = evals.argmin()
        evals[elite_index] = 1_000_000
        elite = inds[elite_index]
        elites.append(elite)
    return np.array(elites)


def run_cem(function, dimensions=100, nr_generations=250, nr_individuals=500, percentage_elite=0.3, nr_runs=3):
    best_fitnesses, worst_fitnesses = [], []

    for run in range(nr_runs):
        mu, sigma = np.random.uniform(-5, 5, size=dimensions), np.eye(dimensions)
        for generation in range(nr_generations):
            individuals = np.random.multivariate_normal(mu, sigma, nr_individuals)
            evaluations = evaluate_individuals(individuals, function)
            elite_set = get_elite_set(percentage_elite, individuals, evaluations)

            mu = np.mean(elite_set, axis=0)
            sigma = []
            for i in range(dimensions):
                sigma.append(np.cov(elite_set[:, i]))
            sigma = np.diag(sigma)

            best_fitnesses.append(np.min(evaluations))
            worst_fitnesses.append(np.max(evaluations))

    best_f = np.mean(np.reshape(np.array(best_fitnesses), newshape=(3, nr_generations)), axis=0)
    worst_f = np.mean(np.reshape(np.array(worst_fitnesses), newshape=(3, nr_generations)), axis=0)

    return best_f, worst_f, mu


def run_nes(function, dimensions=100, nr_generations=2000, nr_individuals=100, alpha=0.00001, nr_runs=3):
    best_fitnesses, worst_fitnesses = [], []

    for run in range(nr_runs):
        mu, sigma = np.random.uniform(-5, 5, size=dimensions), np.random.uniform(5, 10, size=dimensions)
        for generation in range(nr_generations):
            individuals = np.random.multivariate_normal(mu, np.diag(sigma ** 2), nr_individuals)
            evaluations = evaluate_individuals(individuals, function)

            u_gradients = (np.array(individuals) - mu) / sigma ** 2
            s_gradients = ((np.array(individuals) - mu) ** 2 - sigma ** 2) / sigma ** 3

            delta_j_mu = np.matmul(evaluations, u_gradients) / len(individuals)
            delta_j_sigma = np.matmul(evaluations, s_gradients) / len(individuals)

            # Noise in order to make the fisher matrix non-singular and thus invertible
            noise = np.eye(dimensions) * 0.001

            fisher_mu = np.matmul(u_gradients.T, u_gradients) / len(individuals) + noise
            fisher_sigma = np.matmul(u_gradients.T, u_gradients) / len(individuals) + noise

            fisher_mu_inv = np.linalg.inv(fisher_mu)
            fisher_sigma_inv = np.linalg.inv(fisher_sigma)

            mu = mu - alpha * np.matmul(fisher_mu_inv, delta_j_mu)
            sigma = sigma - alpha * np.matmul(fisher_sigma_inv, delta_j_sigma)

            best_fitnesses.append(np.min(evaluations))
            worst_fitnesses.append(np.max(evaluations))

    best_f = np.mean(np.reshape(np.array(best_fitnesses), newshape=(3, nr_generations)), axis=0)
    worst_f = np.mean(np.reshape(np.array(worst_fitnesses), newshape=(3, nr_generations)), axis=0)

    return best_f, worst_f, mu


def run_cma_es(function, dimensions=100, nr_generations=250, nr_individuals=1000, percentage_elite=0.4, nr_runs=3):
    best_fitnesses, worst_fitnesses = [], []

    for run in range(nr_runs):
        mu, sigma = np.random.uniform(-5, 5, size=dimensions), np.ones(
            shape=[dimensions, dimensions])  # [1] * dimensions  OR  np.ones(shape=[dimensions, dimensions])
        for generation in range(nr_generations):
            individuals = np.random.multivariate_normal(mu, sigma, nr_individuals)
            evaluations = evaluate_individuals(individuals, function)
            elite_set = get_elite_set(percentage_elite, individuals, evaluations)

            sigma = np.diag(np.sum((np.array(elite_set) - mu) ** 2, axis=0) / len(elite_set))
            mu = np.mean(elite_set, axis=0)

            best_fitnesses.append(np.min(evaluations))
            worst_fitnesses.append(np.max(evaluations))

    best_f = np.mean(np.reshape(np.array(best_fitnesses), newshape=(3, nr_generations)), axis=0)
    worst_f = np.mean(np.reshape(np.array(worst_fitnesses), newshape=(3, nr_generations)), axis=0)

    return best_f, worst_f, mu


def plot_best_and_worst(best_fitnesses, worst_fitnesses):
    x = []
    for i in range(len(best_fitnesses)):
        x.append(i)

    plt.plot(x, best_fitnesses, label='best')
    plt.plot(x, worst_fitnesses, label='worst')
    plt.show()


def benchmark(cem, cma_es):
    if len(cem) == len(cma_es):

        x = []
        for i in range(len(cem)):
            x.append(i)

        plt.plot(x, cem, 'r-')
        plt.plot(x, cma_es, 'b-')

        plt.show()


def main():
    # Point 1
    plot_contours_sphere()  # A
    plot_contours_rastrigin()  # B
    sample_points()  # C

    # Point 2 (CEM)
    bf_sphere_cem, wf_sphere_cem, final_mu_sphere_cem = run_cem(sphere)

    bf_rastrigin_cem, wf_rastrigin_cem, final_mu_rastrigin_cem = run_cem(rastrigin)

    plot_best_and_worst(bf_sphere_cem, wf_sphere_cem)
    plot_best_and_worst(bf_rastrigin_cem, wf_rastrigin_cem)

    # Point 3 (NES)
    bf_sphere_nes, wf_sphere_nes, final_mu_sphere_nes = run_nes(sphere)
    bf_rastrigin_nes, wf_rastrigin_nes, final_mu_rastrigin_nes = run_nes(rastrigin)

    plot_best_and_worst(bf_sphere_nes, wf_sphere_nes)
    plot_best_and_worst(bf_rastrigin_nes, wf_rastrigin_nes)

    # Point 4 (CMA-ES)
    bf_sphere_cmaes, wf_sphere_cmaes, final_mu_sphere_cmaes = run_cma_es(sphere)
    bf_rastrigin_cmaes, wf_rastrigin_cmaes, final_mu_rastrigin_cmaes = run_cma_es(rastrigin)

    plot_best_and_worst(bf_sphere_cmaes, wf_sphere_cmaes)
    plot_best_and_worst(bf_rastrigin_cmaes, wf_rastrigin_cmaes)

    # Point 5 (Benchmarking)
    benchmark(bf_sphere_cem, bf_sphere_cmaes)
    benchmark(bf_rastrigin_cem, bf_rastrigin_cmaes)

    benchmark(wf_sphere_cem, wf_sphere_cmaes)
    benchmark(wf_rastrigin_cem, wf_rastrigin_cmaes)


if __name__ == '__main__':
    main()
