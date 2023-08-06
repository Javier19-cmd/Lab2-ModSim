import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_uniform_samples(n):
    return np.random.uniform(0, 1, n)

def calculate_mean(samples):
    return np.mean(samples)

def calculate_normalized_mean(sample_mean, mu, chi, n):
    return (sample_mean - mu) / (chi / np.sqrt(n))

def plot_histogram_and_normal_density(data, title, bins=30):
    data = np.array(data)
    plt.hist(data, bins=bins, density=True, alpha=0.6, label='Histograma')
    x = np.linspace(data.min(), data.max(), 100)
    plt.plot(x, norm.pdf(x), 'r', label='Densidad normal N(0,1)')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_empirical_cumulative_distribution(data):
    sorted_data = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    plt.plot(sorted_data, y, marker='.', linestyle='none', label='Frecuencia relativa acumulada')
    x = np.linspace(-4, 4, 100)
    plt.plot(x, norm.cdf(x), 'r', label='Distribución acumulada N(0,1)')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa acumulada')
    plt.title('Función de distribución acumulada empírica y N(0,1)')
    plt.legend()
    plt.show()

def main():
    ns = [20, 40, 60, 80, 100]
    Ns = [50, 100, 1000, 10000]

    mu = 0.5  # Media de la distribución uniforme en (0, 1)
    chi_squared = 1 / 12  # Varianza de la distribución uniforme en (0, 1)

    for n in ns:
        sample_means = [calculate_mean(generate_uniform_samples(n)) for _ in range(max(Ns))]
        normalized_means = [calculate_normalized_mean(mean, mu, chi_squared, n) for mean in sample_means]
        plot_histogram_and_normal_density(normalized_means, f'Histograma y densidad normal para n={n}')

    for N in Ns:
        sample_means = [calculate_mean(generate_uniform_samples(n)) for _ in range(N)]
        normalized_means = [calculate_normalized_mean(mean, mu, chi_squared, n) for n, mean in zip(ns, sample_means)]
        plot_empirical_cumulative_distribution(normalized_means)

if __name__ == "__main__":
    main()
