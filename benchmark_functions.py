import numpy as np


    
    

def sphere(x):
    return np.sum(x**2)



def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def step_2(x):
    return np.sum(np.floor(x + 0.5)**2)

# Add more functions following the pattern above...

def quartic(x):
    return np.sum(7 * x**4 - 10 * x**3 + 5 * x**2)


def schwefel_2_21(x):
    return np.sum(abs(x) + np.prod(abs(np.sin(x))))


def schwefel_2_22(x):
    return -np.sum(np.sin(np.sqrt(abs(x))))



def foxholes(x):
    return np.sum(
      np.exp(-0.2 * np.sqrt(np.mean(x ** 2, axis=0))) + np.exp(np.sin(np.sqrt(np.mean(x ** 2, axis=0)))) + 0.1 *
    x.shape[0])


def kowalik(x):
    return np.sum([i * (x[j] - 1)**2 + (1 - x[j])**2 for j in range(len(x)) for i in range(1, 6)])

# Six-hump camel back Function

def six_hump_camel_back(x):
    return 4 * x[0]**2 - 2.1 * x[0]**4 + 1/3 * x[0]**6 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4


# # Hartman 6 Function
# def hartman_6(x):
#     return -np.sum(np.array([0.3979, 0.4899, 0.6759, 0.7699, 0.9149, 1.0472]) * np.exp(-np.sum(np.array([1, 10, 100, 1000, 10000, 100000]) * ((x - np.array([0.1312, 0.2329, 0.5358, 0.8775, 0.9991, 0.7743]))**2), axis=1)))

# Levi Function N.13

def levi_n13(x):
    return np.sum(np.sin(np.sqrt(abs(x**2 + (1 + np.sin(10 * np.pi * x))**2))))

# Rastrigin Function

def rastrigin(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

# Griewank Function

def griewank(x):
    return np.sum(x**2 / 4000 - np.cos(x) + 1)

# Ackley 1 Function

def ackley_1(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e


# def hartman_3(x):
#     x = np.atleast_2d(x)  # Ensure x is a 2D array for proper broadcasting
#     weights = np.array([0.3689, 0.4699, 1.0472, 1.5701, 0.7473])[:x.shape[1]]
#     return -np.sum(weights * np.exp(-np.sum(np.array([1, 10, 100, 1000]) * (x.T - np.array([0.1312, 0.2329, 0.5358, 0.8775]))**2, axis=1)))

def schwefel_2_26(x):
    return np.sum(np.abs(x) + np.sin(x**2))

def branin_rcos(x):
    return (x[1] - 5.1 / (4 * np.pi**2) * x[0]**2 + 5 * x[0] / np.pi - 6)**2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

def goldstein_price(x):
    return (1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)) * \
           (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))


def penalized_1(x):
    term1 = 10 * np.sin(np.pi * x[0])**2
    term2 = sum((x[i] - 1)**2 * (1 + 10 * np.sin(np.pi * x[i] + 1)**2) for i in range(1, len(x)))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    return term1 + term2 + term3

# def shekel_1(x):
#     a = np.array([[4, 4, 4, 4],
#                   [1, 1, 1, 1],
#                   [8, 8, 8, 8],
#                   [6, 6, 6, 6],
#                   [3, 7, 3, 7]])
#
#     b = np.array([0.1, 0.2, 0.2, 0.4, 0.4])
#
#     x = np.atleast_2d(x)  # Ensure x is a 2D array for proper broadcasting
#     return -np.sum((1 / (np.sum((x[:, np.newaxis, :] - a)**2, axis=-1) + b)), axis=-1)
#


# def shekel_2(x):
#     a = np.array([[4, 4, 4, 4],
#                   [1, 1, 1, 1],
#                   [8, 8, 8, 8],
#                   [6, 6, 6, 6],
#                   [3, 7, 3, 7]])
#
#     b = np.array([0.1, 0.2, 0.2, 0.4, 0.4])
#
#     return -np.sum((1 / (np.sum((x - a)**2, axis=1) + b))**2)
#
# def shekel_3(x):
#     a = np.array([[4, 4, 4, 4],
#                   [1, 1, 1, 1],
#                   [8, 8, 8, 8],
#                   [6, 6, 6, 6],
#                   [3, 7, 3, 7]])
#
#     b = np.array([0.1, 0.2, 0.2, 0.4, 0.4])
#
#     return -np.sum(np.sqrt(1 / (np.sum((x - a)**2, axis=1) + b)))


