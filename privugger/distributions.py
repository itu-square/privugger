"""
Module describing possible distributions
"""
POSSIBLE_INTS = [i for i in range(6)]
POSSIBLE_FLOATS = [i for i in range(6,19)]

#Ints Value
BINOMIAL, BERNOULLI, GEOMETRIC, BETA_BINOMIAL, POISSON, DISCRETE_UNIFORM = range(6)

#Floats Value
NORMAL, UNIFORM, TRUNCATED_NORMAL = range(6,9)
BETA, EXPONENTIAL, LAPLACE, STUDENT_T = range(9, 13)
CAUCHY, GAMMA, LOG_NORMAL = range(13, 16)
CHI_SQUARED, TRIANGULAR, LOGISTIC = range(16,19)