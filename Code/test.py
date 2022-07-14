from scipy.stats import norm,beta
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

from scipy.stats import norm
lower_bound = norm.ppf(0.01)
upper_bound = norm.ppf(0.99)
N = 101
x, step = np.linspace(0, 1, N, retstep=True)
P = beta.pdf(x, 10,5)
print(np.where( np.max(P) == P ))
print(np.sum(P * step))
plt.plot(x, P,
       'r-', lw=5, alpha=0.6, label='beta pdf')
plt.plot(x, P*step,
       'b-', lw=5, alpha=0.6, label='beta pdf')









a=10
b=10
mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

x = np.linspace(beta.ppf(0, a, b),
                beta.ppf(1, a, b), 100)
plt.plot(x, beta.pdf(x, a, b),
       'r-', lw=5, alpha=0.6, label='beta pdf')


likelihood = [.7,.8,.6,.7,.2]
lno = [1-l for l in likelihood]
#likelihood = sum(likelihood)
#lno = sum(lno)
mean2, var2, skew2, kurt2 = beta.stats(likelihood, lno, moments='mvsk')

x1 = np.linspace(beta.ppf(0, likelihood, lno),
                beta.ppf(1, likelihood, lno), 100)
plt.plot(x1, beta.pdf(x1, likelihood, lno),
       'b-', lw=5, alpha=0.6, label='beta pdf')


x_axis = np.arange(0, 1, 0.01)

plt.plot(x_axis, beta.pdf(10,10))
plt.show()