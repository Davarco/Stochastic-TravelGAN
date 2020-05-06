import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('losses.txt', delimiter=' ')

X = np.arange(len(data))

plt.figure(0)
plt.plot(X, data[:, 0])
plt.title('Dreal')

plt.figure(1)
plt.plot(X, data[:, 1])
plt.title('Dfake')

plt.figure(2)
plt.plot(X, data[:, 2])
plt.title('G1_disc')

plt.figure(3)
plt.plot(X, data[:, 3])
plt.title('G2_disc')

plt.figure(4)
plt.plot(X, data[:, 4])
plt.title('siamese diff')

plt.figure(5)
plt.plot(X, data[:, 5])
plt.title('siamese same')

plt.figure(6)
plt.plot(X, data[:, 6])
plt.title('travel loss')

plt.show()
