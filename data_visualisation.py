import numpy as np
import matplotlib.pyplot as plt

max_fit = np.load("data/MaxPIX.npy");
avg_fit = np.load("data/AvgPIX.npy");
hi_score = np.load("data/LengthPIX.npy");
#
hi_scores = []
max = 0

for item in hi_score:
    if item > max:
        max = item
    hi_scores.append(max)

X = np.linspace(0, len(hi_score), len(hi_score))

hi = plt.plot(X, hi_score*10, label="Length")
m_fit = plt.plot(X, max_fit, label="Max Fitness")
a_fit = plt.plot(X, avg_fit, label="Avg Fitness")
plt.legend()
plt.ylabel("Fitness / Length")
plt.xlabel("Generation")
plt.title("NEAT Snek")
plt.grid(True)
plt.show()
