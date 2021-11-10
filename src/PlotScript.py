# de https://www.enthought.com/wp-content/uploads/Enthought-MATLAB-to-Python-White-Paper.pdf
import numpy as np
import matplotlib.pyplot as plt

fs = [1, 2, 4]
all_time = np.linspace(0, 2, 200)
t = all_time[:100]
for f in fs:
    y = np.sin(2 * np.pi * f * t)
    plt.plot(t, y, label='{} Hz'.format(f))

plt.legend()
plt.savefig('basics_python.pdf')