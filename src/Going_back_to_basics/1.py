import numpy as np
import matplotlib.pyplot as plt
# Logistic regression and Bernoulilli distribution
# Avoid log(0)
p = np.linspace(1e-6, 1 - 1e-6, 500)

# Loss for y=1 and y=0
loss_y1 = -np.log(p)
loss_y0 = -np.log(1 - p)

plt.figure()
plt.plot(p, loss_y1, label="y = 1:  -log(p)")
plt.plot(p, loss_y0, label="y = 0:  -log(1-p)")
plt.xlabel("Predicted probability p")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
plt.title("Log Loss for Binary Classification")