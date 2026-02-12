"""
Small visualization toolkit to see how tweaking lr, or number of layers
may affect the results, feel free to change the optim, sequential layers, etc
to see the trainig, try to fit the Gaussian Bump, have fun, maybe one day
 I will also write the most optimized setting for this problems

Requirements:
  pip install torch numpy matplotlib

Run:
  python gat_init_compare.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("TkAgg")   # or "QtAgg"
import numpy as np
import matplotlib.pyplot as plt


class TrainingPlotter:
    """
    Live training plot helper.
    - Panel 1: y_true scatter + model prediction curve
    - Panel 2: loss over time
    """
    def __init__(self, every=50, figsize=(10, 4), sort_x=True):
        self.every = every
        self.figsize = figsize
        self.sort_x = sort_x
        self.loss_history = []
        self.epoch_history = []

        self._fig = None
        self._ax1 = None
        self._ax2 = None

    @torch.no_grad()
    def update(self, epoch, loss, model, distances_norm, times_norm, title_prefix="Epoch"):
        # Record history
        loss_val = float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float(loss)
        self.epoch_history.append(epoch + 1)
        self.loss_history.append(loss_val)

        # Prep data (CPU numpy for plotting)
        x = distances_norm.detach().cpu().view(-1).numpy()
        y = times_norm.detach().cpu().view(-1).numpy()

        model_was_training = model.training
        model.eval()

        # Sort x for a smooth prediction line (important!)
        if self.sort_x:
            idx = np.argsort(x)
            xs = x[idx]
            ys = y[idx]
        else:
            xs, ys = x, y

        xs_t = torch.from_numpy(xs).to(distances_norm.device).view(-1, 1).type_as(distances_norm)
        yhat = model(xs_t).detach().cpu().view(-1).numpy()

        # Restore model mode
        if model_was_training:
            model.train()

        # Initialize figure once
        if self._fig is None:
            plt.ion()
            self._fig, (self._ax1, self._ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Clear axes (redraw fresh)
        self._ax1.clear()
        self._ax2.clear()

        # Panel 1: fit
        self._ax1.scatter(xs, ys, s=25, alpha=0.85, label="Actual Normalized Data")
        self._ax1.plot(xs, yhat, linewidth=2, color="red", label="Model Predictions")
        self._ax1.set_xlabel("ages")
        self._ax1.set_ylabel("incomes")
        self._ax1.set_title(f"{title_prefix}: {epoch+1} | Training Progress")
        self._ax1.grid(True, alpha=0.3)
        self._ax1.legend(loc="best")

        # Panel 2: loss curve
        self._ax2.plot(self.epoch_history, self.loss_history)
        self._ax2.set_xlabel("Epoch")
        self._ax2.set_ylabel("Loss")
        self._ax2.set_title("Loss Curve")
        self._ax2.grid(True, alpha=0.3)

        # Render
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)


# Convenience function if you don't want a class
_plotter_singleton = None

def plot_training_progress(epoch, loss, model, distances_norm, times_norm, every=50):
    global _plotter_singleton
    if _plotter_singleton is None:
        _plotter_singleton = TrainingPlotter(every=every)

    # only update at cadence
    if (epoch + 1) % every == 0:
        _plotter_singleton.update(epoch, loss, model, distances_norm, times_norm)


rng = np.random.default_rng(0)
n = 100

ages = torch.randint(18, 75, (n, 1), dtype=torch.float32)

peak_age = 47.0
peak_income = 6500.0
base_income = 2200.0
width = 10.0

# Torch-only expected curve
expected = base_income + (peak_income - base_income) * torch.exp(
    -0.5 * ((ages - peak_age) / width) ** 2
)

# Noise: make it torch too
noise = torch.from_numpy(rng.lognormal(mean=0.0, sigma=0.25, size=(n, 1))).float()

incomes = expected * noise
incomes = torch.clamp(incomes, 2000.0, 12000.0)


# x normalization
ages_mu, ages_std = ages.mean(), ages.std()
ages_n = (ages - ages_mu) / (ages_std + 1e-8)

# y log-transform (since noise is lognormal-ish)
incomes_log = torch.log(incomes)
incomes_mu, incomes_std = incomes_log.mean(), incomes_log.std()
incomes_n = (incomes_log - incomes_mu) / (incomes_std + 1e-8)

# ----- model -----
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),          # smooth nonlinearity often fits bumps nicely
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=1e-4)
loss_function = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=200
)


for epoch in range(10000):
    # Reset the optimizer's gradients
    optimizer.zero_grad()
    # Make predictions (forward pass)
    outputs = model(ages)
    # Calculate the loss
    loss = loss_function(outputs, incomes)
    # Calculate adjustments (backward pass)
    loss.backward()
    # Update the model's parameters
    optimizer.step()
    #scheduler.step(loss)

    # Create a live plot every 50 epochs
    if (epoch + 1) % 50 == 0: 
        plot_training_progress(
            epoch=epoch,
            loss=loss,
            model=model,
            distances_norm=ages,
            times_norm=incomes
        )

print("\nTraining Complete.")
print(f"\nFinal Loss: {loss.item()}")
