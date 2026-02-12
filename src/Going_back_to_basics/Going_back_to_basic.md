# Going Back to Basics

## Motivation

After years of working with deep learning, it is easy to overlook fundamentals that quietly but significantly affect results. This sub-folder serves as a reminder that *basic choices still matter*, even in seemingly trivial setups.

## Variance Amplification in GATs

### Why GATs Are Sensitive

Graph Attention Networks amplify variance twice:
1. Through the attention mechanism itself
2. Through neighborhood aggregation

This makes them particularly sensitive to weight initialization.

### A Simple Binary Experiment

To make this effect explicit, we construct a small synthetic **binary node-classification task** and train a shallow GAT.

Despite the simplicity of the data, improper initialization is enough to noticeably degrade performance.

### Weight Initialization Matters

Using a uniform distribution with a non–zero mean (e.g. `np.random.rand`) leads to poorer convergence and lower final accuracy compared to zero-mean Gaussian initialization (`np.random.randn`).

This effect persists even when:
- The task is binary
- The data is synthetic
- The model is shallow

### Reproducing the Result

Run the following script to reproduce the experiment:

```bash
python gat_init_compare.py
```

## Visualization of Activations

Well I might rethink the name that I have given to this part "Visualization of Activations", but I was wondering, how will each activation function act on a two-layer MLP, meaning that for me RELU is like a piecewise linear function summed up, which will be more clear if you run the script, feel free to use whatever you can to fit the bump in the random dataset.


### Reproducing the Result

Run the following script to reproduce the experiment:

```bash
python activ_vis.py
