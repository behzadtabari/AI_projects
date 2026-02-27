# Going Back to Basics


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
```

## Prefetching

Prefetching is a technique used in computing to improve performance by retrieving data or instructions before they are needed. By predicting what a program will request in the future, the system can load information in advance to reduced wait times, so how can find the optimal queue that can be used to fetch batches on CPU and then feed them for GPU computation.

### A Simple Experiment

I ran a graph level task with different number of workers to see how we can actually see that playing out.
 - w = num_workers
 - pin = whether pin_memory=True
 - pers = persistent_workers=True
 - pf = prefetch_factor
 - cudaPref = whether CUDA stream prefetching was enabled

please find prefetch.png and prefetch_2.png in the working directory, this traning is compute-bound on GPU, not data-bound on CPU, so it is evident why having zero number of workers yields the best results, But maybe in some tasks the batch creation is not cheap and actually computationally heavy.


### Reproducing the Result

Run the following script to reproduce the experiment:

```bash
python prefetching.py
```