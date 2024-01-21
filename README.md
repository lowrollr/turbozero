# turbozero

📣 If you're looking for the PyTorch version of turbozero, it's been moved here: [turbozero_torch](https://github.com/lowrollr/turbozero_torch) 📣

`turbozero` is a vectorized implementation of AlphaZero written in JAX, implementing:
* Monte Carlo Tree Search with subtree persistence
* Batched Replay Memory
* A complete, customizable training/evaluation loop

With the help of `JAX`, `turbozero` is heavily parallelized, taking full advantage of hardware accelerators with vectorized algorithms and JIT-compilation.

`turbozero` is designed to be extendable, with an underlying search tree implementation that supports custom nodes, expansion logic, and more.

`turbozero` is easy to integrate with you custom JAX environment or neural network architecture. Use the provided training and evaluation utilities, or pick and choose the components that you need.

To get started, check out the [Hello World Notebook](https://github.com/lowrollr/turbozero/blob/main/notebooks/hello_world.ipynb)

## Installation
`turbozero` users `poetry` for dependency management, you can install it with:
```
pip install poetry
```
Then, to install dependencies:
```
poetry install
```
If you're using a GPU/TPU/etc., after running the previous command you'll need to install the device-specific version of JAX.

For a GPU w/ CUDA 12:
```
poetry source add jax https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry add jax[cuda12_pip]
```
See https://jax.readthedocs.io/en/latest/installation.html for other devices/cuda versions

I wish this could be done without extra commands but poetry does not support it :(

To launch an ipython kernel, run:
```
poetry run python -m ipykernel install --user --name turbozero
```

## Issues
If you use this project and encounter an issue, error, or undesired behavior, please submit a [GitHub Issue](https://github.com/lowrollr/turbozero/issues) and I will do my best to resolve it as soon as I can. You may also contact me directly via `hello@jacob.land`.

## Contributing 
Contributions, improvements, and fixes are more than welcome! For now I don't have a formal process for this, other than creating a [Pull Request](https://github.com/lowrollr/turbozero/pulls).

## Cite This Work
If you found this work useful, please cite it with:
```
@software{Marshall_TurboZero_Vectorized_AlphaZero,
  author = {Marshall, Jacob},
  title = {{TurboZero: Vectorized AlphaZero, MCTS, and Environments}},
  url = {https://github.com/lowrollr/turbozero}
}
```
