📣 [JAX is coming](https://github.com/lowrollr/turbozero/discussions/4)

(readme coming soon)

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
Contributions, improvements, and fixes are more than welcome! I've written a lot in the [Wiki](https://github.com/lowrollr/turbozero/wiki), I hope it provides enough information to get started. For now I don't have a formal process for this, other than creating a [Pull Request](https://github.com/lowrollr/turbozero/pulls).

## Cite This Work
If you found this work useful, please cite it with:
```
@software{Marshall_TurboZero_Vectorized_AlphaZero,
  author = {Marshall, Jacob},
  title = {{TurboZero: Vectorized AlphaZero, MCTS, and Environments}},
  url = {https://github.com/lowrollr/turbozero}
}
```
