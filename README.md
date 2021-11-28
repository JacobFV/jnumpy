# Jnumpy

[![PyPI version](https://badge.fury.io/py/jnumpy.svg)](https://badge.fury.io/py/jnumpy)
[![](https://img.shields.io/badge/license-MIT-blue)](https://github.com/JacobFV/jnumpy/blob/main/LICENSE)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e1cb295484424f36acf2c813fae6f57e)](https://app.codacy.com/gh/JacobFV/jnumpy?utm_source=github.com&utm_medium=referral&utm_content=JacobFV/jnumpy&utm_campaign=Badge_Grade_Settings)

Jacob's numpy library for machine learning

## Getting started

1. Install from `pip` or clone locally:

```bash
$ pip install jnumpy
# or
$ git clone https://github.com/JacobFV/jnumpy.git
$ cd jnumpy
$ pip install .
```

2. Import the `jnumpy` module.

```python
import jnumpy as jnp
```

## Example

```python
```

## Limitations and Future Work

Version 2.0 is under development (see the dev branch) and will feature:
- static execution graphs
- a keras-style neural network API with `fit`, `evaluate`, and `predict`
- richer collections of optimizers, metrics, and losses
- more examples

Also maybe for the future:
- custom backends (i.e.: tensorflow or pytorch instead of numpy)

## License

All code in this repository is licensed under the MIT license. No restrictions, but no warranties. See the [LICENSE](https://github.com/JacobFV/jnumpy/blob/main/LICENSE) file for details.

## Contributing

This is a small project, and I don't plan on growing it much. You are welcome to fork and contribute or email me `jacob` [dot] `valdez` [at] `limboid` [dot] `ai` if you would like to take over. You can add your name to the copyright if you make a PR or your own branch.

The codebase is kept in only a few files, and I have tried to minimize the use of module prefixes because my CSE 4308/4309/4392 classes require the submissions to be stitched togethor in a single file. 