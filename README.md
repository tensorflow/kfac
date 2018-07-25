# K-FAC: Kronecker-Factored Approximate Curvature

[![Travis](https://img.shields.io/travis/tensorflow/kfac.svg)](https://travis-ci.org/tensorflow/kfac)

**K-FAC in TensorFlow** is an implementation of [K-FAC][kfac-paper], an
approximate second-order optimization method, in TensorFlow. When applied to
feedforward and convolutional neural networks, K-FAC can converge `>3.5x`
faster in `>14x` fewer iterations than SGD with Momentum.

[kfac-paper]: https://arxiv.org/abs/1503.05671

## Installation

`kfac` is compatible with Python 2 and 3 and can be installed directly via
`pip`,

```shell
# Assumes tensorflow or tensorflow-gpu installed
$ pip install kfac

# Installs with tensorflow-gpu requirement
$ pip install 'kfac[tensorflow_gpu]'

# Installs with tensorflow (cpu) requirement
$ pip install 'kfac[tensorflow]'
```

## KFAC DOCS

Please check [KFAC docs][kfac_docs] for detailed description with examples
of how to use KFAC.

[kfac_docs]: https://github.com/tensorflow/kfac/tree/master/docs/index.md
