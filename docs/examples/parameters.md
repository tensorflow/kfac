# K-FAC Parameters.

## Table of Contents

*   [Damping](#damping)
*   [Learning Rate](#learning-rate)
*   [Subsample covariance computation](#subsample-covariance-computation)
*   [KFAC norm constraint](#kfac-norm-constraint)
*   [Covariance decay](#covariance-decay)
*   [Train batch size](#train-batch-size)
    <br>

We list below various parameters which can be tuned to improve training and run
time performance of K-FAC.

## Damping

Damping is a crucial aspect of K-FAC, as it is for any second order
optimization/natural gradient method. Broadly speaking, it refers to the
practice of penalizing or constraining the size of the update in various ways so
that it doesn't leave the local region where the quadratic approximation to the
objective (which is used to compute the update) remains accurate. This region
commonly referred to as the "trust region". In some literature damping is called
"regularization" although we will avoid that term due to its related but
distinct meaning as a method to combat overfitting.

The damping strategy used in KFAC is to (approximately) add a multiple of the
identity to the Fisher before inverting it. This is essentially equivalent to
enforcing that the update lie in a spherical trust region centered at the
current location in parameter space.

The `damping` parameter represents the multiple of identity which is used.
Higher values correspond to smaller trust regions, although the precise
relationship between `damping` and the size of the trust region depends on the
scale of the objective, and will vary from iteration to iteration. (If the loss
function is multiplied by scalar 'alpha' then damping should be multiplied by
'alpha' as well.) Higher values of `damping` can allow higher learning rates,
but as damping tends to infinity the KFAC updates will start to resemble regular
gradient descent updates (scaled by `1/damping`).

The `damping` parameter depends on the scale of the loss function. `damping` is
a critical parameter that needs to be tuned. Options for tuning include a grid
sweep (must be simultaneous with learning rate optimization - NOT independent)
or auto-tuned using the Levenberg-Marquardt (LM) algorithm (see the [`Auto
Damping`][auto_damping] section for further details). For grid sweeps a typical
range to consider would be logarithmically spaced values between `1e-5` to
`100`, although the optimal value could be any non-negative real number in
principle (because the scale of the loss is arbitrary). Another option for
tuning `damping` is [`Population based training`][PBT] (PBT).

Refer to section `6` of the [KFAC paper][kfac_paper] for a more detailed
discussion of damping and how it can be used/tuned in KFAC

[auto_damping]:
https://github.com/tensorflow/kfac/tree/master/docs/examples/auto_damp.md
[PBT]:
https://arxiv.org/abs/1711.09846
[kfac_paper]:
https://arxiv.org/pdf/1503.05671.pdf

## Learning Rate

Typically sweep over values in the range 1e-5 to 100. It is important to tune
the learning in conjunction with damping, since the two are closely coupled
(higher damping allows higher learning rates). The learning rate can also be
tuned using PBT. Note that the optimal learning rate will be generally different
from the learning rate used for SGD/RMSProp/Adam optimizer.

## Subsample covariance computation

If you are using Conv layers and observe that the KFAC iterations is
significantly slower than Adam or if you run out of memory then a possible
remedy is to use subsampling in the covariance computation. To turn on
subsampling set `kfac_ff.sub_sample_inputs` to `True` and
`kfac_ff.sub_sample_outer_products` to `True`. The former flag subsamples the
batch of inputs used for covariance computation and the later flag subsamples
extracted patches based on the size of the covariance matrix. Check the
documentation of `tensorflow_kfac.fisher_factors` for detailed explanation of
various subsampling parameters. Also check [`Distributed training`][dist_train]
section for how to distribute the computation of these ops over multiple
devices.

[dist_train]:
https://github.com/tensorflow/kfac/tree/master/docs/examples/distributed_training.md

## KFAC norm constraint

Scales the K-FAC update so that its approximate Fisher norm is bounded.
Typically use an initial value of 1.0 and tune it using PBT or perform grid
search. Norm constraint can used as an alternative to learning rate schedules.
See Section 5 of the [Distributed Second-Order Optimization using
Kronecker-Factored Approximations][ba_paper] paper for further details.

[ba_paper]:
https://jimmylba.github.io/papers/nsync.pdf

## Covariance decay

During the course of the algorithm, an exponential moving average tracks
statistics for each layer. Slower decays mean that the statistics are based on
more data, but will suffer more from the issue of staleness (because of the
changing model parameters). This parameter can usually be left at its default
value but may occasionally matter for some problems. In such cases some
reasonable values to sweep over are `[0.9, 0.95, 0.99, 0.999]`.

## Train batch size

Typically try using a larger batch size compared to training with
SGD/RMSprop/Adam.
