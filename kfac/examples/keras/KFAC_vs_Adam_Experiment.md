# KFAC vs Adam Experiment

## Set Up

We compare KFAC and Adam on a RESNET-20 on the CIFAR10 dataset. We split CIFAR10
into a training (40k), validation (10k), and test (10k) sets. We ran a random
hyperparameter search where the best hyperparameters were chosen by the run that
reaches 89% validation accuracy first in terms of number of steps. We decay both
learning rate and damping/epsilon exponentially. The final learning rate is
fixed at 1e-4, final damping (KFAC) at 1e-6, and final epsilon (Adam) at 1e-8.
Below are the ranges of the tuned hyperparamters. The random search samples all
the hyperparameters from a log uniform scale:

| Hyperparameter            | Min  | Max   |
|---------------------------|------|-------|
| Init Learning Rate        | 1e-2 | 10.0  |
| Init Damping (KFAC)       | 1e-2 | 100.0 |
| Init Epsilon (Adam)       | 1e-4 | 1.0   |
| 1 - Learning Rate Decay   | 1e-4 | 0.1   |
| 1 - Damping/Epsilon Decay | 1e-4 | 0.1   |
| 1 - Momentum              | 1e-2 | 0.3   |

The initial tuning run was with seed 20190524 with the GPU training script on an
NVIDIA Tesla P100. Then, after choosing the best hyperparameters, we ran those
hyperparameters with the following 10 random seeds: 351515, 382980, 934126,
891369, 64379, 402680, 672242, 421590, 498163, 448799.

# Results

The chosen hyperparameters were the following (to 6 decimal places):

| Hyperparameter            | KFAC     | Adam     |
|---------------------------|----------|----------|
| Init Learning Rate        | 0.227214 | 2.242663 |
| Init Damping (KFAC)       | 0.288721 |          |
| Init Epsilon (Adam)       |          | 0.183230 |
| 1 - Learning Rate Decay   | 0.001090 | 0.000610 |
| 1 - Damping/Epsilon Decay | 0.000287 | 0.000213 |
| 1 - Momentum              | 0.018580 | 0.029656 |

## Training Curves

Below are the loss and accuracy training curves with the training and test sets.
The line represents the mean of the 10 seed runs and the coloured region
represents the bootstrapped standard deviation. KFAC reaches 89% validation
accuracy at step 4640 and Adam at step 6560 (measurements were taken every 40
steps).

![](https://github.com/tensorflow/kfac/tree/master/kfac/examples/keras/plots/kfac_v_adam_loss_curve.png?raw=true)

![](https://github.com/tensorflow/kfac/tree/master/kfac/examples/keras/plots/kfac_v_adam_accuracy_curve.png?raw=true)

Among the other runs, KFAC decreases training loss quicker than Adam early in
training, then show similar performance later in training.

## Hyperparameter Analysis

We offer some analysis of the learning rate and damping for KFAC to aid in
choosing appropriate values for these hyperparameters. Plots with the rest of
the hyperparameters for both KFAC and Adam are in the plots folder.

![](https://github.com/tensorflow/kfac/tree/master/kfac/examples/keras/plots/kfac_lr_v_damping.png?raw=true)

In general, a higher learning rate requires a higher damping. A large learning
rate with low damping leads to divergence, whereas a low learning rate with high
damping leads to SGD-like behaviour, which is suboptimal. The plot above shows
little correlation due to the decay schedules playing a large role, which is
shown below:

![](https://github.com/tensorflow/kfac/tree/master/kfac/examples/keras/plots/kfac_damping_v_dampingdecay.png?raw=true)

A fast damping decay allows for faster training, but can easily lead to
divergence. The best runs are often close to diverging.

![](https://github.com/tensorflow/kfac/tree/master/kfac/examples/keras/plots/kfac_lr_v_lrdecay.png?raw=true)

As expected, a high learning rate with a low decay can lead to divergence.

![](https://github.com/tensorflow/kfac/tree/master/kfac/examples/keras/plots/kfac_lrdecay_v_dampingdecay.png?raw=true)

Just like with the learning rate and damping, the learning rate decay should
be proportional the damping decay to prevent divergence while training quickly.
