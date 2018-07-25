# Automatic tuning of damping parameter.

[TOC]

The [KFAC damping parameter][kfac_damp] can be auto tuned using
Levenberg-Marquardt (LM) algorithm. For a detailed description of the algorithm
refer to `Section 6` of the [KFAC Paper][kfac_paper]. Note this is still a
heuristic and may not always produce optimal results. It can be better or worse
than a carefully tuned fixed value, depending on the problem.

[kfac_paper]: https://arxiv.org/pdf/1503.05671.pdf
[kfac_damp]: https://github.com/tensorflow/kfac/tree/master/docs/examples/parameters.md

**Example code**:
https://github.com/tensorflow/kfac/tree/master/kfac/examples/kfac_mnist_autoencoder_auto_damping.py

Using this method to auto tune damping requires changes to the basic KFAC
training script, which are described below. We only highlight additional steps
required vs training with a fixed damping value (as in the [Convnet
example][convexamplesec])

[convexamplesec]: https://github.com/tensorflow/kfac/tree/master/docs/examples/convolutional.md

## 1. Cached Reader

Wrap the dataset into `CachedReader`. This allows us to access previous batch of
data.

```python
  # Load a dataset.
  data_set = mnist.load_mnist(
      FLAGS.data_dir,
      num_epochs=FLAGS.num_epochs,
      batch_size=_BATCH_SIZE,
      flatten_images=True)
  # Wrap the data set into cached_reader which provides variable sized training
  # and caches the read train batch.
  cached_reader = data_reader.CachedDataReader(data_set, _BATCH_SIZE)
  cur_batch = cached_reader(batch_size)[0]
```

## 2. Build optimizer and set damping parameters

```python
  optimizer = kfac_opt.PeriodicInvCovUpdateKfacOpt(
      invert_every=FLAGS.inverse_update_period,
      cov_update_every=FLAGS.cov_update_period,
      learning_rate=1.0,
      damping=150.,
      cov_ema_decay=0.95,
      momentum=0.95,
      layer_collection=layer_collection,
      batch_size=batch_size
      )
  # Set the damping parameters required to adapt damping.
  optimizer.set_damping_adaptation_params(
      prev_train_batch=cached_reader.cached_batch,
      is_chief=True,
      loss_fn=loss_fn,
      damping_adaptation_decay=0.95,
      damping_adaptation_interval=FLAGS.damping_adaptation_interval,
  )
  train_op = optimizer.minimize(loss, global_step=global_step)
```

## TIPS:

1.  Damping can also be tuned using Population based training ([PBT][PBT_link]).
    In our observations PBT works on par with auto tuning using LM algorithm,
    although is obviously more computationally expensive. However if you are
    already doing PBT for other hyperparams then consider tuning damping using
    PBT as well.

[PBT_link]: https://arxiv.org/abs/1711.09846
