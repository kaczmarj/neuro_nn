import numpy as np
import pytest
import tensorflow as tf

from nobrainer.models.highresnet import highresnet
from nobrainer.models.meshnet import meshnet
from nobrainer.models.meshnet import meshnet_vwn
from nobrainer.models.unet import unet
from nobrainer.models.autoencoder import autoencoder
from nobrainer.models.dcgan import dcgan

def model_test(model_cls, n_classes, input_shape, kwds={}):
    """Tests for models."""
    x = 10 * np.random.random(input_shape)
    y = np.random.choice([True, False], input_shape)

    # Assume every model class has n_classes and input_shape arguments.
    model = model_cls(n_classes=n_classes, input_shape=input_shape[1:], **kwds)
    model.compile(tf.train.AdamOptimizer(), 'binary_crossentropy')
    model.fit(x, y)

    actual_output = model.predict(x)
    assert actual_output.shape == x.shape[:-1] + (n_classes,)


def test_highresnet():
    model_test(highresnet, n_classes=1, input_shape=(1, 32, 32, 32, 1))


def test_meshnet():
    model_test(meshnet, n_classes=1, input_shape=(1, 32, 32, 32, 1), kwds={'receptive_field': 37})
    model_test(meshnet, n_classes=1, input_shape=(1, 32, 32, 32, 1), kwds={'receptive_field': 67})
    model_test(meshnet, n_classes=1, input_shape=(1, 32, 32, 32, 1), kwds={'receptive_field': 129})
    with pytest.raises(ValueError):
        model_test(meshnet, n_classes=1, input_shape=(1, 32, 32, 32, 1), kwds={'receptive_field': 50})


@pytest.mark.skip("this overloads memory on travis ci")
def test_meshnet_vwn():
    model_test(meshnet_vwn, n_classes=1, input_shape=(1, 32, 32, 32, 1), kwds={'receptive_field': 37})
    model_test(meshnet_vwn, n_classes=1, input_shape=(1, 32, 32, 32, 1), kwds={'receptive_field': 67})
    model_test(meshnet_vwn, n_classes=1, input_shape=(1, 32, 32, 32, 1), kwds={'receptive_field': 129})
    with pytest.raises(ValueError):
        model_test(meshnet, n_classes=1, input_shape=(1, 32, 32, 32, 1), kwds={'receptive_field': 50})


def test_unet():
    model_test(unet, n_classes=1, input_shape=(1, 32, 32, 32, 1))


def test_autoencoder():
    """Special test for autoencoder."""

    input_shape=(1,32,32,32,1)
    x = 10 * np.random.random(input_shape)

    model = autoencoder(input_shape[1:], encoding_dim=128, n_base_filters=32)
    model.compile(tf.train.AdamOptimizer(), 'mse')
    model.fit(x, x)

    actual_output = model.predict(x)
    assert actual_output.shape == x.shape


def test_dcgan():
    """Special test for dcgan."""

    output_shape = (1,32,32,32,1)
    z_dim = 32
    z = np.random.random((1,z_dim))

    pred_shape = (1,8,8,8,1)

    generator, discriminator = dcgan(output_shape[1:], z_dim=z_dim)
    generator.compile(tf.train.AdamOptimizer(), 'mse')
    discriminator.compile(tf.train.AdamOptimizer(), 'mse')

    fake_images = generator.predict(z)
    fake_pred = discriminator.predict(fake_images)

    assert fake_images.shape == output_shape and fake_pred.shape == pred_shape
