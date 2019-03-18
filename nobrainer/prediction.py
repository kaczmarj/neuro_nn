# -*- coding: utf-8 -*-
"""Methods to predict using trained models."""

import math
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import tensorflow as tf

from nobrainer.transform import get_affine
from nobrainer.transform import warp
from nobrainer.volume import from_blocks
from nobrainer.volume import from_blocks_numpy
from nobrainer.volume import standardize_numpy
from nobrainer.volume import to_blocks
from nobrainer.volume import to_blocks_numpy

DT_X = "float32"
_INFERENCE_CLASSES_KEY = "class_ids"


def predict(inputs,
            model,
            block_shape,
            return_variance=False,
            return_entropy=False,
            return_array_from_images=False,
            n_samples=1,
            batch_size=4,
            dtype=DT_X):
    """Return predictions from `inputs`.

    This is a general prediction method that can accept various types of
    `inputs`.

    Parameters
    ----------
    inputs: 3D array or Nibabel image or filepath or list of filepaths.
    model: str: path to saved model, either HDF5 or SavedModel.
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.
    return_array_from_images: Boolean. If set True and the given input is either image,
        filepath, or filepaths, it will return arrays of [mean, variance, entropy]
        instead of images of them. Also, if the input is array, it will
        simply return array, whether or not this flag is True or False.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value. The default value is 1
    block_shape: tuple of length 3, shape of sub-volumes on which to
        predict.
    normalizer: callable, function that accepts two arguments
        `(features, labels)` and returns a tuple of modified
        `(features, labels)`.
    batch_size: int, number of sub-volumes per batch for predictions.
    dtype: str or dtype object, datatype of features array.

    Returns
    -------
    If `inputs` is a:
        - 3D numpy array, return an iterable of maximum 3 elements;
            3D array of mean, variance(optional),and entropy(optional) of prediction.
            if the flags for variance or entropy is set False, it won't be returned at all
            The specific order of the elements are:
            mean, variance(default=None) , entropy(default=None)
            Note, variance is only defined when n_sample  > 1
        - Nibabel image or filepath, return a set of Nibabel images of mean, variance,
            entropy of predictions or just the pure arrays of them,
            if return_array_from_images is True.
        - list of filepaths, return generator that yields one set of Nibabel images
            or arrays(if return_array_from_images is set True) of means, variance, and
            entropy predictions per iteration.
    """
    if n_samples < 1:
        raise Exception('n_samples cannot be lower than 1.')

    model = _get_model(model)

    if isinstance(inputs, np.ndarray):
        out = predict_from_array(
            inputs=inputs,
            model=model,
            block_shape=block_shape,
            return_variance=return_variance,
            return_entropy=return_entropy,
            return_array_from_images=return_array_from_images,
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size)
    elif isinstance(inputs, nib.spatialimages.SpatialImage):
        out = predict_from_img(
            img=inputs,
            model=model,
            block_shape=block_shape,
            return_variance=return_variance,
            return_entropy=return_entropy,
            return_array_from_images=return_array_from_images,
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size,
            dtype=dtype)
    elif isinstance(inputs, str):
        out = predict_from_filepath(
            filepath=inputs,
            model=model,
            block_shape=block_shape,
            return_variance=return_variance,
            return_entropy=return_entropy,
            return_array_from_images=return_array_from_images,
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size,
            dtype=dtype)
    elif isinstance(inputs, (list, tuple)):
        out = predict_from_filepaths(
            filepaths=inputs,
            model=model,
            block_shape=block_shape,
            return_variance=return_variance,
            return_entropy=return_entropy,
            return_array_from_images=return_array_from_images,
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size,
            dtype=dtype)
    else:
        raise TypeError("Input to predict is not a valid type")
    return out


def predict_from_array(inputs,
                       model,
                       block_shape,
                       return_variance=False,
                       return_entropy=False,
                       return_array_from_images=False,
                       n_samples=1,
                       normalizer=None,
                       batch_size=4):
    """Return a prediction given a filepath and an ndarray of features.

    Parameters
    ----------
    inputs: ndarray, array of features.
    model: `tf.keras.Model`, trained model.
    block_shape: tuple of len 3, shape of blocks on which to predict.
    return_variance: 'y' or 'n'. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.
    return_array_from_images: Boolean. If set True and the given input is either image,
        filepath, or filepaths, it will return arrays of [mean, variance, entropy]
        instead of images of them. Also, if the input is array, it will
        simply return array, whether or not this flag is True or False.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.
    batch_size: int, number of sub-volumes per batch for prediction.

    Returns
    -------
    ndarray of predictions.
    """
    if normalizer:
        features = normalizer(inputs)
    else:
        features = inputs
    features = to_blocks_numpy(features, block_shape=block_shape)
    means = np.zeros_like(features)
    variances = np.zeros_like(features)
    entropies = np.zeros_like(features)

    features = features[..., None]  # Add a dimension for single channel.

    # Predict per block to reduce memory consumption.
    n_blocks = features.shape[0]
    n_batches = math.ceil(n_blocks / batch_size)
    progbar = tf.keras.utils.Progbar(n_batches)
    progbar.update(0)
    for j in range(0, n_blocks, batch_size):

        this_x = features[j:j + batch_size]

        new_prediction = model.predict(this_x, batch_size=1, verbose=0)

        prev_mean = np.zeros_like(new_prediction['probabilities'])
        curr_mean = new_prediction['probabilities']

        M = np.zeros_like(new_prediction['probabilities'])
        for n in range(1, n_samples):

            new_prediction = model.predict(this_x)
            prev_mean = curr_mean
            curr_mean = prev_mean + (new_prediction['probabilities'] - prev_mean)/float(n+1)
            M = M + np.multiply(prev_mean - new_prediction['probabilities'], curr_mean - new_prediction['probabilities'])

        means[j:j + batch_size] = np.argmax(curr_mean, axis = -1 ) # max mean
        variances[j:j + batch_size] = np.sum(M/n_samples, axis = -1)
        entropies[j:j + batch_size] = -np.sum(np.multiply(np.log(curr_mean+0.001),curr_mean), axis = -1) # entropy
        progbar.add(1)

    total_means = from_blocks_numpy(means, output_shape=inputs.shape)
    total_variance = from_blocks_numpy(variances, output_shape=inputs.shape)
    total_entropy = from_blocks_numpy(entropies, output_shape=inputs.shape)

    mean_var_voxels = np.mean(total_variance)
    std_var_voxels = np.std(total_variance)

    include_variance = ((n_samples > 1) and (return_variance))
    if include_variance:
        if return_entropy:
            return total_means, total_variance, total_entropy
        else:
            return total_means, total_variance
    else:
        if return_entropy:
            return total_means, total_entropy
        else:
            return total_means,


def predict_from_img(img,
                     model,
                     block_shape,
                     return_variance=False,
                     return_entropy=False,
                     return_array_from_images=False,
                     n_samples=1,
                     normalizer=None,
                     batch_size=4,
                     dtype=DT_X):
    """Return a prediction given a Nibabel image instance and a predictor.

    Parameters
    ----------
    img: nibabel image, image on which to predict.
    model: `tf.keras.Model`, trained model.
    block_shape: tuple of len 3, shape of blocks on which to predict.
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.
    return_array_from_images: Boolean. If set True and the given input is either image,
        filepath, or filepaths, it will return arrays of [mean, variance, entropy]
        instead of images of them. Also, if the input is array, it will
        simply return array, whether or not this flag is True or False.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.
    batch_size: int, number of sub-volumes per batch for prediction.
    dtype: str or dtype object, dtype of features.

    Returns
    -------
    `nibabel.spatialimages.SpatialImage` or arrays of prediction of mean,
    variance(optional) or entropy (optional).
    """
    if not isinstance(img, nib.spatialimages.SpatialImage):
        raise ValueError("image is not a nibabel image type")
    inputs = np.asarray(img.dataobj)
    if dtype is not None:
        inputs = inputs.astype(dtype)
    img.uncache()
    y = predict_from_array(
        inputs=inputs,
        model=model,
        block_shape=block_shape,
        return_variance=return_variance,
        return_entropy=return_entropy,
        return_array_from_images=return_array_from_images,
        n_samples=n_samples,
        normalizer=normalizer,
        batch_size=batch_size)

    if return_array_from_images:
        return y
    else:
        if len(y) == 1:
            return nib.spatialimages.SpatialImage(
                dataobj=y[0], affine=img.affine, header=img.header, extra=img.extra),

        elif len(y) == 2:
            return nib.spatialimages.SpatialImage(
                dataobj=y[0], affine=img.affine, header=img.header, extra=img.extra),\
                nib.spatialimages.SpatialImage(
                dataobj=y[1], affine=img.affine, header=img.header, extra=img.extra)
        else:           # 3 inputs
            return nib.spatialimages.SpatialImage(
                dataobj=y[0], affine=img.affine, header=img.header, extra=img.extra),\
                nib.spatialimages.SpatialImage(
                dataobj=y[1], affine=img.affine, header=img.header, extra=img.extra),\
                nib.spatialimages.SpatialImage(
                dataobj=y[2], affine=img.affine, header=img.header, extra=img.extra)


def predict_from_filepath(filepath,
                          model,
                          block_shape,
                          return_variance=False,
                          return_entropy=False,
                          return_array_from_images=False,
                          n_samples=1,
                          normalizer=None,
                          batch_size=4,
                          dtype=DT_X):
    """Return a prediction given a filepath and Predictor object.

    Parameters
    ----------
    filepath: path-like, path to existing neuroimaging volume on which
        to predict.
    model: `tf.keras.Model`, trained model.
    block_shape: tuple of len 3, shape of blocks on which to predict.
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
        along with mean.
    return_array_from_images: Boolean. If set True and the given input is either image,
        filepath, or filepaths, it will return arrays of [mean, variance, entropy]
        instead of images of them. Also, if the input is array, it will
        simply return array, whether or not this flag is True or False.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.

    Returns
    -------
    `nibabel.spatialimages.SpatialImage` or arrays of predictions of
        mean, variance(optional), and entropy (optional).
    """
    if not Path(filepath).is_file():
        raise FileNotFoundError("could not find file {}".format(filepath))
    img = nib.load(filepath)
    return predict_from_img(
        img=img,
        model=model,
        block_shape=block_shape,
        return_variance=return_variance,
        return_entropy=return_entropy,
        return_array_from_images=return_array_from_images,
        n_samples=n_samples,
        normalizer=normalizer,
        batch_size=batch_size)


def predict_from_filepaths(filepaths,
                           predictor,
                           block_shape,
                           return_variance=False,
                           return_entropy=False,
                           return_array_from_images=False,
                           n_samples=1,
                           normalizer=None,
                           batch_size=4,
                           dtype=DT_X):
    """Yield predictions from filepaths using a SavedModel.

    Parameters
    ----------
    filepaths: list, neuroimaging volume filepaths on which to predict.
    model: `tf.keras.Model`, trained model.
    block_shape: tuple of len 3, shape of blocks on which to predict.
    normalizer: callable, function that accepts an ndarray and returns
        an ndarray. Called before separating volume into blocks.
    batch_size: int, number of sub-volumes per batch for prediction.
    dtype: str or dtype object, dtype of features.

    Returns
    -------
    Generator object that yields a `nibabel.spatialimages.SpatialImage` or
    arrays of predictions per filepath in list of input filepaths.
    """
    for filepath in filepaths:
        yield predict_from_filepath(
            filepath=filepath,
            model=model,
            block_shape=block_shape,
            return_variance=return_variance,
            return_entropy=return_entropy,
            return_array_from_images=return_array_from_images,
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size,
            dtype=dtype)


def _get_model(path):
    """Return `tf.keras.Model` object from a filepath.

    Parameters
    ----------
    path: str, path to HDF5 or SavedModel file.

    Returns
    -------
    Instance of `tf.keras.Model`.

    Raises
    ------
    `ValueError` if cannot load model.
    """
    try:
        return tf.keras.models.load_model(path, compile=False)
    except OSError:
        # Not an HDF5 file.
        pass

    try:
        path = Path(path)
        if path.suffix == '.json':
            path = path.parent.parent
        return tf.keras.experimental.load_from_saved_model(str(path))
    except Exception:
        pass

    raise ValueError(
        "Failed to load model. Is the model in HDF5 format or SavedModel"
        " format?")


def _transform_and_predict(model, x, block_shape, rotation, translation=[0, 0, 0]):
    """Predict on rigidly transformed features.

    The rigid transformation is applied to the volumes prior to prediction, and
    the prediced labels are transformed with the inverse warp, so that they are
    in the same space.

    Parameters
    ----------
    model: `tf.keras.Model`, model used for prediction.
    x: 3D array, volume of features.
    block_shape: tuple of length 3, shape of non-overlapping blocks to take
        from the features. This also corresponds to the input of the model, not
        including the batch or channel dimensions.
    rotation: tuple of length 3, rotation angle in radians in each dimension.
    translation: tuple of length 3, units of translation in each dimension.

    Returns
    -------
    Array of predictions with the same shape and in the same space as the
    original input features.
    """

    x = np.asarray(x).astype(np.float32)
    affine = get_affine(x.shape, rotation=rotation, translation=translation)
    inverse_affine = np.linalg.inv(affine)
    x_warped = warp(x, affine, order=1)

    x_warped_blocks = to_blocks_numpy(x_warped, block_shape)
    x_warped_blocks = x_warped_blocks[..., np.newaxis]  # add grayscale channel
    x_warped_blocks = standardize_numpy(x_warped_blocks)
    y = model.predict(x_warped_blocks, batch_size=1)

    n_classes = y.shape[-1]
    if n_classes == 1:
        y = y.squeeze(-1)
    else:
        # Usually, the argmax would be taken to get the class membership of
        # each voxel, but if we get hard values, then we cannot average
        # multiple predictions.
        raise ValueError(
            "This function is not compatible with multi-class predictions.")

    y = from_blocks_numpy(y, x.shape)
    y = warp(y, inverse_affine, order=0)

    return y
