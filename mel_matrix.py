# MODIFIED BY Thomas O'Malley 12/24/2017 to include optional Vocal Tract Length Perturbation
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""mel conversion ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.signal.python.ops import shape_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _mel_to_hertz(mel_values, name=None):
  """Converts frequencies in `mel_values` from the mel scale to linear scale.
  Args:
    mel_values: A `Tensor` of frequencies in the mel scale.
    name: An optional name for the operation.
  Returns:
    A `Tensor` of the same shape and type as `mel_values` containing linear
    scale frequencies in Hertz.
  """
  with ops.name_scope(name, 'mel_to_hertz', [mel_values]):
    mel_values = ops.convert_to_tensor(mel_values)
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        math_ops.exp(mel_values / _MEL_HIGH_FREQUENCY_Q) - 1.0
    )


def _hertz_to_mel(frequencies_hertz, is_training=tf.constant(False),name=None,sampling_rate=16000):
  """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.
  Args:
    frequencies_hertz: A `Tensor` of frequencies in Hertz.
    name: An optional name for the operation.
  Returns:
    A `Tensor` of the same shape and type of `frequencies_hertz` containing
    frequencies in the mel scale.
  """
  vtlp_a = tf.truncated_normal([],1.0,0.1,dtype=tf.float64)
  vtlp_f = tf.cast(4800,tf.float64)
  with ops.name_scope(name, 'hertz_to_mel', [frequencies_hertz]):
    frequencies_hertz = ops.convert_to_tensor(frequencies_hertz)
    def vtlp(frequencies_hertz):
        max_freq = tf.cast(sampling_rate / 2.0,tf.float64)
        vtlp_numerator = max_freq - vtlp_f*tf.minimum(vtlp_a,1)
        vtlp_denominator = max_freq - vtlp_f*tf.minimum(vtlp_a,1)/vtlp_a
        below_cutoff_distortion = vtlp_a*frequencies_hertz
        above_cutoff_distortion = max_freq - (vtlp_numerator/vtlp_denominator)*(max_freq - frequencies_hertz)
        frequencies_hertz = tf.where(frequencies_hertz <= vtlp_f, below_cutoff_distortion, above_cutoff_distortion)
        return frequencies_hertz
    frequencies_hertz = tf.cond(is_training,lambda: vtlp(frequencies_hertz), lambda: tf.identity(frequencies_hertz))
    return _MEL_HIGH_FREQUENCY_Q * math_ops.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def _validate_arguments(num_mel_bins, num_spectrogram_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz, dtype):
  """Checks the inputs to linear_to_mel_weight_matrix."""
  if num_mel_bins <= 0:
    raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
  if num_spectrogram_bins <= 0:
    raise ValueError('num_spectrogram_bins must be positive. Got: %s' %
                     num_spectrogram_bins)
  if sample_rate <= 0.0:
    raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
  if lower_edge_hertz < 0.0:
    raise ValueError('lower_edge_hertz must be non-negative. Got: %s' %
                     lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                     (lower_edge_hertz, upper_edge_hertz))
  if not dtype.is_floating:
    raise ValueError('dtype must be a floating point type. Got: %s' % dtype)


def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=8000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0,
                                is_training=False,
                                dtype=dtypes.float32,
                                name=None):
  """Returns a matrix to warp linear scale spectrograms to the [mel scale][mel].
  Returns a weight matrix that can be used to re-weight a `Tensor` containing
  `num_spectrogram_bins` linearly sampled frequency information from
  `[0, sample_rate / 2]` into `num_mel_bins` frequency information from
  `[lower_edge_hertz, upper_edge_hertz]` on the [mel scale][mel].
  For example, the returned matrix `A` can be used to right-multiply a
  spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear
  scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram"
  `M` of shape `[frames, num_mel_bins]`.
      # `S` has shape [frames, num_spectrogram_bins]
      # `M` has shape [frames, num_mel_bins]
      M = tf.matmul(S, A)
  The matrix can be used with @{tf.tensordot} to convert an arbitrary rank
  `Tensor` of linear-scale spectral bins into the mel scale.
      # S has shape [..., num_spectrogram_bins].
      # M has shape [..., num_mel_bins].
      M = tf.tensordot(S, A, 1)
      # tf.tensordot does not support shape inference for this case yet.
      M.set_shape(S.shape[:-1].concatenate(A.shape[-1:]))
  Args:
    num_mel_bins: Python int. How many bands in the resulting mel spectrum.
    num_spectrogram_bins: Python int. How many bins there are in the source
      spectrogram data, which is understood to be `fft_size // 2 + 1`, i.e. the
      spectrogram only contains the nonredundant FFT bins.
    sample_rate: Python float. Samples per second of the input signal used to
      create the spectrogram. We need this to figure out the actual frequencies
      for each spectrogram bin, which dictates how they are mapped into the mel
      scale.
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
      included in the mel spectrum. This corresponds to the lower edge of the
      lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
      frequency band.
    is_training: if True, implement Vocal Tract Length Perturbation
    vtlp_f: The cutoff frequency Fhi as in Jaitly, Hinton
    dtype: The `DType` of the result matrix. Must be a floating point type.
    name: An optional name for the operation.
  Returns:
    A `Tensor` of shape `[num_spectrogram_bins, num_mel_bins]`.
  Raises:
    ValueError: If num_mel_bins/num_spectrogram_bins/sample_rate are not
      positive, lower_edge_hertz is negative, or frequency edges are incorrectly
      ordered.
  [mel]: https://en.wikipedia.org/wiki/Mel_scale
  """
  with ops.name_scope(name, 'linear_to_mel_weight_matrix') as name:
    _validate_arguments(num_mel_bins, num_spectrogram_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz, dtype)

    # To preserve accuracy, we compute the matrix at float64 precision and then
    # cast to `dtype` at the end. This function can be constant folded by graph
    # optimization since there are no Tensor inputs.
    sample_rate = ops.convert_to_tensor(
        sample_rate, dtypes.float64, name='sample_rate')
    lower_edge_hertz = ops.convert_to_tensor(
        lower_edge_hertz, dtypes.float64, name='lower_edge_hertz')
    upper_edge_hertz = ops.convert_to_tensor(
        upper_edge_hertz, dtypes.float64, name='upper_edge_hertz')
    zero_float64 = ops.convert_to_tensor(0.0, dtypes.float64)

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = math_ops.linspace(
        zero_float64, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:]
    # will apply Vocal Tract Length Perturbation if vltp is not None
    spectrogram_bins_mel = array_ops.expand_dims(
        _hertz_to_mel(linear_frequencies,is_training), 1)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = shape_ops.frame(
        math_ops.linspace(_hertz_to_mel(lower_edge_hertz),
                          _hertz_to_mel(upper_edge_hertz),
                          num_mel_bins + 2), frame_length=3, frame_step=1)

    # Split the triples up and reshape them into [1, num_mel_bins] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = tuple(array_ops.reshape(
        t, [1, num_mel_bins]) for t in array_ops.split(
            band_edges_mel, 3, axis=1))

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = math_ops.maximum(
        zero_float64, math_ops.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    mel_weights_matrix = array_ops.pad(
        mel_weights_matrix, [[bands_to_zero, 0], [0, 0]])

    # Cast to the desired type.
    return math_ops.cast(mel_weights_matrix, dtype, name=name)
