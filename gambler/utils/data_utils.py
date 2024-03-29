import numpy as np
import os.path
import h5py
import socket
import time
import math
from typing import List, Union, Tuple, Iterable
from argparse import ArgumentParser
from collections import namedtuple, Counter
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import Optional, List, Tuple


def reconstruct_sequence(measurements: np.ndarray, collected_indices: List[int], seq_length: int) -> np.ndarray:
    """
    Reconstructs a sequence using a linear interpolation.
    Args:
        measurements: A [K, D] array of sub-sampled features
        collected_indices: A list of [K] indices of the collected features
        seq_length: The length of the full sequence (T)
    Returns:
        A [T, D] array of reconstructed measurements.
    """
    feature_list: List[np.ndarray] = []
    seq_idx = list(range(seq_length))

    # Interpolate the unseen measurements using a linear function
    for feature_idx in range(measurements.shape[1]):
        collected_features = measurements[:, feature_idx]  # [K]
        reconstructed = np.interp(x=seq_idx,
                                  xp=collected_indices,
                                  fp=collected_features,
                                  left=collected_features[0],
                                  right=collected_features[-1])

        feature_list.append(np.expand_dims(reconstructed, axis=-1))

    return np.concatenate(feature_list, axis=-1)  # [T, D]


MAX_ITER = 100
ERROR_TOL = 0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-1 * x))


def to_fixed_point(x: float, precision: int, width: int) -> int:
    assert width >= 1, 'Must have a non-negative width'

    multiplier = 1 << abs(precision)
    fp = int(round(x * multiplier)) if precision > 0 else int(round(x / multiplier))

    max_val = (1 << (width - 1)) - 1
    min_val = -max_val

    if fp > max_val:
        return max_val
    elif fp < min_val:
        return min_val
    return fp


def to_float(fp: int, precision: int) -> float:
    multiplier = float(1 << abs(precision))
    return float(fp) / multiplier if precision > 0 else float(fp) * multiplier


def array_to_fp(arr: np.ndarray, precision: int, width: int) -> np.ndarray:
    multiplier = 1 << abs(precision)

    if precision > 0:
        quantized = arr * multiplier
    else:
        quantized = arr / multiplier

    quantized = np.round(quantized).astype(int)

    max_val = (1 << (width - 1)) - 1
    min_val = -max_val

    return np.clip(quantized, a_min=min_val, a_max=max_val)


def array_to_fp_shifted(arr: np.ndarray, precision: int, width: int, shifts: np.ndarray) -> np.ndarray:
    assert len(arr.shape) == 1, 'Must provide a 1d array'
    assert arr.shape == shifts.shape, 'Misaligned data {0} and shifts {1}'.format(arr.shape, shifts.shape)

    shifted_precisions = precision - shifts
    multipliers = np.left_shift(np.ones_like(shifts), np.abs(shifted_precisions))

    quantized = np.where(shifted_precisions > 0, arr * multipliers, arr / multipliers)
    quantized = np.round(quantized).astype(int)

    max_val = (1 << (width - 1)) - 1
    min_val = -max_val

    return np.clip(quantized, a_min=min_val, a_max=max_val)


def array_to_float(fp_arr: Union[np.ndarray, List[int]], precision: int) -> np.ndarray:
    multiplier = float(1 << abs(precision))

    if isinstance(fp_arr, list):
        fp_arr = np.array(fp_arr)

    if precision > 0:
        return fp_arr.astype(float) / multiplier
    else:
        return fp_arr.astype(float) * multiplier


def array_to_float_shifted(arr: Union[np.ndarray, List[int]], precision: int, shifts: np.ndarray) -> np.ndarray:
    shifted_precisions = precision - shifts
    multipliers = np.left_shift(np.ones_like(shifts), np.abs(shifted_precisions))

    if isinstance(arr, list):
        fp_arr = np.array(arr).astype(float)
    else:
        fp_arr = arr.astype(float)

    assert len(fp_arr.shape) == 1, 'Must provide a 1d array'
    assert fp_arr.shape == shifts.shape, 'Misaligned data {0} and shifts {1}'.format(fp_arr.shape, shifts.shape)

    recovered = np.where(shifted_precisions > 0, fp_arr / multipliers, fp_arr * multipliers)

    return recovered


def select_range_shift(measurement: int, old_width: int, old_precision: int, new_width: int, num_range_bits: int, prev_shift: int) -> int:
    """
    Selects the lowest-error range multiplier.
    Args:
        measurement: The fixed point measurement feature
        old_width: The standard width of each feature
        old_precision: The existing precision of each feature
        new_width: The quantized width of each feature (this is how we will express values)
        num_range_bits: The number of bits for the range exponent
        prev_shift: The previous range shift (to have better tie-breaking)
    Returns:
        The range exponent in [-2^{range_bits - 1}, 2^{range_bits - 1}]
    """
    assert num_range_bits >= 1, 'Number of range bits must be non-negative'
    assert (old_width >= 1) and (new_width >= 1), 'Number of width bits must be non-negative'

    # Create the constants necessary for selecting the range shift
    non_fractional = old_width - old_precision
    width_mask = (1 << (new_width - 1)) - 1  # Masks out all non-data bits (including the sign bit)
    recovered_mask = (1 << (old_width - 1)) - 1  # Mask out all non-data bits in the old size (including sign bit)

    # Get base shift value using the width differences
    base_shift = old_width - new_width

    # Get the absolute value of the given fixed point measurement. The routine
    # performs all operations on positive values for simplicity.
    abs_value = abs(measurement)

    # Try the previous shift first for potential early-exiting
    conversion_shift = base_shift + prev_shift

    if (conversion_shift >= 0):
        quantized = (abs_value >> conversion_shift) & width_mask
        recovered = (quantized << conversion_shift)
    else:
        conversion_shift *= -1
        quantized = (abs_value << conversion_shift) & width_mask
        recovered = (quantized >> conversion_shift)

    recovered &= recovered_mask

    prev_error = abs(abs_value - recovered)

    if (prev_error <= ERROR_TOL):
        return prev_shift

    # Perform an exhaustive search over all potential shifts
    last_error = BIG_NUMBER
    best_error = prev_error
    best_shift = prev_shift

    offset = (1 << (num_range_bits - 1))
    limit = 1 << num_range_bits

    for idx in range(limit):
        shift = idx - offset

        # Convert the value to and from floating point
        conversion_shift = base_shift + shift

        if (conversion_shift >= 0):
            quantized = (abs_value >> conversion_shift) & width_mask
            recovered = (quantized << conversion_shift)
        else:
            conversion_shift *= -1
            quantized = (abs_value << conversion_shift) & width_mask
            recovered = (quantized >> conversion_shift)

        recovered &= recovered_mask

        # Compute the error and save the best result
        error = abs(abs_value - recovered)

        if (error < best_error):
            best_shift = shift
            best_error = error

        if (best_error == 0):
            break  # Stop if we ever get zero error (can't do any better)

        last_error = error

    if (prev_error <= best_error):
        return prev_shift

    return best_shift


def select_range_shifts_array(measurements: np.ndarray, old_width: int, old_precision: int, new_width: int, num_range_bits: int) -> np.ndarray:
    """
    Selects the lowest-error range multiplier.
    Args:
        measurements: A 1d array of measurement features
        old_width: The standard width of existing features
        old_precision: The precision of existing features
        new_width: The new width of each feature (this is how values will be encoded)
        precision: The precision of each feature
        num_range_bits: The number of bits for the range exponent
    Returns:
        The range exponent in [-2^{range_bits - 1}, 2^{range_bits - 1}]
    """
    assert num_range_bits >= 1, 'Number of range bits must be non-negative'
    assert (old_width >= 1) and (new_width >= 1), 'Number of width bits must be non-negative'
    assert len(measurements.shape) == 1, 'Must provide a 1d numpy array'

    # Convert all values to fixed point
    fp_values = array_to_fp(measurements, width=old_width, precision=old_precision)

    num_values = measurements.shape[0]
    best_shifts = np.empty(num_values)
    prev_shift = -1 * (1 << (num_range_bits - 1))

    elapsed = []

    for idx in range(num_values):
        shift = select_range_shift(measurement=fp_values[idx],
                                   old_width=old_width,
                                   old_precision=old_precision,
                                   new_width=new_width,
                                   num_range_bits=num_range_bits,
                                   prev_shift=prev_shift)

        best_shifts[idx] = int(shift)
        prev_shift = shift

    return best_shifts.astype(int)


def linear_extrapolate(prev: np.ndarray, curr: np.ndarray, delta: float, num_steps: int) -> np.ndarray:
    """
    This function uses a linear approximation over the given readings to project
    the next value 'delta' units ahead.
    Args:
        prev: A [D] array containing the previous value
        curr: A [D] array containing the current value
        delta: The time between consecutive readings
        num_steps: The number of steps to extrapolate ahead
    Returns:
        A [D] array containing the projected reading (`num_steps * delta` steps ahead).
    """
    slope = (curr - prev) / delta
    return slope * delta * num_steps + curr


def pad_to_length(message: bytes, length: int) -> bytes:
    """
    Pads larger messages to the given length by appending
    random bytes.
    """
    if len(message) >= length:
        return message

    padding = get_random_bytes(length - len(message))
    return message + padding


def round_to_block(length: Union[int, float], block_size: int) -> int:
    """
    Rounds the given length to the nearest (larger) multiple of
    the block size.
    """
    if isinstance(length, int) and (length % block_size == 0):
        return length

    return int(math.ceil(length / block_size)) * block_size


def truncate_to_block(length: Union[int, float], block_size: int) -> int:
    """
    Rounds the given length to the nearest (smaller) multiple of
    the block size.
    """
    return int(math.floor(length / block_size)) * block_size


def set_widths(group_sizes: List[int], is_all_zero: List[bool], target_bytes: int, start_width: int, max_width: int) -> List[int]:
    """
    Sets the group widths in a round-robin fashion
    to saturate the target bytes.
    Args:
        group_sizes: The size (in number of features) of each group
        is_all_zero: Whether the group is all zero values (true) or not (false)
        target_bytes: The target number of data bytes
        start_width: The starting number of bits per feature
        max_width: The maximum width of a single group
    Returns:
        A list of the bit widths for each group
    """
    num_groups = len(group_sizes)
    num_values = sum(group_sizes)
    target_bits = target_bytes * BITS_PER_BYTE

    # Set the initial widths based on an even distribution
    consumed_bytes = sum((int(math.ceil((start_width * size) / BITS_PER_BYTE)) for size in group_sizes))

    start_widths = min(start_width, max_width)
    widths: List[int] = [start_width for _ in range(num_groups)]

    if start_width >= max_width:
        return widths

    counter = 0
    has_improved = True
    while (has_improved and counter < MAX_ITER):

        has_improved = False

        for idx in range(len(group_sizes)):
            if (widths[idx] == max_width):
                continue
            elif (is_all_zero[idx]):
                widths[idx] = MIN_WIDTH
                continue

            # Increase this group's width by 1 bit
            widths[idx] += 1

            # Calculate the new number of data bytes
            candidate_bytes = sum((int(math.ceil((w * size) / BITS_PER_BYTE)) for w, size in zip(widths, group_sizes)))

            if (candidate_bytes <= target_bytes):
                consumed_bytes = candidate_bytes
                has_improved = True
            else:
                widths[idx] -= 1

        counter += 1

    return widths


def prune_sequence(measurements: np.ndarray, collected_indices: List[int], max_collected: int, seq_length: int) -> Tuple[np.ndarray, List[int]]:
    """
    Prunes the given sequence to use at most the maximum number
    of measurements. We remove measurements that induce the approximate lowest
    amount of additional error.
    Args:
        measurements: A [K, D] array of collected measurement vectors
        collected_indices: A list of [K] indices of the collected measurements
        max_collected: The maximum number of allowed measurements
        seq_length: The full sequence length
    Returns:
        A tuple of two elements:
            (1) A [K', D] array of the pruned measurements
            (2) A [K'] list of the remaining indices
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d array of measurements'
    assert measurements.shape[0] == len(collected_indices), 'Misaligned measurements ({0}) and collected indices ({1})'.format(measurements.shape[0], len(collected_indices))

    # Avoid pruning measurements which are under budget
    num_collected = len(collected_indices)
    if num_collected <= max_collected:
        return measurements, collected_indices

    # Compute the differences in measurements
    first = measurements[:-1]  # [L - 1, D]
    last = measurements[1:]  # [L - 1, D]
    measurement_diff = np.sum(np.abs(last - first), axis=-1).astype(int)  # [L - 1]

    # Compute the two-step differences in indices
    idx_diff = np.array([(collected_indices[i+1] - collected_indices[i]) for i in range(len(collected_indices) - 1)])  # [L - 1]

    scores = measurement_diff + (0.125 * idx_diff)  # [L - 1]
    scores = scores[:-1]  # [L - 2]

    num_to_prune = len(measurements) - max_collected

    if num_to_prune >= len(scores):
        to_remove_set = set(i + 1 for i in range(len(scores)))
    else:
        idx_to_prune = np.argpartition(scores, num_to_prune)[0:num_to_prune]
        idx_to_prune += 1

        # Remove the given elements from the measurements and collected lists
        to_remove_set = set(idx_to_prune)

    idx_to_keep = [i for i in range(len(measurements)) if i not in to_remove_set]

    pruned_measurements = measurements[idx_to_keep]
    pruned_indices = [collected_indices[i] for i in idx_to_keep]

    return pruned_measurements, pruned_indices


def create_groups(measurements: np.ndarray, max_num_groups: int, max_group_size: int) -> List[np.ndarray]:
    """
    Creates measurement groups using a greedy algorithm based on similar signs.
    Args:
        measurements: A [K, D] array of measurements
        max_num_groups: The maximum number of groups (L)
        max_group_size: The maximum number of features in a group
    Returns:
        A list of 1d arrays of flattened measurements
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d measurements array'

    # Flatten the features into a 1d array (feature-wise)
    flattened = measurements.T.reshape(-1)

    min_group_size = int(math.ceil(len(flattened) / max_num_groups))

    groups: List[np.ndarray] = []
    current_idx = 0
    current_size = min_group_size

    indices = np.arange(len(flattened))
    signs = (np.greater_equal(flattened, 0)).astype(float)

    while (current_idx < len(flattened)):
        end_idx = current_idx + current_size

        if (end_idx > len(flattened)):
            groups.append(flattened[current_idx:])
            break

        is_positive = np.all(flattened[current_idx:end_idx] > 0)
        is_negative = np.all(flattened[current_idx:end_idx] < 0)

        if (is_positive or is_negative):
            end_idx += 1

            while (end_idx < len(flattened)) and ((is_positive and flattened[end_idx] >= 0) or (is_negative and flattened[end_idx] <= 0)):
                end_idx += 1

        groups.append(flattened[current_idx:end_idx])

        current_size = end_idx - current_idx
        current_idx += current_size
        current_size = min_group_size

    return groups


def combine_groups(groups: List[np.ndarray], num_features: int) -> np.ndarray:
    """
    Combines the given groups back into a 2d measurement matrix.
    Args:
        groups: A list of 1d, flattened groups
        num_features: The number of features in each measurement (D)
    Returns:
        A [K, D] array containing the recovered measurements.
    """
    flattened = np.concatenate(groups)  # [K * D]
    return flattened.reshape(num_features, -1).T


def integer_part(x: float) -> int:
    """
    Returns the integer part of the given number.
    """
    return int(math.modf(x)[1])


def fractional_part(x: float) -> float:
    """
    Returns the fractional part of the given number
    """
    return math.modf(x)[0]


def get_signs(array: List[int]) -> List[int]:
    """
    Returns a binary array of the signs of each value.
    """
    return [1 if a >= 0 else 0 for a in array]


def apply_signs(array: List[int], signs: List[int]) -> List[int]:
    """
    Applies the signs to the given (absolute value) array.
    """
    assert len(array) == len(signs), 'Misaligned inputs ({0} vs {1})'.format(len(array), len(signs))
    return [x * (2 * s - 1) for x, s in zip(array, signs)]


def fixed_point_integer_part(fixed_point_val: int, precision: int) -> int:
    """
    Extracts the integer part from the given fixed point value.
    """
    if (precision >= 0):
        return fixed_point_val >> precision

    return fixed_point_val << precision


def fixed_point_frac_part(fixed_point_val: int, precision: int) -> int:
    """
    Extracts the fractional part from the given fixed point value.
    """
    if (precision >= 0):
        mask = (1 << precision) - 1
        return fixed_point_val & mask

    return 0


def num_bits_for_value(x: int) -> int:
    """
    Calculates the number if bits required to
    represent the given integer.
    """
    num_bits = 0
    while (x != 0):
        x = x >> 1
        num_bits += 1

    return max(num_bits, 1)


def run_length_encode(values: List[int], signs: List[int]) -> str:
    if len(values) <= 0:
        return ''

    current = abs(values[0])
    current_count = 1
    current_sign = signs[0]

    encoded: List[int] = []
    compressed_signs: List[int] = []
    reps: List[int] = []

    for i in range(1, len(values)):
        val = abs(values[i])

        if (val != current) or (signs[i] != current_sign):
            encoded.append(current)
            reps.append(current_count)
            compressed_signs.append(current_sign)

            current = val
            current_count = 1
            current_sign = signs[i]
        else:
            current_count += 1

    # Always include the final element
    encoded.append(current)
    reps.append(current_count)
    compressed_signs.append(current_sign)

    # Calculate the maximum number of bits needed to encode the values and repetitions
    max_encoded = np.max(np.abs(encoded))
    max_reps = np.max(np.abs(reps))

    encoded_bits = num_bits_for_value(max_encoded)
    reps_bits = num_bits_for_value(max_reps)

    encoded_values = pack(encoded, width=encoded_bits)
    encoded_reps = pack(reps, width=reps_bits)
    encoded_signs = pack(compressed_signs, width=1)

    metadata = ((encoded_bits << 4) | (reps_bits & 0xF)) & 0xFF
    metadata = ((len(encoded) << 8) | metadata) & 0xFFFFFF

    metadata_bytes = metadata.to_bytes(3, 'little')

    return metadata_bytes + encoded_values + encoded_reps + encoded_signs


def run_length_decode(encoded: bytes) -> List[int]:
    """
    Decodes the given RLE values.
    """
    metadata = int.from_bytes(encoded[0:3], 'little')
    encoded = encoded[3:]

    num_values = (metadata >> 8) & 0xFFF
    value_bits = (metadata >> 4) & 0xF
    rep_bits = metadata & 0xF

    value_bytes = int(math.ceil((num_values * value_bits) / BITS_PER_BYTE))
    rep_bytes = int(math.ceil((num_values * rep_bits) / BITS_PER_BYTE))
    sign_bytes = int(math.ceil(num_values / BITS_PER_BYTE))

    decoded_values = unpack(encoded[0:value_bytes], width=value_bits, num_values=num_values)
    encoded = encoded[value_bytes:]

    decoded_reps = unpack(encoded[0:rep_bytes], width=rep_bits, num_values=num_values)
    encoded = encoded[rep_bytes:]

    decoded_signs = unpack(encoded[0:sign_bytes], width=1, num_values=num_values)

    values: List[int] = []
    signs: List[int] = []

    for i in range(num_values):
        for j in range(decoded_reps[i]):
            values.append(decoded_values[i])
            signs.append(decoded_signs[i])

    return values, signs


def pack(values: List[int], width: int) -> bytes:
    """
    Packs the list of (quantized) values with the given width
    into a packed bit-string.
    Args:
        values: The list of quantized values
        width: The width of each quantized value
    Returns:
        A packed string containing the quantized values.
    """
    packed: List[int] = [0]
    consumed = 0
    num_bytes = int(math.ceil(width / 8))

    for value in values:
        for i in range(num_bytes):
            # Get the current byte
            current_byte = (value >> (i * 8)) & 0xFF

            # Get the number of used bits in the current byte
            if i < (num_bytes - 1) or (width % 8) == 0:
                num_bits = 8
            else:
                num_bits = width % 8

            # Set bits in the packed string
            packed[-1] |= current_byte << consumed
            packed[-1] &= 0xFF

            # Add to the number of consumed bits
            used_bits = min(8 - consumed, num_bits)
            consumed += num_bits

            # If we have consumed more than a byte, then the remaining amount
            # spills onto the next byte
            if consumed > 8:
                consumed = consumed - 8
                remaining_value = current_byte >> used_bits

                # Add the remaining value to the running string
                packed.append(remaining_value)

    return bytes(packed)


def unpack(encoded: bytes, width: int,  num_values: int) -> List[int]:
    """
    Unpacks the encoded values into a list of integers of the given bit-width.
    Args:
        encoded: The encoded list of values (output of pack())
        width: The bit width for each value
        num_value: The number of encoded values
    Returns:
        A list of integer values
    """
    result: List[int] = []
    current = 0
    current_length = 0
    byte_idx = 0
    mask = (1 << width) - 1

    for i in range(num_values):
        # Get at at least the next 'width' bits
        while (current_length < width):
            current |= (encoded[byte_idx] << current_length)
            current_length += 8
            byte_idx += 1

        # Truncate down to 'width' bits
        value = current & mask
        result.append(value)

        current = current >> width
        current_length = current_length - width

    # Include any residual values
    if len(result) < num_values:
        result.append(current)

    return result
