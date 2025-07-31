import math

import numpy as np


def zigzag_order(matrix):
    length, _ = matrix.shape
    result = []
    for counter in range(length * 2 - 1):
        if counter < length:
            row = counter
            column = 0
            while column <= counter:
                result.append(matrix[column][row])
                column += 1
                row -= 1
        else:
            column = counter - length + 1
            row = counter - column
            while column < length:
                result.append(matrix[column][row])
                column += 1
                row -= 1

    return np.array(result)


def rle(array):
    result = []
    idx = 0
    while idx < len(array):
        counter = 0
        while idx != (len(array) - 1) and array[idx] == array[idx + 1]:
            counter += 1
            idx += 1
        result.append((counter + 1, int(array[idx])))
        idx += 1

    return result


def zero_encoding(array):
    result = []
    values = []
    idx = 0

    for num_count, num_value in array:
        if num_value != 0:
            # value not zero
            if idx == (len(array) - 1):
                # last element in the array, add it to the value list, append value list length into final result
                # add everything in the value list
                for i in range(num_count):
                    values.append(num_value)
                result.append(len(values) * -1)
                result += values
                break

            # add non zero value into the value list as specified in the rle tuple's number count
            for i in range(num_count):
                values.append(num_value)

        else:
            # value is zero
            if idx == (len(array) - 1):
                # last element in the array, add value list length and everything in it if it's not empty
                # append final zero in the result
                if len(values) != 0:
                    result.append(len(values) * -1)
                    result += values
                result.append(0)
                break

            elif num_count < 4:
                # not enough zeros, treat as non zero
                for i in range(num_count):
                    values.append(num_value)

            else:
                # enough of zeros, add value list length and everything in it if it's not empty
                # append number of zeros
                if len(values) != 0:
                    result.append(len(values) * -1)
                    result += values
                    values = []
                result.append(num_count)

        # Tracking last element
        idx += 1

    return result


def entropy_encode(residuals):
    zigzag_residual = zigzag_order(residuals)
    residual_rle = rle(zigzag_residual)
    result = zero_encoding(residual_rle)
    return result


def zero_decoding(encoded_array):
    result = []
    idx = 0

    # Ensure encoded_array is a flat list
    if isinstance(encoded_array, np.ndarray):
        encoded_array = encoded_array.tolist()

    while idx < len(encoded_array):
        count = encoded_array[idx]
        idx += 1

        if isinstance(count, list):
            # Convert any lists within encoded_array to integers
            count = int(count[0])

        if count < 0:
            # Negative count indicates a run of non-zero values
            count = -count
            for _ in range(count):
                # Append each non-zero value as (1, value)
                result.append((1, encoded_array[idx]))
                idx += 1
        elif count == 0:
            # End of sequence marker
            break
        else:
            # Positive count represents a run of zeros
            result.append((count, 0))

    return result


def rle_decode(rle_array):
    result = []
    for count, value in rle_array:
        result.extend([value] * count)
    return np.array(result)


def inverse_zigzag_order(array, size):
    matrix = np.zeros((size, size), dtype=int)
    length = size
    idx = 0

    for counter in range(length * 2 - 1):
        if counter < length:
            row = counter
            column = 0
            while column <= counter and idx < len(array):
                if 0 <= row < length and 0 <= column < length:
                    matrix[column][row] = array[idx]
                    idx += 1
                column += 1
                row -= 1
        else:
            column = counter - length + 1
            row = counter - column
            while column < length and idx < len(array):
                if 0 <= row < length and 0 <= column < length:
                    matrix[column][row] = array[idx]
                    idx += 1
                column += 1
                row -= 1
    return matrix


def entropy_decode(encoded_array, size):
    rle_decoded = zero_decoding(encoded_array)
    zigzag_decoded = rle_decode(rle_decoded)
    original_matrix = inverse_zigzag_order(zigzag_decoded, size)
    return original_matrix


def exp_golomb_encode(value):
    if value <= 0:
        value = (-value) * 2
    else:
        value = value * 2 - 1

    value = value + 1
    binary_value = bin(value)[2:]

    num_bits = len(binary_value)

    prefix_zeros = '0' * (num_bits - 1)

    exp_golomb_code = prefix_zeros + binary_value

    return exp_golomb_code


def exp_golomb_decode(encoded_value):
    int_value = int(encoded_value, 2)
    # decoded_value = 0

    # int value is odd
    if int_value % 2 != 0:
        int_value -= 1
        decoded_value = int_value // -2
    else:
        decoded_value = int_value // 2

    return decoded_value


def differential_encode_qp(qp_values):
    encoded_qp = []
    prev_qp = 5
    for qp in qp_values:
        encoded_qp.append(exp_golomb_encode(qp - prev_qp))
        prev_qp = qp
    return encoded_qp


def differential_decode_qp(encoded_qp):
    decoded_qp = []
    prev_qp = 5
    for encoded in encoded_qp:
        qp = prev_qp + exp_golomb_decode(encoded)
        decoded_qp.append(qp)
        prev_qp = qp
    return decoded_qp


def differential_encode_modes(modes_per_frame):
    previous_mode = 0
    encoded_modes = []

    for mode in modes_per_frame:
        diff = mode - previous_mode
        encoded_modes.append(diff)
        previous_mode = mode

    return encoded_modes


def differential_decode_modes(encoded_modes):
    previous_mode = 0
    decoded_modes = []

    for diff in encoded_modes:
        current_mode = previous_mode + diff
        decoded_modes.append(current_mode)
        previous_mode = current_mode

    return decoded_modes


def differential_encode_motion_vectors(motion_vectors_per_frame):
    prev_frame_idx, prev_mv_x, prev_mv_y = 0, 0, 0
    diff_encoded_mvs = []

    for block in motion_vectors_per_frame:
        frame_idx = block[0]
        mv_x = block[1]
        mv_y = block[2]
        diff_frame_idx = frame_idx - prev_frame_idx
        diff_mv_x = mv_x - prev_mv_x
        diff_mv_y = mv_y - prev_mv_y

        diff_encoded_mvs.append([diff_frame_idx, diff_mv_x, diff_mv_y])

        prev_frame_idx, prev_mv_x, prev_mv_y = frame_idx, mv_x, mv_y

    return diff_encoded_mvs


def differential_decode_motion_vectors(diff_encoded_mvs):
    prev_frame_idx, prev_mv_x, prev_mv_y = 0, 0, 0
    decoded_mvs = []

    for diff_mv in diff_encoded_mvs:
        diff_frame_idx, diff_mv_x, diff_mv_y = diff_mv

        frame_idx = prev_frame_idx + diff_frame_idx
        mv_x = prev_mv_x + diff_mv_x
        mv_y = prev_mv_y + diff_mv_y

        decoded_mvs.append((frame_idx, mv_x, mv_y))
        prev_frame_idx, prev_mv_x, prev_mv_y = frame_idx, mv_x, mv_y

    return decoded_mvs


def write_qp_to_bitstream(qp_list, filepath):
    with open(filepath, 'w') as f:
        for qp_row in qp_list:
            encoded_qp = differential_encode_qp(qp_row)
            f.write(' '.join(encoded_qp) + '\n')

    print(f"Encoded QPs written to {filepath}")


def read_qp_from_bitstream(filepath):
    qp_table = []
    with open(filepath, 'r') as f:
        for line in f:
            encoded_qp = line.strip().split()
            qp_table.append(differential_decode_qp(encoded_qp))
    return qp_table


def write_modes_to_bitstream(modes, filepath):
    with open(filepath, 'w') as f:
        for modes_per_frame in modes:
            encoded_modes = [exp_golomb_encode(mode) for mode in modes_per_frame]
            encoded_string = ' '.join(encoded_modes)
            f.write(encoded_string + '\n')

    print(f"Encoded modes written to {filepath}")


def read_modes_from_bitstream(filepath):
    modes = []
    with open(filepath, 'r') as f:
        for line in f:
            encoded_modes = line.strip().split(' ')
            decoded_modes = [exp_golomb_decode(mode) for mode in encoded_modes]
            decoded_modes = differential_decode_modes(decoded_modes)
            modes.append(decoded_modes)
    return modes


def write_mvs_to_bitstream(mvs, filepath):
    with open(filepath, 'w') as f:
        for mv_per_frame in mvs:
            encoded_mv = []
            for mv in mv_per_frame:
                reference_frame_idx_encoded = exp_golomb_encode(mv[0])
                dx_encoded = exp_golomb_encode(mv[1])
                dy_encoded = exp_golomb_encode(mv[2])
                encoded_mv.append(f'{reference_frame_idx_encoded} {dx_encoded} {dy_encoded}')

            f.write(' '.join(encoded_mv) + '\n')

    print(f"Encoded motion vectors written to {filepath}")


def read_mvs_from_bitstream(filepath):
    mvs = []

    with open(filepath, 'r') as f:
        for line in f:
            encoded_mv = line.strip().split(' ')
            diff_encoded_mvs = []

            for i in range(0, len(encoded_mv), 3):
                reference_frame_idx = exp_golomb_decode(encoded_mv[i])
                dx = exp_golomb_decode(encoded_mv[i + 1])
                dy = exp_golomb_decode(encoded_mv[i + 2])

                diff_encoded_mvs.append((reference_frame_idx, dx, dy))

            decoded_mvs = differential_decode_motion_vectors(diff_encoded_mvs)
            mvs.append(decoded_mvs)

    return mvs


def write_residuals_to_bitstream(residuals, filepath):
    with open(filepath, 'w') as f:
        for residuals_per_frame in residuals:
            encoded_residuals = []
            for split_sign, residual_block in residuals_per_frame:

                encoded_split_sign = exp_golomb_encode(split_sign)

                tmp = entropy_encode(residual_block)
                encoded_residual = [exp_golomb_encode(i) for i in tmp]

                encoded_residuals.append(f'{encoded_split_sign}-{" ".join(encoded_residual)}')

            encoded_string = '|'.join(encoded_residuals)
            f.write(f'{encoded_string}\n')

    print(f"Encoded residuals written to {filepath}")


def read_residuals_from_bitstream(filepath, block_size):
    residuals = []

    with open(filepath, 'r') as f:
        for line in f:
            encoded_residuals = line.strip().split('|')
            residuals_per_frame = []

            for encoded_residual in encoded_residuals:
                split_sign_encoded, residual_encoded = encoded_residual.split('-')

                split_sign = exp_golomb_decode(split_sign_encoded)
                if split_sign == 0:
                    size = block_size
                else:
                    size = block_size // 2

                residual_values = [exp_golomb_decode(val) for val in residual_encoded.split()]
                residual_block = entropy_decode(residual_values, size)

                residuals_per_frame.append((split_sign, residual_block))

            residuals.append(residuals_per_frame)

    print(f"Decoded residuals read from {filepath}")
    return residuals


def write_ip_list_as_binary(boolean_list, file_path):
    binary_string = ''.join('1' if value else '0' for value in boolean_list)
    padded_binary_string = binary_string.ljust((len(binary_string) + 7) // 8 * 8, '0')
    byte_data = int(padded_binary_string, 2).to_bytes(len(padded_binary_string) // 8, byteorder='big')
    with open(file_path, 'wb') as binary_file:
        binary_file.write(byte_data)

    print(f"Encoded ip-frame flags saved as binary to {file_path}")


def read_ip_file_as_list(file_path):
    with open(file_path, 'rb') as binary_file:
        byte_data = binary_file.read()
    binary_string = ''.join(f"{byte:08b}" for byte in byte_data)
    boolean_list = [char == '1' for char in binary_string]
    
    return boolean_list

