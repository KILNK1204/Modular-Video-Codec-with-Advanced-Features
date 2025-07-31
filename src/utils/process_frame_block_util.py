import numpy as np
import math
from scipy.fftpack import dct, idct
from src.utils.quality_assment_util import *
from src.utils.rc_config import *


def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def get_quantization_matrix(block_size, QP):
    Q = np.zeros((block_size, block_size), dtype=int)
    for x in range(block_size):
        for y in range(block_size):
            if x + y < block_size - 1:
                Q[x, y] = 2 ** QP
            elif x + y == block_size - 1:
                Q[x, y] = 2 ** (QP + 1)
            else:
                Q[x, y] = 2 ** (QP + 2)
    return Q


def quantize_dct(dct_block, Q):
    quantized_block = np.round(dct_block / Q).astype(np.int32)
    return quantized_block


def frame_into_blocks(y_frame, block_size):
    height, width = y_frame.shape
    blocks = []

    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            block = y_frame[row:row + block_size, col:col + block_size]
            blocks.append(block)

    return blocks


def pad_frame(y_frame, block_size):
    height, width = y_frame.shape

    padded_height = ((height + block_size - 1) // block_size) * block_size
    padded_width = ((width + block_size - 1) // block_size) * block_size

    padded_frame = np.full((padded_height, padded_width), 128, dtype=np.uint8)

    padded_frame[:height, :width] = y_frame

    return padded_frame


def integer_pixel_full_search(current_block, reference_frame, block_size, search_range, current_x, current_y,
                              fast_me=False, mvp=(0, 0)):
    height, width = reference_frame.shape
    best_mae = float('inf')
    best_mv = (0, 0)
    prediction_block = None

    if fast_me:
        search_candidates = [
            (mvp[0], mvp[1]),
            (mvp[0] - 1, mvp[1]), (mvp[0] + 1, mvp[1]),
            (mvp[0], mvp[1] - 1), (mvp[0], mvp[1] + 1)]
    else:
        search_candidates = [
            (dx, dy) for dy in range(-search_range, search_range + 1)
            for dx in range(-search_range, search_range + 1)]

    for dx, dy in search_candidates:
        ref_y = current_y + dy
        ref_x = current_x + dx

        if ref_x < 0 or ref_y < 0 or ref_x + block_size > width or ref_y + block_size > height:
            continue

        ref_block = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

        mae = get_mae(current_block, ref_block)

        best_mae, best_mv, prediction_block = update_best_mv(
            mae, dx, dy, best_mae, best_mv, ref_block, prediction_block
        )

    return best_mv, best_mae, prediction_block


def update_best_mv(mae, dx, dy, best_mae, best_mv, ref_block, prediction_block):
    if mae < best_mae:
        return mae, (dx, dy), ref_block
    elif mae == best_mae:
        current_mv_norm = abs(dx) + abs(dy)
        best_mv_norm = abs(best_mv[0]) + abs(best_mv[1])
        if current_mv_norm < best_mv_norm or (current_mv_norm == best_mv_norm and (
                dy < best_mv[1] or (dy == best_mv[1] and dx < best_mv[0]))):
            return mae, (dx, dy), ref_block
    return best_mae, best_mv, prediction_block


def interpolate_reference_block(reference_frame, ref_x, ref_y, block_size):
    int_x = int(ref_x)
    int_y = int(ref_y)
    dx = ref_x - int_x
    dy = ref_y - int_y

    height, width = reference_frame.shape

    p00 = reference_frame[int_y:int_y + block_size,
          int_x:int_x + block_size] if int_x >= 0 and int_y >= 0 and int_x + block_size <= width and int_y + block_size <= height else np.full(
        (block_size, block_size), 128, dtype=np.uint8)
    p01 = reference_frame[int_y:int_y + block_size,
          int_x + 1:int_x + 1 + block_size] if int_x + 1 >= 0 and int_y >= 0 and int_x + 1 + block_size <= width and int_y + block_size <= height else np.full(
        (block_size, block_size), 128, dtype=np.uint8)
    p10 = reference_frame[int_y + 1:int_y + 1 + block_size,
          int_x:int_x + block_size] if int_x >= 0 and int_y + 1 >= 0 and int_x + block_size <= width and int_y + 1 + block_size <= height else np.full(
        (block_size, block_size), 128, dtype=np.uint8)
    p11 = reference_frame[int_y + 1:int_y + 1 + block_size,
          int_x + 1:int_x + 1 + block_size] if int_x + 1 >= 0 and int_y + 1 >= 0 and int_x + 1 + block_size <= width and int_y + 1 + block_size <= height else np.full(
        (block_size, block_size), 128, dtype=np.uint8)

    interpolated_block = np.ceil((1 - dx) * (1 - dy) * p00 +
                                 dx * (1 - dy) * p01 +
                                 (1 - dx) * dy * p10 +
                                 dx * dy * p11).astype(reference_frame.dtype)
    return interpolated_block


def fractional_pixel_full_search(current_block, reference_frame, block_size, search_range, current_x, current_y,
                                 fast_me=False, mvp=(0, 0)):
    height, width = reference_frame.shape
    best_mae = float('inf')
    best_mv = (0, 0)
    prediction_block = None

    if fast_me:
        search_candidates = [
            (mvp[0] * 2, mvp[1] * 2),
            (mvp[0] * 2 - 1, mvp[1] * 2), (mvp[0] * 2 + 1, mvp[1] * 2),
            (mvp[0] * 2, mvp[1] * 2 - 1), (mvp[0] * 2, mvp[1] * 2 + 1)
        ]
    else:
        search_candidates = [
            (dx, dy) for dy in range(-search_range * 2, search_range * 2 + 1)
            for dx in range(-search_range * 2, search_range * 2 + 1)
        ]

    for dx, dy in search_candidates:
        ref_y = current_y * 2 + dy
        ref_x = current_x * 2 + dx

        if ref_x < 0 or ref_y < 0 or ref_x + block_size * 2 > width * 2 or ref_y + block_size * 2 > height * 2:
            continue

        if dx % 2 == 0 and dy % 2 == 0:
            ref_block = reference_frame[ref_y // 2: ref_y // 2 + block_size,
                        ref_x // 2: ref_x // 2 + block_size]
        else:
            ref_block = interpolate_reference_block(reference_frame, ref_x / 2, ref_y / 2, block_size)

        mae = get_mae(current_block, ref_block)

        best_mae, best_mv, prediction_block = update_best_mv(
            mae, dx, dy, best_mae, best_mv, ref_block, prediction_block
        )

    return best_mv, best_mae, prediction_block


def intra_predict_block(x, y, current_frame, block_size):
    # Extract the current block
    current_block = current_frame[y:y + block_size, x:x + block_size]

    # Horizontal prediction: Propagate left boundary values
    if x == 0:
        # Use constant value at left edge
        left_boundary = np.full((block_size,), 128)
    else:
        left_boundary = current_frame[y:y + block_size, x - 1]

    horizontal_pred = np.tile(left_boundary[:, None], (1, block_size))

    # Vertical prediction: Propagate top boundary values
    if y == 0:
        # Use constant value at top edge
        top_boundary = np.full((block_size,), 128)
    else:
        top_boundary = current_frame[y - 1, x:x + block_size]

    vertical_pred = np.tile(top_boundary[None, :], (block_size, 1))

    # Select mode that yields the lowest MAE
    horizontal_mae = get_mae(current_block, horizontal_pred)
    vertical_mae = get_mae(current_block, vertical_pred)

    if horizontal_mae < vertical_mae:
        # Horizontal mode
        prediction = horizontal_pred
        mode = 0
        mae_value = horizontal_mae
    else:
        # Vertical mode
        prediction = vertical_pred
        mode = 1
        mae_value = vertical_mae

    return prediction, mode, mae_value


def inverses_intra_prediction_block(current_frame, x, y, block_size, mode):
    if mode == 0:
        if x == 0:
            left_boundary = np.full((block_size,), 128)
        else:
            left_boundary = current_frame[y:y + block_size, x - 1]

        prediction_block = np.tile(left_boundary[:, None], (1, block_size))

    elif mode == 1:
        if y == 0:
            top_boundary = np.full((block_size,), 128)
        else:
            top_boundary = current_frame[y - 1, x:x + block_size]

        prediction_block = np.tile(top_boundary[None, :], (block_size, 1))

    else:
        raise ValueError("Invalid mode. Mode should be 0 (horizontal) or 1 (vertical).")

    return prediction_block


def block_processing_pipeline(current_block, prediction_block, Q):
    residual_block = current_block - prediction_block

    # residual
    single_residual_block = dct_2d(residual_block)
    single_residual_block = quantize_dct(single_residual_block, Q)

    # reconstructed residual
    dequantized_block = single_residual_block * Q
    approximated_residual_block = idct_2d(dequantized_block)

    reconstructed_block = (approximated_residual_block + prediction_block).astype(np.uint8)

    return single_residual_block, reconstructed_block


def inverse_block_processing_pipeline(residual_block, Q):
    residual_block = residual_block * Q
    residual_block = idct_2d(residual_block)
    return residual_block


def get_refx_refy(height, width, x, y, mv_x, mv_y, block_size, sub_x=0, sub_y=0):
    ref_x = max(0, min(width - block_size, x + sub_x + mv_x))
    ref_y = max(0, min(height - block_size, y + sub_y + mv_y))
    return ref_x, ref_y


def vbs_on(x, y, reference_frames, reconstructed_frame, current_block, block_size, search_range, sub_q, lambda_value,
           previous_mv, frame_sign, fm_enable, fast_me):
    sub_outputs = []
    sub_residuals = []
    sub_block_size = block_size // 2
    sub_total_rd_cost = 0
    tmp_block = np.full((block_size, block_size), 128, dtype=np.uint8)

    sub_previous_mv = previous_mv
    for sub_y in range(0, block_size, sub_block_size):
        for sub_x in range(0, block_size, sub_block_size):
            sub_current_block = current_block[sub_y:sub_y + sub_block_size, sub_x:sub_x + sub_block_size]

            # intra_frame
            if frame_sign:
                sub_prediction_block, sub_mode, sub_mae_value = intra_predict_block(x + sub_x, y + sub_y,
                                                                                    reconstructed_frame,
                                                                                    sub_block_size)
                sub_residual_block, sub_reconstructed_block = block_processing_pipeline(sub_current_block,
                                                                                        sub_prediction_block, sub_q)
                tmp_block[sub_y:sub_y + sub_block_size, sub_x:sub_x + sub_block_size] = sub_reconstructed_block
                sub_total_rd_cost += get_rd_cost(get_sad(sub_current_block, sub_reconstructed_block),
                                                 get_mode_bit_cost(sub_mode) + get_residual_bit_cost(
                                                     sub_residual_block), lambda_value)
                sub_outputs.append(sub_mode)
            else:
                best_mv, best_mae, best_prediction_block = get_best_reference_block(reference_frames, sub_current_block,
                                                                                    sub_block_size, search_range,
                                                                                    x + sub_x,
                                                                                    y + sub_y, fm_enable, fast_me)
                sub_residual_block, sub_reconstructed_block = block_processing_pipeline(sub_current_block,
                                                                                        best_prediction_block, sub_q)
                tmp_block[sub_y:sub_y + sub_block_size, sub_x:sub_x + sub_block_size] = sub_reconstructed_block
                sub_total_rd_cost += get_rd_cost(get_sad(sub_current_block, sub_reconstructed_block),
                                                 get_mv_bit_cost(best_mv, sub_previous_mv) + get_residual_bit_cost(
                                                     sub_residual_block), lambda_value)
                sub_previous_mv = best_mv
                sub_outputs.append(best_mv)
            sub_residuals.append((1, sub_residual_block))

    return sub_outputs, sub_residuals, sub_total_rd_cost, tmp_block


def intra_prediction(reference_frames, current_frame, block_size, height, width, single_q, sub_q, vbs_enabled,
                     lambda_value, fm_enable, fast_me, RCflag, targetBR, fps, qp_bit_count_table, round, 
                     rc2_frame_bit_count, first_pass_modes = None, first_pass_vbs = None):
    prediction_modes = []
    frame_residual_values = []
    frame_qp = []
    frame_bit_count = []

    reconstructed_frame = np.full((height, width), 128, dtype=np.uint8)

    # RC mode
    if (RCflag and round != 0) or RCflag == 1:
        frame_bit_budget = targetBR // fps
        remaining_height = height
        bit_budget_per_row = compute_bit_budget_per_row(frame_bit_budget, width, remaining_height, block_size)
    
    for y in range(0, height, block_size):
        # RC mode
        row_bit_count = 0
        if (RCflag and round != 0) or RCflag == 1:
            if (RCflag and round == 1):
                bit_budget_per_row = rc2_frame_bit_count[y//block_size] / 100 * frame_bit_budget
            else:
                bit_budget_per_row = compute_bit_budget_per_row(frame_bit_budget, width, remaining_height, block_size)
            qp = find_qp_for_bit_budget(bit_budget_per_row, qp_bit_count_table)
            single_q = get_quantization_matrix(block_size, qp)
            sub_q = get_quantization_matrix(block_size//2, qp-1 if qp > 0 else 0)
            if not (RCflag and round == 1):
                frame_bit_budget -= bit_budget_per_row 
                remaining_height -= block_size
            frame_qp.append(qp)
        for x in range(0, width, block_size):
            current_block = current_frame[y:y + block_size, x:x + block_size]

            # single block encoding
            prediction_block, mode, mae_value = intra_predict_block(x, y, reconstructed_frame, block_size)
            single_residual_block, reconstructed_block = block_processing_pipeline(current_block, prediction_block,
                                                                                   single_q)
            single_rd_cost = get_rd_cost(get_sad(current_block, reconstructed_block),
                                         get_mode_bit_cost(mode) + get_residual_bit_cost(single_residual_block),
                                         lambda_value)

            if vbs_enabled and (math.log(block_size, 2)) <= 4:
                sub_modes, sub_residuals, sub_total_rd_cost, sub_reconstructed_block = vbs_on(x, y, reference_frames,
                                                                                              reconstructed_frame,
                                                                                              current_block,
                                                                                              block_size, 0,
                                                                                              sub_q, lambda_value,
                                                                                              (0, 0, 0), True,
                                                                                              fm_enable, fast_me)
                
                row_bit_count += get_mode_bit_cost(mode) + get_residual_bit_cost(single_residual_block)

                if single_rd_cost <= sub_total_rd_cost:
                    # single block encoding
                    prediction_modes.append(mode)
                    frame_residual_values.append((0, single_residual_block))
                else:
                    # sub block encoding
                    reconstructed_block = sub_reconstructed_block
                    prediction_modes.extend(sub_modes)
                    frame_residual_values.extend(sub_residuals)
            else:
                prediction_modes.append(mode)
                frame_residual_values.append((0, single_residual_block))
            reconstructed_frame[y:y + block_size, x:x + block_size] = reconstructed_block
        frame_bit_count.append(row_bit_count)
    return prediction_modes, frame_residual_values, reconstructed_frame, frame_qp, frame_bit_count


def inter_prediction(current_frame, reference_frames, block_size, height, width, search_range, single_q, sub_q,
                     vbs_enabled, lambda_value, fm_enable, fast_me, RCflag, targetBR, fps, qp_bit_count_table, round,
                     rc2_frame_bit_count, first_pass_mvs = None, first_pass_vbs = None):
    prediction_mvs = []
    frame_residual_values = []
    frame_qp = []
    frame_bit_count = []

    if RCflag == 3 and round == 1 and first_pass_mvs and first_pass_vbs and search_range > 1:
        search_range = search_range // 2

    reconstructed_frame = np.full((height, width), 128, dtype=np.uint8)

    previous_mv = (0, 0, 0)

    # RC mode
    if (RCflag and round != 0) or RCflag == 1:
        frame_bit_budget = targetBR // fps
        remaining_height = height
        bit_budget_per_row = compute_bit_budget_per_row(frame_bit_budget, width, remaining_height, block_size)

    
    for y in range(0, height, block_size):
        # RC mode
        row_bit_count = 0
        if (RCflag and round != 0) or RCflag == 1:
            if (RCflag and round == 1):
                bit_budget_per_row = rc2_frame_bit_count[y//block_size] / 100 * frame_bit_budget
            else:
                bit_budget_per_row = compute_bit_budget_per_row(frame_bit_budget, width, remaining_height, block_size)
            qp = find_qp_for_bit_budget(bit_budget_per_row, qp_bit_count_table)
            single_q = get_quantization_matrix(block_size, qp)
            sub_q = get_quantization_matrix(block_size//2, qp-1 if qp > 0 else 0)
            if not (RCflag and round == 1):
                frame_bit_budget -= bit_budget_per_row 
                remaining_height -= block_size
            # print(f"frame_bit_budget: {frame_bit_budget}, remaining_height: {remaining_height}")
            frame_qp.append(qp)
        for x in range(0, width, block_size):
            current_block = current_frame[y:y + block_size, x:x + block_size]

            best_mv, best_mae, best_prediction_block = get_best_reference_block(reference_frames, current_block,
                                                                                block_size, search_range, x, y,
                                                                                fm_enable, fast_me)
            single_residual_block, reconstructed_block = block_processing_pipeline(current_block, best_prediction_block,
                                                                                   single_q)
            single_rd_cost = get_rd_cost(get_sad(current_block, reconstructed_block),
                                         get_mv_bit_cost(best_mv, previous_mv) + get_residual_bit_cost(
                                             single_residual_block),
                                         lambda_value)
            
            row_bit_count += get_mv_bit_cost(best_mv, previous_mv) + get_residual_bit_cost(single_residual_block)
            

            if vbs_enabled and (math.log(block_size, 2)) <= 4:
                sub_mvs, sub_residuals, sub_total_rd_cost, sub_reconstructed_block = vbs_on(x, y, reference_frames,
                                                                                            reconstructed_frame,
                                                                                            current_block,
                                                                                            block_size, search_range,
                                                                                            sub_q, lambda_value,
                                                                                            previous_mv, False,
                                                                                            fm_enable, fast_me)
                if single_rd_cost <= sub_total_rd_cost:
                    # single block encoding
                    previous_mv = best_mv
                    prediction_mvs.append(best_mv)
                    frame_residual_values.append((0, single_residual_block))
                else:
                    # sub block encoding
                    previous_mv = sub_mvs[-1]
                    reconstructed_block = sub_reconstructed_block
                    prediction_mvs.extend(sub_mvs)
                    frame_residual_values.extend(sub_residuals)
            else:
                prediction_mvs.append(best_mv)
                frame_residual_values.append((0, single_residual_block))
            reconstructed_frame[y:y + block_size, x:x + block_size] = reconstructed_block
        frame_bit_count.append(row_bit_count)
    return prediction_mvs, frame_residual_values, reconstructed_frame, frame_qp, frame_bit_count


def get_best_reference_block(reference_frames, current_block, block_size, search_range, x, y, fm_enable, fast_me):
    best_mv = None
    best_mae = float('inf')
    best_prediction_block = None

    for ref_idx, reference_frame in enumerate(reference_frames):

        if fm_enable:
            mv, mae, prediction_block = fractional_pixel_full_search(
                current_block, reference_frame, block_size, search_range, x, y, fast_me
            )

        else:
            mv, mae, prediction_block = integer_pixel_full_search(
                current_block, reference_frame, block_size, search_range, x, y, fast_me
            )

        if mae < best_mae:
            best_mv = (ref_idx, mv[0], mv[1])
            best_mae = mae
            best_prediction_block = prediction_block

    return best_mv, best_mae, best_prediction_block
