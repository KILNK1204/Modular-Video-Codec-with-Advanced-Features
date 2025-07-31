from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from threading import Lock, Barrier, Condition
import numpy as np
from config import config
from src.utils.process_frame_block_util import *
from src.utils.generated_file_process_util import *
from src.utils.read_write_input_output_util import *
import time
import concurrent.futures

RCflag = config.Features["RCflag"]
targetBR = config.Features["targetBR"]
fps = config.Features["fps"]
qp_bit_count_table = config.Features["table"]


# Reconstruct a frame by using a list of block infos (dictionary)
def reconstruct_frame_from_blocks(blocks, block_size, frame_height, frame_width):
    reconstructed_frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for block_data in blocks:
        block = block_data["reconstructed_block"]
        block_y, block_x = block_data["block_index"]
        reconstructed_frame[block_y:block_y + block_size, block_x:block_x + block_size] = block
    return reconstructed_frame


# Simple block encoding method, dct and quantize dct, takes block cordinate for parallel 1 process reference
def encode_block_independently(args):
    block, block_index, Q = args

    prediction_block = np.full(block.shape, 128, dtype=np.uint8)

    residual_block = block - prediction_block
    single_residual_block = dct_2d(residual_block)
    single_residual_block = quantize_dct(single_residual_block, Q)

    dequantized_block = single_residual_block * Q
    approximated_residual_block = idct_2d(dequantized_block)

    reconstructed_block = (approximated_residual_block + prediction_block).astype(np.uint8)

    block_bit_count = get_residual_bit_cost(single_residual_block)

    return {
        "block": block,
        "block_index": block_index,
        "reconstructed_block": reconstructed_block,
        "block_bit_count": block_bit_count
    }


# Parallel 1, create as many threads as possible
def parallel_mode_1(blocks, block_indices, Q):
    paired_data = zip(blocks, block_indices, [Q] * len(blocks))
    result_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(encode_block_independently, paired_data)

    for result in results:
        result_list.append(result)

    return result_list


# Simple block encoding method, dct and quantize dct
def encode_block(block, previous_block, Q):
    prediction_block = previous_block

    residual_block = block - prediction_block
    single_residual_block = dct_2d(residual_block)
    single_residual_block = quantize_dct(single_residual_block, Q)

    dequantized_block = single_residual_block * Q
    approximated_residual_block = idct_2d(dequantized_block)

    reconstructed_block = (approximated_residual_block + prediction_block).astype(np.uint8)

    return reconstructed_block


def parallel_mode_2(frame, block_size, Q):
    height, width = frame.shape
    encoded_frame = np.zeros_like(frame, dtype=np.float32)

    bitstream_lock = Lock()
    bitstream_buffer = []
    barrier = Barrier(2)

    def process_row(row_start, thread_id):
        previous_block = np.full((block_size, block_size), 128, dtype=np.uint8)
        row_end = min(row_start + block_size, height)
        for col_start in range(0, width, block_size):
            col_end = min(col_start + block_size, width)
            block = frame[row_start:row_end, col_start:col_end]
            reconstructed_block = encode_block(block, previous_block, Q)
            previous_block = reconstructed_block

            # Simulation of writing into bitstream, lock the bitstream then write.
            with bitstream_lock:
                encoded_frame[row_start:row_end, col_start:col_end] = reconstructed_block
                bitstream_buffer.append((thread_id, (row_start, col_start), reconstructed_block))

        barrier.wait()

    pool = ThreadPool(2)
    for row_start in range(0, height, block_size * 2):
        pool.apply_async(process_row, args=(row_start, 1))
        pool.apply_async(process_row, args=(row_start + block_size, 2))

    pool.close()
    pool.join()

    # Order and write bitstream
    bitstream_buffer.sort(key=lambda x: x[1])
    return encoded_frame, bitstream_buffer


def parallel_mode_3(frames, block_size, Q, i_period, intended_num_of_frames, fm_enable, fast_me):
    height, width = frames[0].shape
    encoded_frames = [np.zeros_like(frame, dtype=np.uint8) for frame in frames]

    row_condition = [0] * intended_num_of_frames

    def process_frame(i_period, frame_idx, thread_id, fm_enable, fast_me):
        # print(f"Thread {thread_id} starts processing Frame {frame_idx}")
        frame = frames[frame_idx]
        encoded_frame = encoded_frames[frame_idx]
        prev_frame = encoded_frames[frame_idx - 1]
        is_i_frame = (frame_idx % i_period == 0)

        for row_start in range(0, height, block_size):
            row_end = min(row_start + block_size, height)
            for col_start in range(0, width, block_size):
                col_end = min(col_start + block_size, width)
                block = frame[row_start:row_end, col_start:col_end]

                # Handle I-frame and P-frame separately
                if not is_i_frame:
                    while (row_condition[frame_idx - 1] < (height - block_size)) and (
                            row_condition[frame_idx - 1] < (row_start + block_size * 2)):
                        pass

                    if fm_enable:
                        _, _, prediction_block = fractional_pixel_full_search(block, prev_frame, block_size, 3,
                                                                              col_start, row_start, fast_me)
                    else:
                        _, _, prediction_block = integer_pixel_full_search(block, prev_frame, block_size, 3, col_start,
                                                                           row_start, fast_me)

                    # encoded_block = encode_block(block, prev_frame[row_start:row_end, col_start:col_end], Q)
                    encoded_block = encode_block(block, prediction_block, Q)

                else:
                    # I-frame encoding (no dependencies)
                    encoded_block = encode_block(block, np.full((block_size, block_size), 128, dtype=np.uint8), Q)

                # Store the encoded block
                encoded_frame[row_start:row_end, col_start:col_end] = encoded_block

            row_condition[frame_idx] = row_start

        # print(f"Thread {thread_id} completed for Frame {frame_idx}")

    # Create a thread pool to process frames in parallel
    pool = ThreadPool(2)
    for frame_idx in range(intended_num_of_frames):
        pool.apply_async(process_frame, args=(i_period, frame_idx, frame_idx, fm_enable, fast_me))

    pool.close()
    pool.join()

    return encoded_frames