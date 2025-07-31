from src.utils.generated_file_process_util import *

def compute_bit_budget_per_row(frame_bit_budget, width, height, block_size):
    rows_per_frame = height // block_size
    return frame_bit_budget // rows_per_frame


def find_qp_for_bit_budget(row_budget, table):
    for qp_value, avg_bit_count in table:
        if avg_bit_count <= row_budget:
            return qp_value
    return table[-1][0]
    # raise Exception("No correcponding qp found in the RC/bit-count table")


def compute_scene_change_threshold(qp):
    base_threshold = 1190000  # Example base threshold
    return int(base_threshold * (2 ** (-qp / 3.485))) 


def list_to_percentage(input_list):
    total_sum = sum(input_list)
    if total_sum == 0:
        raise Exception("The total sum of the list is 0")
    return [(x / total_sum) * 100 for x in input_list]


def modify_qp_bitcount_table(bit_counts, intended_num_of_frames, height, block_size, qp, qp_bit_count_table):
    temp_total_bits = 0
    for frame_bit_count in bit_counts:
        temp_total_bits += sum(frame_bit_count)
    temp_row_bit_count = temp_total_bits // intended_num_of_frames // (height // block_size)
    value_for_qp = next((value for key, value in qp_bit_count_table if key == qp), None)
    referece_rate = temp_row_bit_count / value_for_qp
    qp_bit_count_table = [(key, int(value * referece_rate)) for key, value in qp_bit_count_table] 
    return qp_bit_count_table


def calculate_avg_qp(qp_list):
    total_qp = 0
    num_blocks = 0
    for frame_qp in qp_list:
        num_blocks = len(frame_qp)
        for qp in frame_qp:
            total_qp += qp

    return total_qp / len(qp_list) / num_blocks




