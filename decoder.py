from config import config
from src.utils.process_frame_block_util import *
from src.utils.generated_file_process_util import *
from src.utils.read_write_input_output_util import dump_reconstructed_frame

input_file_path = config.Paths["Input_Video"]
output_folder_path = config.Paths["Output_Directory"]
output_modes_path = config.Paths["Output_Modes_Path"]
output_mvs_path = config.Paths["Output_Mvs_Path"]
output_residuals_path = config.Paths["Output_Residuals_Path"]
output_reconstructed_frames_path = config.Paths["Output_Reconstructed_frames"]
output_qp_path = config.Paths["Output_QP_path"]
output_frame_type_path = config.Paths["Output_frame_type_path"]

nref_frames_enabled = config.Features["nRefFrames"]
nRefFrames = min(4, max(1, nref_frames_enabled))
vbs_enabled = config.Features["VBSEnable"]
lambda_value = config.lambda_value
fm_enable = config.Features["FMEEnable"]
fast_me = config.Features["FastME"]
frame_width = config.Resolution["width"]
frame_height = config.Resolution["height"]
search_range = config.search_range
block_size = config.Block_Size
sub_block_size = block_size // 2
i_period = config.I_Period
qp = config.QP
RCflag = config.Features["RCflag"]

single_q = get_quantization_matrix(block_size, qp)
sub_q = get_quantization_matrix(block_size // 2, qp - 1 if qp > 0 else 0)
modes = read_modes_from_bitstream(output_folder_path + "/" + output_modes_path)
mvs = read_mvs_from_bitstream(output_folder_path + "/" + output_mvs_path)
residuals = read_residuals_from_bitstream(output_folder_path + "/" + output_residuals_path, block_size)
qp_list = read_qp_from_bitstream(output_folder_path + "/" + output_qp_path)
i_p_frame_flags = read_ip_file_as_list(output_folder_path + "/" + output_frame_type_path)

reference_frame = pad_frame(np.full((frame_height, frame_width), 128, dtype=np.uint8), block_size)
height = reference_frame.shape[0]
width = reference_frame.shape[1]
reference_frames = []
reconstructed_frames = []
i_frame_counter = 0

total_frames = len(residuals)
for frame_idx in range(total_frames):
    # print(frame_idx)
    if RCflag < 2:
        i_frame_flag = (frame_idx % i_period == 0)
    else:
        i_frame_flag = i_p_frame_flags[frame_idx]
    
    residual_blocks = residuals[frame_idx]

    reconstructed_frame = np.full((frame_height, frame_width), 128, dtype=np.uint8)
    reconstructed_frame = pad_frame(reconstructed_frame, block_size)

    block_idx = 0
    frame_qp = qp_list[frame_idx]
    for y in range(0, height, block_size):
        
        if RCflag:
            qp = frame_qp[y // block_size]
            single_q = get_quantization_matrix(block_size, qp)
            sub_q = get_quantization_matrix(block_size // 2, qp - 1 if qp > 0 else 0)

        for x in range(0, width, block_size):

            split_sign, residual_block = residual_blocks[block_idx]

            if split_sign == 0:
                residual_block = inverse_block_processing_pipeline(residual_block, single_q)

                if i_frame_flag:
                    prediction_block = inverses_intra_prediction_block(reconstructed_frame, x, y, block_size,
                                                                       modes[i_frame_counter][block_idx])
                    reconstructed_block = (residual_block + prediction_block).astype(np.uint8)
                else:
                    if fm_enable:
                        ref_frame_idx, mv_x, mv_y = mvs[frame_idx - frame_idx // i_period - 1][block_idx]
                        mv_x = mv_x / 2
                        mv_y = mv_y / 2

                        if mv_x.is_integer() and mv_y.is_integer():
                            ref_x, ref_y = get_refx_refy(height, width, x, y, int(mv_x), int(mv_y), block_size)
                            ref_block = reference_frames[ref_frame_idx][ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                        else:
                            ref_x = x + mv_x
                            ref_y = y + mv_y
                            ref_block = interpolate_reference_block(reference_frames[ref_frame_idx], ref_x, ref_y, block_size)

                        reconstructed_block = (residual_block + ref_block).astype(np.uint8)

                    else:
                        ref_frame_idx, mv_x, mv_y = mvs[frame_idx - frame_idx // i_period - 1][block_idx]

                        ref_x, ref_y = get_refx_refy(height, width, x, y, mv_x, mv_y, block_size)
                        ref_block = reference_frames[ref_frame_idx][ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                        reconstructed_block = (residual_block + ref_block).astype(np.uint8)

                block_idx += 1


            else:
                reconstructed_block = np.full((block_size, block_size), 128, dtype=np.uint8)
                for sub_y in range(0, block_size, sub_block_size):
                    for sub_x in range(0, block_size, sub_block_size):
                        _, sub_residual_block = residual_blocks[block_idx]
                        sub_residual_block = residual_block = inverse_block_processing_pipeline(sub_residual_block, sub_q)

                        if i_frame_flag:
                            prediction_block = inverses_intra_prediction_block(reconstructed_frame, x + sub_x,
                                                                               y + sub_y, sub_block_size,
                                                                               modes[i_frame_counter][block_idx])
                            sub_reconstructed_block = (sub_residual_block + prediction_block).astype(np.uint8)
                        else:
                            if fm_enable:
                                ref_frame_idx, mv_x, mv_y = mvs[frame_idx - frame_idx // i_period - 1][block_idx]
                                mv_x = mv_x / 2
                                mv_y = mv_y / 2

                                if mv_x.is_integer() and mv_y.is_integer():
                                    ref_x, ref_y = get_refx_refy(height, width, x, y, int(mv_x), int(mv_y), sub_block_size, sub_x, sub_y)
                                    ref_block = reference_frames[ref_frame_idx][ref_y:ref_y + sub_block_size, ref_x:ref_x + sub_block_size]
                                else:
                                    ref_x = x + mv_x + sub_x
                                    ref_y = y + mv_y + sub_y
                                    ref_block = interpolate_reference_block(reference_frames[ref_frame_idx], ref_x, ref_y, sub_block_size)
                            else:
                                ref_frame_idx, mv_x, mv_y = mvs[frame_idx - frame_idx // i_period - 1][block_idx]

                                ref_x, ref_y = get_refx_refy(height, width, x, y, mv_x, mv_y, sub_block_size, sub_x, sub_y)
                                ref_block = reference_frames[ref_frame_idx][ref_y:ref_y + sub_block_size, ref_x:ref_x + sub_block_size]
                            sub_reconstructed_block = (sub_residual_block + ref_block).astype(np.uint8)

                        reconstructed_block[sub_y:sub_y + sub_block_size, sub_x:sub_x + sub_block_size] = sub_reconstructed_block
                        block_idx += 1

            reconstructed_frame[y:y + block_size, x:x + block_size] = reconstructed_block
            
    reference_frame = reconstructed_frame
    reconstructed_frames.append(reconstructed_frame)
    reference_frames.append(reconstructed_frame)
    if len(reference_frames) > nRefFrames:
        reference_frames.pop(0)
    if i_frame_flag:
        i_frame_counter += 1

dump_reconstructed_frame(reconstructed_frames, output_folder_path + "/" + output_reconstructed_frames_path)