from src.utils.pool import *
import time

t1 = time.perf_counter()

input_file_path = config.Paths["Input_Video"]
output_folder_path = config.Paths["Output_Directory"]
output_modes_path = config.Paths["Output_Modes_Path"]
output_mvs_path = config.Paths["Output_Mvs_Path"]
output_residuals_path = config.Paths["Output_Residuals_Path"]
output_qp_path = config.Paths["Output_QP_path"]
output_frame_type_path = config.Paths["Output_frame_type_path"]


parallel_mode = config.Features["parallel_mode"]
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
i_period = config.I_Period
qp = config.QP

RCflag = config.Features["RCflag"]
targetBR = config.Features["targetBR"]
fps = config.Features["fps"]
cif_i_qp_bit_count_table = config.Features["cif_i_table"]
cif_p_qp_bit_count_table = config.Features["cif_p_table"]


modes = []
mvs = []
residuals = []
frames = read_y_only_component(input_file_path, frame_width, frame_height)
frames2 = read_y_only_component("./src/resources/akiyo_qcif.yuv", frame_width, frame_height)
frames3 = frames[:7] + frames2[:7] + frames[7:14]
reference_frame = pad_frame(np.full((frame_height, frame_width), 128, dtype=np.uint8), block_size)
height = reference_frame.shape[0]
width = reference_frame.shape[1]
reference_frames = [reference_frame]
reconstructed_frames = []
single_q = get_quantization_matrix(block_size, qp)
sub_q = get_quantization_matrix(block_size//2, qp-1 if qp > 0 else 0)
qp_list = []
i_p_frame_flags = []
bit_counts_percentage = []
bit_counts = []
total_bit_count = 0

intended_num_of_frames = 21

first_pass_mvs = []
first_pass_modes = []
first_pass_vbs = []

frames = [pad_frame(frame, block_size) for frame in frames]
start_time = time.time()
# ---------------------------------------------------No Parallel Processing-------------------------------------------------------

if not parallel_mode:
    for round in range(2 if (RCflag > 1) else 1):
        print(f"RC Round {round}")
        reference_frame = pad_frame(np.full((frame_height, frame_width), 128, dtype=np.uint8), block_size)
        reference_frames = [reference_frame]
        total_bits = 0
        test_total_bits = 0
        total_psnr =0

        for frame_idx in range(len(frames)):
            if frame_idx == 21:
                break
            # if frame_idx == intended_num_of_frames:
            #     break
            current_frame = pad_frame(frames3[frame_idx], block_size)

            if RCflag and round == 1:
                rc2_frame_bit_count = bit_counts_percentage[frame_idx]
            else:
                rc2_frame_bit_count = []

            if not RCflag or (RCflag and round == 0):
                i_frame_flag = (frame_idx % i_period == 0)
            else:
                i_frame_flag = i_p_frame_flags[frame_idx]

            if i_frame_flag:
                # print(f"Frame: {frame_idx + 1} processed as i-frame")
                prediction_modes, frame_residual_values, reconstructed_frame, frame_qp, frame_bit_count = intra_prediction(reference_frames, current_frame, block_size,
                                                                                                height, width,
                                                                                                single_q, sub_q,
                                                                                                vbs_enabled, lambda_value, fm_enable, fast_me,
                                                                                                RCflag, targetBR, fps, cif_i_qp_bit_count_table, round, rc2_frame_bit_count,
                                                                                                first_pass_modes if RCflag == 3 and round == 1 else None,
                                                                                                first_pass_vbs if RCflag == 3 and round == 1 else None
                                                                                                )
                # differential encoding， append to modes
                if (RCflag == 1 or (RCflag and round == 1) or not RCflag):
                    encode_modes = differential_encode_modes(prediction_modes)
                    modes.append(encode_modes)
                    total_bits += sum((len(exp_golomb_encode(i)) for i in encode_modes))

                reference_frames = [reconstructed_frame]
                i_p_frame_flags.append(True) if not round == 1 else None

            else:
                # print(f"Frame: {frame_idx + 1} processed as f-frame")
                prediction_mvs, frame_residual_values, reconstructed_frame, frame_qp, frame_bit_count = inter_prediction(current_frame, reference_frames,
                                                                                            block_size, height,
                                                                                            width, search_range,
                                                                                            single_q, sub_q,
                                                                                            vbs_enabled, lambda_value, fm_enable, fast_me,
                                                                                            RCflag, targetBR, fps, cif_p_qp_bit_count_table, round, rc2_frame_bit_count,
                                                                                            first_pass_mvs if RCflag == 3 and round == 1 else None,
                                                                                            first_pass_vbs if RCflag == 3 and round == 1 else None
                                                                                            )

                if (RCflag == 1 or (RCflag and round == 1) or not RCflag):
                    encode_mvs = differential_encode_motion_vectors(prediction_mvs)
                    mvs.append(encode_mvs)
                    for i in encode_mvs:
                        total_bits += sum(len(exp_golomb_encode(j)) for j in i)

                reference_frames.append(reconstructed_frame)
                if len(reference_frames) > nRefFrames:
                    reference_frames.pop(0)

                i_p_frame_flags.append(sum(frame_bit_count) > compute_scene_change_threshold(qp) if (RCflag > 1 and round == 0) else False)  if not round == 1 else None

            for residuals_per_frame in frame_residual_values:  #
                split_sign = residuals_per_frame[0]
                residual_block = residuals_per_frame[1]
                encoded_split_sign = exp_golomb_encode(split_sign)

                tmp = entropy_encode(residual_block)
                encoded_residual = [exp_golomb_encode(i) for i in tmp]
                for residual in encoded_residual:
                    total_bits += len(residual)
                total_bits += len(encoded_split_sign)

            if (RCflag == 1 or (RCflag and round == 1) or not RCflag):
                reconstructed_frames.append(reconstructed_frame)
                qp_list.append(frame_qp)
                residuals.append(frame_residual_values)

            psnr_value, ssim_value, mae_value = psnr_ssim_mae(current_frame, reconstructed_frame)
            total_psnr += psnr_value
            print(f"Frame {frame_idx + 1} bit count: {sum(frame_bit_count)}")
            print(f"Frame {frame_idx + 1}: PSNR={psnr_value:.5f}, SSIM={ssim_value:.5f}, MAE={mae_value:.5f}")

            if not (RCflag == 2 and round == 0) or (RCflag == 3 and round == 0):
                total_bit_count += sum(frame_bit_count)
            if (RCflag and round == 0):
                bit_counts_percentage.append(list_to_percentage(frame_bit_count))
                bit_counts.append(frame_bit_count)
            # RC3
            if RCflag == 3 and round == 0:
                if not i_frame_flag:
                    first_pass_mvs.append(prediction_mvs)
                else:
                    first_pass_modes.append(prediction_modes)
                first_pass_vbs.append(frame_residual_values)

        if (RCflag and round == 0):
            qp_bit_count_table = modify_qp_bitcount_table(bit_counts, intended_num_of_frames, height, block_size, qp, cif_p_qp_bit_count_table)
            print(f"QP table modified: {qp_bit_count_table}")

        print(f"IP_Frame_Flags: {i_p_frame_flags}\n")

    print(
        f"avg_psnr = {total_psnr / 21}, bitrate/bps = {total_bits / 0.7}, QP = {qp}, budget_bitrate = {targetBR}, Rc_flag = {RCflag}")
    end_time = time.time()  # 记录结束时间
    encoding_time = end_time - start_time
    print(f"Encoding time: {encoding_time}")
    print(f"Average qp for RC=2 is: {calculate_avg_qp(qp_list)}") if RCflag == 2 or RCflag == 3 else None
    print(f"Total bit count of all frames: {total_bit_count}")
    dump_reconstructed_frame(reconstructed_frames, "reconstrcted_frames")
    write_modes_to_bitstream(modes, output_folder_path + "/" + output_modes_path)
    write_mvs_to_bitstream(mvs, output_folder_path + "/" + output_mvs_path)
    write_residuals_to_bitstream(residuals, output_folder_path + "/" + output_residuals_path)
    write_qp_to_bitstream(qp_list, output_folder_path + "/" + output_qp_path)
    write_ip_list_as_binary(i_p_frame_flags, output_folder_path + "/" + output_frame_type_path)


# ---------------------------------------------------Parallel Type 1-------------------------------------------------------
elif parallel_mode == 1:
    for frame_idx in range(len(frames)):
        if frame_idx == intended_num_of_frames:
            break

        frame_blocks = []
        frame_blocks_indices = []

        current_frame = pad_frame(frames3[frame_idx], block_size)

        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                frame_blocks.append(current_frame[y:y + block_size, x:x + block_size])
                frame_blocks_indices.append((y, x))

        frame_results = parallel_mode_1(frame_blocks, frame_blocks_indices, single_q)

        reconstructed_frame = reconstruct_frame_from_blocks(frame_results, block_size, frame_height, frame_width)

        psnr_value, ssim_value, mae_value = psnr_ssim_mae(current_frame, reconstructed_frame)
        print(f"Frame {frame_idx + 1}: PSNR={psnr_value:.5f}, SSIM={ssim_value:.5f}, MAE={mae_value:.5f}")



# ---------------------------------------------------Parallel Type 2-------------------------------------------------------
elif parallel_mode == 2:
    for frame_idx in range(len(frames)):
        if frame_idx == intended_num_of_frames:
            break

        current_frame = pad_frame(frames3[frame_idx], block_size)

        reconstructed_frame, bitstream_buffer = parallel_mode_2(current_frame, block_size, single_q)

        psnr_value, ssim_value, mae_value = psnr_ssim_mae(current_frame, reconstructed_frame)
        print(f"Frame {frame_idx + 1}: PSNR={psnr_value:.5f}, SSIM={ssim_value:.5f}, MAE={mae_value:.5f}")

        # print(reconstructed_frame)
        # print(reconstructed_frame.shape)



# ---------------------------------------------------Parallel Type 3-------------------------------------------------------
elif parallel_mode == 3:
    reconstructed_frames = parallel_mode_3(frames3[:intended_num_of_frames], block_size, single_q, i_period, intended_num_of_frames, fm_enable, fast_me)

    for frame_idx in range(intended_num_of_frames):
        reconstructed_frame = reconstructed_frames[frame_idx]
        current_frame = pad_frame(frames3[frame_idx], block_size)

        psnr_value, ssim_value, mae_value = psnr_ssim_mae(current_frame, reconstructed_frame)
        print(f"Frame {frame_idx + 1}: PSNR={psnr_value:.5f}, SSIM={ssim_value:.5f}, MAE={mae_value:.5f}")



else:
    print("Invalid Encoding Mode")



t2 = time.perf_counter()

print(f"\n----------------Execution time: {t2-t1} seconds--------------------")