import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.utils.generated_file_process_util import *


def get_psnr(frame1, frame2):
    psnr_value = psnr(frame1, frame2, data_range=255)
    return psnr_value


def get_ssim(frame1, frame2):
    ssim_value = ssim(frame1, frame2, data_range=255)
    return ssim_value


def get_mae(frame1, frame2):
    return np.mean(np.abs(frame1.astype(np.float64) - frame2.astype(np.float64)))


def get_rd_cost(distortion, rate, lambda_value):
    rd_cost = distortion + lambda_value * rate
    return rd_cost


def get_sad(block1, block2):
    sad = np.sum(np.abs(block1 - block2))
    return sad


def get_mode_bit_cost(mode):
    return len(exp_golomb_encode(mode))


def get_mv_bit_cost(current_mv, previous_mv):
    diff_mv = [current_mv[i] - previous_mv[i] for i in range(len(current_mv))]
    mv_bit_cost = sum(len(exp_golomb_encode(i)) for i in diff_mv)
    return mv_bit_cost


def get_residual_bit_cost(residual_block):
    rle_encoded_residual = entropy_encode(residual_block)
    residual_bit_cost = sum(len(exp_golomb_encode(i)) for i in rle_encoded_residual)

    return residual_bit_cost


def psnr_ssim_mae(frame1, frame2):
    psnr_value = get_psnr(frame1, frame2)
    ssim_value = get_ssim(frame1, frame2)
    mae_value = get_mae(frame1, frame2)
    return psnr_value, ssim_value, mae_value


def plot_rd_graph(bitrates, psnr_values, rcflags, title="R-D Graph"):
    plt.figure(figsize=(12, 6))


    for rcflag in rcflags:
        if rcflag in bitrates and rcflag in psnr_values:
            bitrates_kbps = [rate / 1000 for rate in bitrates[rcflag]]
            plt.plot(
                bitrates_kbps,
                psnr_values[rcflag],
                marker='o',
                label=f"RCflag={rcflag}"
            )
        else:
            print(f"Warning: RCflag {rcflag} data is incomplete.")

    plt.title(title)
    plt.xlabel("Bitrate (kbps)", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_per_frame_psnr(frame_psnr, parallel_mode, title="Per-frame PSNR Graph"):
    frame_numbers = list(range(1, 22))

    plt.figure(figsize=(12, 6))


    for mode in parallel_mode:
        if mode in frame_psnr:
            plt.plot(
                frame_numbers,
                frame_psnr[mode],
                marker='o',
                label=f"=parallel_mode{mode}"
            )

    plt.title(title, fontsize=14)
    plt.xlabel("Frame Number", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    plt.show()



bitrates = {
    0: [24022390, 9725317, 1657705],
    1: [6272312, 1709198, 417925],
    2: [5830244, 1693704, 6600],
    3: [5124655, 1345670, 6242]
}


psnr_values = {
    0: [33.38, 13.54, 9.45],
    1: [11.17, 9.20, 7.82],
    2: [10.90, 9.24, 8.47],
    3: [10.43, 8.78, 7.82]
}


rcflags = [0, 1, 2, 3]

#plot_rd_graph(bitrates, psnr_values, rcflags, title="Rate-Distortion Graph")



