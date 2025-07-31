import os
import numpy as np
from PIL import Image
from config import config
from src.utils.process_frame_block_util import *
from src.utils.generated_file_process_util import *

input_file_path = config.Paths["Input_Video"]
output_folder_path = config.Paths["Output_Directory"]
output_modes_path = config.Paths["Output_Modes_Path"]
output_mvs_path = config.Paths["Output_Mvs_Path"]
output_residuals_path = config.Paths["Output_Residuals_Path"]
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
output_reconstructed_frames_path = config.Paths["Output_Reconstructed_frames"]

PNG_folder = "PNGs"


def convert_y_to_png(input_folder, output_folder, width, height):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all .y files in the input folder
    y_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.y')])
    counter = 0

    for y_file in y_files:
        if counter >= 10:
            break
        # Construct the full path to the .y file
        y_path = os.path.join(input_folder, y_file)

        # Read the raw .y data
        with open(y_path, 'rb') as f:
            y_data = f.read()

        # Convert the byte data into a numpy array
        y_array = np.frombuffer(y_data, dtype=np.uint8)

        # Reshape the array into a 2D array with the given resolution
        y_frame = y_array.reshape((height, width))

        # Convert the 2D array into a PIL Image
        img = Image.fromarray(y_frame, mode='L')  # 'L' mode for grayscale images

        # Save the image as .png
        png_filename = os.path.join(output_folder, y_file.replace('.y', '.png'))
        img.save(png_filename)
        print(f"Converted {y_file} to {png_filename}")
        counter += 1


def compare_decoded_frames(decoded_frames_dir, reconstructed_frames_dir, width, height):
    # Get sorted lists of .y files in each directory
    decoded_files = sorted([f for f in os.listdir(decoded_frames_dir) if f.endswith('.y')])
    reconstructed_files = sorted([f for f in os.listdir(reconstructed_frames_dir) if f.endswith('.y')])

    for decoded_file, reconstructed_file in zip(decoded_files, reconstructed_files):
        # Read the decoded frame
        with open(os.path.join(decoded_frames_dir, decoded_file), 'rb') as f:
            decoded_data = f.read()
        decoded_frame = np.frombuffer(decoded_data, dtype=np.uint8).reshape((height, width))

        # Read the reconstructed frame
        with open(os.path.join(reconstructed_frames_dir, reconstructed_file), 'rb') as f:
            reconstructed_data = f.read()
        reconstructed_frame = np.frombuffer(reconstructed_data, dtype=np.uint8).reshape((height, width))

        # Compare the frames
        if not np.array_equal(decoded_frame, reconstructed_frame):
            raise AssertionError(f"Mismatch in {decoded_file} and {reconstructed_file}")
        print(f"{decoded_file} matches with {reconstructed_file}")



input_folder_d = output_folder_path + "/" + output_reconstructed_frames_path
output_folder_d = "Decoded_PNG_Folder"

input_folder_y = output_folder_path + "/y-only"
output_folder_y = "Y_Only_PNG_Folder"

input_folder_r = "reconstrcted_frames"
output_folder_r = "Reconstructed_PNG_Folder"

width = 352
height = 288

convert_y_to_png(input_folder_r, PNG_folder + "/" + output_folder_r, width, height)
convert_y_to_png(input_folder_y, PNG_folder + "/" + output_folder_y, width, height)
convert_y_to_png(input_folder_d, PNG_folder + "/" + output_folder_d, width, height)

compare_decoded_frames(input_folder_d, input_folder_r, width, height)
