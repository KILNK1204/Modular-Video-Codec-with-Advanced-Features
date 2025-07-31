import os
import shutil
import numpy as np


def read_yuv420_all_frames(file, width, height):
    y_size = width * height
    uv_size = y_size // 4

    y = []
    u = []
    v = []
    with open(file, 'rb') as f:
        while True:
            y_data = f.read(y_size)
            if len(y_data) < y_size:
                break
            y_frame = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))
            y.append(y_frame)

            u_data = f.read(uv_size)
            if len(u_data) < uv_size:
                break
            u_frame = np.frombuffer(u_data, dtype=np.uint8).reshape((height // 2, width // 2))
            u.append(u_frame)

            v_data = f.read(uv_size)
            if len(v_data) < uv_size:
                break
            v_frame = np.frombuffer(v_data, dtype=np.uint8).reshape((height // 2, width // 2))
            v.append(v_frame)

    return y, u, v


def read_y_only_component(file_path, width, height):
    y_size = width * height
    y = []

    with open(file_path, 'rb') as f:
        while True:
            y_data = f.read(y_size)
            if len(y_data) < y_size:
                break
            uv_data = f.read(y_size // 2)
            if len(uv_data) < y_size // 2:
                break

            y_frame = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))
            y.append(y_frame)

    return y


def dump_y_component(y_frames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for frame_index, y in enumerate(y_frames):
        y_data = y.astype(np.uint8).tobytes()

        output_file = os.path.join(output_folder, f"frame_{frame_index:04d}.y")

        with open(output_file, 'wb') as f:
            f.write(y_data)


def dump_reconstructed_frame(y_frames, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for frame_index, y in enumerate(y_frames):
        y_data = y.astype(np.uint8).tobytes()

        output_file = os.path.join(output_folder, f"reconstructed_frame_{frame_index:04d}.y")

        with open(output_file, 'wb') as f:
            f.write(y_data)