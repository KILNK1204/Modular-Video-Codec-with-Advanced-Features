import os
import numpy as np
from PIL import Image


def yuv420_to_yuv444(y, u, v):
    yuv444_frames = []

    for i in range(len(y)):
        y_frame = y[i]
        u_frame = u[i]
        v_frame = v[i]

        u_444 = np.repeat(np.repeat(u_frame, 2, axis=0), 2, axis=1)
        v_444 = np.repeat(np.repeat(v_frame, 2, axis=0), 2, axis=1)

        yuv444_frames.append([y_frame, u_444, v_444])

    return yuv444_frames


def yuv_to_rgb(yuv444_frames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for frame_index, (y, u, v) in enumerate(yuv444_frames):
        y = y.astype(np.float32) - 16
        u = u.astype(np.float32) - 128
        v = v.astype(np.float32) - 128

        r = 1.164 * y + 1.596 * v
        g = 1.164 * y - 0.392 * u - 0.813 * v
        b = 1.164 * y + 2.017 * u

        r = np.clip(r, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)

        rgb_frame = np.stack((r, g, b), axis=-1)
        img = Image.fromarray(rgb_frame, 'RGB')

        img.save(f"{output_folder}/frame_{frame_index:04d}.png")
