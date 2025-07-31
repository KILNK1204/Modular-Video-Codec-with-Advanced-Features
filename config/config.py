QP = 5
I_Period = 10
Block_Size = 16
search_range = 16
lambda_value = 0.5  # best value

Resolution = {
    "width": 352,
    "height": 288,
}

Features = {
    "nRefFrames": 1,  # 1 mean off, Enable for minor PSNR improvement, may lower PSNR
    "VBSEnable": True,  # Enable for significant PSNR improvement
    "FMEEnable": True,  # Enable for minor PSNR improvement
    "FastME": True,     # Enable for fast encoding improvement, may lower PSNR
    "RCflag": 0,  # Enable RC mode control
    "targetBR": 2000000,  # Target bitrate in bps  2.4mbps = 24000000，960kbps = 960000 7000000, 2400000 360000
    "fps": 30,  # Frame rate
    "parallel_mode": 0,   # Parallel process mode
    "cif_i_table": [(0, 37923), (1, 37086), (2, 29687), (3, 23052), (4, 17230), (5, 12496), (6, 8626), (7, 5599),
                    (8, 3362), (9, 1598), (10, 780), (11, 340)],  # Temporary QP->bit-count per row table
    "cif_p_table": [(0, 62420), (1, 61822), (2, 53263), (3, 44095), (4, 34355), (5, 25283), (6, 17963), (7, 10931),
                    (8, 6415), (9, 3023), (10, 1184), (11, 772)],
    "qcif_i_table": [(0, 20133), (1, 20006), (2, 16148), (3, 12568), (4, 9563), (5, 6996), (6, 4775), (7, 3057),
                     (8, 1857), (9, 917), (10, 424), (11, 195)],  # Temporary QP->bit-count per row table
    "qcif_p_table": [(0, 31619), (1, 31220), (2, 27190), (3, 22559), (4, 17408), (5, 12830), (6, 9043), (7, 5508),
                     (8, 3214), (9, 1524), (10, 589), (11, 384)]
}

Paths = {
    "Input_Video": "./src/resources/foreman_cif-1.yuv",
    "Output_Reconstructed_frames": "reconstructed_frames",
    "Output_Directory": "./src/output",
    "Output_Modes_Path": "modes.txt",
    "Output_Mvs_Path": "mvs.txt",
    "Output_Residuals_Path": "residuals.txt",
    "Output_QP_path": "qp.txt",
    "Output_frame_type_path": "frame_type.txt"
}
