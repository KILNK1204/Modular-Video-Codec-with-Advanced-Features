# Modular-Video-Codec-with-Advanced-Features

Decoder and encoder that support motion estimation, transform coding, multiple reference frames, variable block sizes, and advanced rate control mechanisms.

## Results

![Resulting Img](src/output/Screenshot%202025-07-31%20140714.png)

## Features

### Core Encoding/Decoding Pipeline
- Block-based motion estimation and compensation
- Intra and inter prediction (P/I-frames)
- Residual computation, 2D DCT transform, quantization, and inverse transform
- Bitstream generation using Exponential-Golomb coding and RLE
- Block reconstruction and Y-only frame output

### Advanced Functionalities
- **Multiple Reference Frames** (up to 4)
- **Variable Block Size (VBS)** with RD-cost comparison
- **Fractional Motion Estimation (FME)** with half-pixel accuracy
- **Fast Search Algorithms** using motion vector prediction
- **Rate Control** (RCflag = 1, 2, 3)
- **Scene Change Detection and Multi-Pass Encoding**
- **Parallelism** (block-level and frame-level using threading)
