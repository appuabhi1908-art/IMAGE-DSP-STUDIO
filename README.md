# Image DSP Studio

A Python desktop application for 2D spatial filtering and edge detection, built from first principles using NumPy — no OpenCV or pre-built computer vision library required.

![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- Load any color photo and view it alongside its grayscale conversion
- Apply spatial filters in real time:
  - Box blur
  - Gaussian blur
  - Median filter (noise removal)
  - Sharpen
- Apply edge detection algorithms:
  - Sobel
  - Prewitt
  - Laplacian
  - Canny (full multi-stage pipeline: Gaussian smoothing → gradient computation → non-maximum suppression → hysteresis thresholding)
- Live sliders for kernel size and Gaussian sigma
- Save the filtered result to a file
- Clean, modern dark-themed interface built with CustomTkinter

## How it works

Every filter is implemented manually as a 2D convolution over a NumPy array — no `cv2.blur()`, no `cv2.Canny()`. This makes the underlying math fully visible: a small kernel matrix is slid across the image, multiplying and summing pixel neighborhoods to produce each output pixel.

## Requirements

- Python 3.10+
- See `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
python my_app2.py
```

## Usage

1. Click **Open Image** and choose a photo.
2. Pick a filter from the dropdown.
3. Adjust **Kernel Size** and **Sigma** to control filter strength.
4. Click **Save Result** to export the filtered image.

## Tech stack

- **NumPy** — convolution and all filter math
- **Pillow (PIL)** — image loading, format conversion, saving
- **CustomTkinter** — graphical user interface

## License

MIT
