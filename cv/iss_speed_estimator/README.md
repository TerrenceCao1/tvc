# ISS Speed Estimation with Optical Flow

This is a technology demonstrator, similar commands and techniques can be used by the rocket to estimate traveled distance during flight.

This program estimates the speed of the International Space Station (ISS) relative to the Earth's surface using optical flow techniques. It processes video frames to track feature points, calculates their average movement, and uses this information to estimate the ISS's speed based on a scaling factor derived from its altitude and camera field of view (FOV).

## Features

- Tracks feature points across video frames using **Lucas-Kanade Optical Flow**.
- Calculates average pixel movement and converts it to real-world speed (in km/h).
- Uses exponential moving averages and frame-based averages for speed smoothing.
- Dynamically adds and updates tracking points to ensure robust estimation.
- Visualizes tracking points and their movement paths on the video.
- Includes options to toggle visualization and debug print statements.

---

## Installation

Before running the program, ensure that you have Python 3.7 or later installed.

### Install Required Packages

You can install the required Python packages using `pip`. Run the following command:

```bash
pip install opencv-python numpy

## Example usage
python iss_speed_estimator.py --video "data\ISS_video.mp4" --mask "data\ISS_mask.jpg"
