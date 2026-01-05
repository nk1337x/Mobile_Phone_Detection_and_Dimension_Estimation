# Mobile Phone Detection and Dimension Estimation

Real-time mobile phone detection and dimension estimation using YOLOv8 and camera calibration.

## Features

- Detects mobile phones using YOLOv8
- Measures phone dimensions (width and height) in cm
- Estimates distance from camera
- Camera calibration for accurate measurements
- **Persistent calibration** - calibrate once, reuse across sessions

## Requirements

```bash
pip install opencv-python numpy ultralytics
```

## Usage

1. Run the program:
   ```bash
   python main.py
   ```

2. **First time only**: Hold phone at 35cm and press `c` five times to calibrate
   - Calibration is saved to `calibration_data.json` and loaded automatically on next run

3. View real-time measurements

## Controls

- `c` - Calibrate (at 35cm distance, only needed once)
- `r` - Reset calibration (deletes saved calibration file)
- `q` - Quit

## How It Works

Uses camera focal length calibration to convert pixel measurements to real-world dimensions based on detected bounding box size and known phone dimensions.
