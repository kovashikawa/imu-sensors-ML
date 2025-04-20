# IMU Sensors ML - Fall Detection System

A machine learning system for fall detection using Inertial Measurement Unit (IMU) sensors.

## Project Structure

```
imu-sensors-ML/
├── configs/            # Configuration files
├── data/               # Data directory
│   ├── 01_raw/         # Raw IMU data
│   ├── 02_intermediate/# Intermediate processed data
│   └── 03_processed/   # Final processed data
├── logs/               # Log files
├── models/             # Saved models
├── notebooks/          # Jupyter notebooks for exploration
├── scripts/            # Utility scripts
├── src/                # Source code
│   └── fall_detector/  # Main package
├── tests/              # Unit tests
└── reports/            # Analysis reports
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imu-sensors-ML.git
cd imu-sensors-ML
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Install additional requirements:
```bash
pip install -r requirements.txt
```

## Data Preprocessing

The preprocessing pipeline normalizes column names, builds features, and creates sequences for training:

1. Make sure your raw data follows this structure:
```
data/01_raw/
└── sub1/                  # Subject directory
    ├── ADLs/              # Activity category
    │   ├── trial1.xlsx    # Trial file
    │   └── trial2.xlsx
    └── Falls/
        ├── trial1.xlsx
        └── trial2.xlsx
```

2. Run the preprocessing script:
```bash
# Basic usage
python scripts/preprocess_data.py

# With debug logging
python scripts/preprocess_data.py --debug

# With custom config
python scripts/preprocess_data.py --config path/to/config.yaml
```

3. Check the processed data in `data/03_processed/`

## Training

1. First, make sure you've run the preprocessing step above.

2. Train the model:
```bash
python scripts/train_mlflow.py
```

3. View results in MLflow:
```bash
mlflow ui --port 5001
```

## Troubleshooting

- If you encounter issues with column names, check that the raw data columns can be properly mapped to the expected format (`acc_x`, `acc_y`, `acc_z`, `gyr_x`, `gyr_y`, `gyr_z`).
- For skipped trials, check the `skipped_trials.txt` file in the processed data directory.
- If you get "module not found" errors, make sure you've installed the package in development mode with `pip install -e .`

## Project Configuration

The project configuration is in `configs/config.yaml`. Key settings:

- `data`: Paths to data directories
- `data_loading`: Settings for loading raw data
- `features`: Feature selection and engineering
- `preprocessing`: Sequence length and sliding window settings
- `model`: Model architecture and parameters
- `training`: Training parameters
- `mlflow`: MLflow tracking settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.