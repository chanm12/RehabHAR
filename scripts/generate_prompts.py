"""
RehabHAR – Example: Generate a prompt from sensor data
Based on: LanHAR prompt_generation/prompt_example.py

Quick CLI example showing how to load a .npy file and
generate an analysis prompt using the prompt_generation package.
"""

import numpy as np
from prompt_generation.prompt import generate_promt


if __name__ == "__main__":

    data = np.load("data/uci_data.npy")
    acc = data[0, :, :3].astype(float)
    gyro = data[0, :, 3:6].astype(float)
    dataset_name = "uci"
    prompt = generate_promt(acc, gyro, dataset_name, '')
    print(prompt)
