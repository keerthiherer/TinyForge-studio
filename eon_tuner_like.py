"""
EON Tuner-like Hyperparameter Sweep Tool for IoT ML

This script helps you quickly run hyperparameter sweeps that explore different pre-processing and model architectures optimized for your defined objectives and IoT device constraints.

Usage:
    - Run the script and follow the prompts to select your IoT device.
    - The script will automatically select suitable hyperparameters and architectures for the chosen device.
    - It will then perform a sweep and report the best configuration.

Note: This is a template and should be extended with your dataset and training logic.
"""

import json

from ml_utils import read_choice

# Example device profiles (extend as needed)
IOT_DEVICES = {
    # ========= Linux / SBCs (run Python + OpenCV + ONNX) =========
    "raspberry_pi_4": {
        "display_name": "Raspberry Pi 4",
        "processor_family": "ARM Cortex-A (Linux)",
        "clock_mhz_default": 1500,
        "max_ram_mb": 2048,
        "max_flash_mb": 16384,
        "preferred_model_types": ["cnn", "mlp"],
        "preferred_runtimes": ["python", "onnxruntime", "opencv"],
        "accelerators": ["cpu"],
        "connectivity": ["wifi", "ethernet", "bluetooth"],
        "quantization": False,
        "opencv_expected": True,
        "edge_only": False,
    },
    "raspberry_pi_5": {
        "display_name": "Raspberry Pi 5",
        "processor_family": "ARM Cortex-A (Linux)",
        "clock_mhz_default": 1800,
        "max_ram_mb": 4096,
        "max_flash_mb": 16384,
        "preferred_model_types": ["cnn", "mlp"],
        "preferred_runtimes": ["python", "onnxruntime", "opencv"],
        "accelerators": ["cpu"],
        "connectivity": ["wifi", "ethernet", "bluetooth"],
        "quantization": False,
        "opencv_expected": True,
        "edge_only": False,
    },
    "grove_vision_ai": {
        "display_name": "Seeed Studio Grove Vision AI",
        "processor_family": "ARM + accelerator (SBC/AI box)",
        "clock_mhz_default": 1000,
        "max_ram_mb": 1024,
        "max_flash_mb": 8192,
        "preferred_model_types": ["cnn"],
        "preferred_runtimes": ["tflite", "opencv"],
        "accelerators": ["npu (if available)"],
        "connectivity": ["wifi", "serial"],
        "quantization": True,
        "opencv_expected": True,
        "edge_only": True,
    },
    "jetson_nano": {
        "display_name": "NVIDIA Jetson Nano",
        "processor_family": "ARM + NVIDIA GPU",
        "clock_mhz_default": 1400,
        "max_ram_mb": 4096,
        "max_flash_mb": 16384,
        "preferred_model_types": ["cnn"],
        "preferred_runtimes": ["tensorrt", "cuda", "onnxruntime"],
        "accelerators": ["gpu (cuda)"],
        "connectivity": ["wifi", "ethernet"],
        "quantization": False,
        "opencv_expected": True,
        "edge_only": False,
    },
    "jetson_orin": {
        "display_name": "NVIDIA Jetson Orin",
        "processor_family": "ARM + NVIDIA GPU",
        "clock_mhz_default": 1700,
        "max_ram_mb": 8192,
        "max_flash_mb": 32768,
        "preferred_model_types": ["cnn"],
        "preferred_runtimes": ["tensorrt", "cuda", "onnxruntime"],
        "accelerators": ["gpu (cuda)"],
        "connectivity": ["wifi", "ethernet"],
        "quantization": False,
        "opencv_expected": True,
        "edge_only": False,
    },

    # ========= MCUs =========
    "arduino_uno": {
        "display_name": "Arduino Uno",
        "processor_family": "AVR (8-bit)",
        "clock_mhz_default": 16,
        "max_ram_mb": 0.002,
        "max_flash_mb": 0.03,
        "preferred_model_types": ["tinyml"],
        "preferred_runtimes": ["tflite_micro"],
        "accelerators": [],
        "connectivity": ["serial"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },
    "arduino_uno_q": {
        "display_name": "Arduino Uno Q",
        "processor_family": "Arduino-class (Q variant)",
        "clock_mhz_default": 240,
        "max_ram_mb": 0.5,
        "max_flash_mb": 1.0,
        "preferred_model_types": ["tinyml"],
        "preferred_runtimes": ["tflite_micro"],
        "accelerators": [],
        "connectivity": ["serial"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },
    "arduino_nano_33_ble_sense": {
        "display_name": "Arduino Nano 33 BLE Sense",
        "processor_family": "nRF52840 (Cortex-M4F)",
        "clock_mhz_default": 64,
        "max_ram_mb": 0.256,
        "max_flash_mb": 1.0,
        "preferred_model_types": ["tinyml", "mlp"],
        "preferred_runtimes": ["tflite_micro", "cmsis_nn"],
        "accelerators": ["DSP (CMSIS-NN if available)"],
        "connectivity": ["bluetooth", "serial"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },
    "esp32": {
        "display_name": "ESP32 DevKit V1",
        "processor_family": "Xtensa LX6 (dual-core)",
        "clock_mhz_default": 240,
        "max_ram_mb": 320,
        "max_flash_mb": 4096,
        "preferred_model_types": ["tinyml", "cnn", "mlp"],
        "preferred_runtimes": ["tflite_micro", "esp_dl"],
        "accelerators": ["esp-dl (if used)"],
        "connectivity": ["wifi", "bluetooth"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },
    "esp_eye": {
        "display_name": "ESP Eye",
        "processor_family": "ESP32 + vision front-end",
        "clock_mhz_default": 240,
        "max_ram_mb": 320,
        "max_flash_mb": 8192,
        "preferred_model_types": ["tinyml", "cnn"],
        "preferred_runtimes": ["tflite_micro", "esp_dl"],
        "accelerators": ["esp-dl (if used)"],
        "connectivity": ["wifi"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },
    "esp32_cam": {
        "display_name": "ESP32-CAM",
        "processor_family": "ESP32 + camera",
        "clock_mhz_default": 240,
        "max_ram_mb": 320,
        "max_flash_mb": 8192,
        "preferred_model_types": ["tinyml", "cnn"],
        "preferred_runtimes": ["tflite_micro", "esp_dl"],
        "accelerators": ["esp-dl (if used)"],
        "connectivity": ["wifi"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },

    "esp8266": {
        "display_name": "ESP8266 NodeMCU",
        "processor_family": "Xtensa LX106",
        "clock_mhz_default": 80,
        "max_ram_mb": 0.05,
        "max_flash_mb": 2.0,
        "preferred_model_types": ["tinyml"],
        "preferred_runtimes": ["tflite_micro"],
        "accelerators": [],
        "connectivity": ["wifi"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },

    # ========= STM32 =========
    "stm32f103c8_blue_pill": {
        "display_name": "STM32F103C8 Blue Pill",
        "processor_family": "STM32F1 (Cortex-M3)",
        "clock_mhz_default": 72,
        "max_ram_mb": 0.02,
        "max_flash_mb": 0.064,
        "preferred_model_types": ["tinyml"],
        "preferred_runtimes": ["tflite_micro", "cmsis_nn"],
        "accelerators": ["cmsis_nn"],
        "connectivity": ["serial"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },
    "stm32h7": {
        "display_name": "STM32H7 Series",
        "processor_family": "STM32H7 (Cortex-M7)",
        "clock_mhz_default": 480,
        "max_ram_mb": 512,
        "max_flash_mb": 2048,
        "preferred_model_types": ["cnn", "tinyml", "mlp"],
        "preferred_runtimes": ["tflite_micro", "cmsis_nn"],
        "accelerators": ["cmsis_nn"],
        "connectivity": ["serial", "i2c", "spi"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },

    # ========= Nordic / AI boards =========
    "nrf52840": {
        "display_name": "nRF52840 DK",
        "processor_family": "nRF52840 (Cortex-M4F)",
        "clock_mhz_default": 64,
        "max_ram_mb": 0.256,
        "max_flash_mb": 1.0,
        "preferred_model_types": ["tinyml", "mlp"],
        "preferred_runtimes": ["tflite_micro", "cmsis_nn"],
        "accelerators": ["cmsis_nn"],
        "connectivity": ["bluetooth", "usb"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },

    "k210": {
        "display_name": "Kendryte K210",
        "processor_family": "RISC-V + AI accelerator",
        "clock_mhz_default": 400,
        "max_ram_mb": 64,
        "max_flash_mb": 16,
        "preferred_model_types": ["tinyml", "cnn"],
        "preferred_runtimes": ["onnxruntime", "esp_dl (if ported)"],
        "accelerators": ["k210 ai"],
        "connectivity": ["serial", "wifi (if module)"],
        "quantization": True,
        "opencv_expected": False,
        "edge_only": True,
    },
}



# Example hyperparameter search space
HYPERPARAM_SPACE = {
    "cnn": {
        "num_layers": [1, 2, 3],
        "filters": [8, 16, 32],
        "kernel_size": [3, 5],
        "activation": ["relu", "tanh"],
    },
    "mlp": {
        "num_layers": [1, 2, 3],
        "units": [16, 32, 64],
        "activation": ["relu", "tanh"],
    },
    "tinyml": {
        "num_layers": [1, 2],
        "units": [8, 16],
        "activation": ["relu"],
    },
}

def choose_device():
    print("Available IoT devices:")
    device_name = read_choice("Select your IoT device (number): ", list(IOT_DEVICES.keys()))
    return device_name, IOT_DEVICES[device_name]

def suggest_hyperparams(device_profile):
    suggestions = {}
    for model_type in device_profile["preferred_model_types"]:
        space = HYPERPARAM_SPACE[model_type]
        # Simple heuristic: pick the smallest values to fit device constraints
        params = {k: v[0] for k, v in space.items()}
        suggestions[model_type] = params
    return suggestions

def run_sweep(device_name, device_profile, suggestions):
    print(f"\nRunning hyperparameter sweep for {device_name}...")
    for model_type, params in suggestions.items():
        print(f"\nModel type: {model_type}")
        print(f"Suggested hyperparameters: {json.dumps(params)}")
        estimated_ram = estimate_ram_mb(model_type, params)
        fits = estimated_ram <= device_profile["max_ram_mb"]
        print(f"Estimated RAM use: {estimated_ram:.2f} MB")
        print(f"Fits RAM budget: {'yes' if fits else 'no'}")

def estimate_ram_mb(model_type, params):
    if model_type == "cnn":
        weights = params["num_layers"] * params["filters"] * params["kernel_size"] * params["kernel_size"] * 4
    else:
        weights = params["num_layers"] * params["units"] * params["units"] * 4
    return max(weights / (1024 * 1024), 0.01)

def main():
    device_name, device_profile = choose_device()
    suggestions = suggest_hyperparams(device_profile)
    run_sweep(device_name, device_profile, suggestions)

if __name__ == "__main__":
    main()
