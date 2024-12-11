import torch
import subprocess

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Check PyTorch's CUDA version
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# Check NVIDIA driver version
try:
    nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode("utf-8").strip()
    print(f"NVIDIA driver version: {nvidia_smi_output}")
except:
    print("Unable to fetch NVIDIA driver version. Make sure nvidia-smi is available.")

 
