import os
import sys
import importlib
import traceback
from types import ModuleType
from typing import Callable, Optional
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRDPARTY_ROOT = os.path.join(PROJECT_ROOT, "thirdparty")

VENDOR_PATHS = {
    "nvidia_0.9.1": os.path.join(THIRDPARTY_ROOT, "nvidia_vllm_0_9_1"),
    "nvidia_0.8.5": os.path.join(THIRDPARTY_ROOT, "nvidia_vllm_0_8_5"),
    "amd_0.9.1": os.path.join(THIRDPARTY_ROOT, "amd_vllm_0_9_1"),
    "hygon": os.path.join(THIRDPARTY_ROOT, "hygon_vllm_0_8_5_post1"),
}

_PROVIDER = os.environ.get("VLLM_VERSION_PROVIDER", "nvidia_vllm_0_9_1").lower()
if _PROVIDER not in VENDOR_PATHS:
    raise ValueError(f"Invalid MEGATRON_PROVIDER='{_PROVIDER}'")

_SELECTED_ROOT = VENDOR_PATHS[_PROVIDER]
_VENDOR_MEGA_PATH = os.path.join(_SELECTED_ROOT, "vllm")

if not os.path.isdir(_SELECTED_ROOT):
    raise ImportError(f"Vendor root not found: {_SELECTED_ROOT}")

if _SELECTED_ROOT not in sys.path:
    sys.path.insert(0, _SELECTED_ROOT)

def _init_vendor_submodules(vendor_root: str):
    """
    Initialize and update git submodules inside the selected vendor root.
    """
    try:
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive", vendor_root],
            cwd=vendor_root,
        )
        print(f"[INFO] Git submodules initialized in {vendor_root}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to init submodules in {vendor_root}")
        raise

def _install_vendor(vendor_root: str):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", vendor_root],
        cwd=vendor_root,
    )
_init_vendor_submodules(_SELECTED_ROOT)
_install_vendor(_SELECTED_ROOT)