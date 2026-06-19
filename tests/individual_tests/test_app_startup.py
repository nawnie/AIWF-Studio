from aiwf.app import _friendly_device_name, _friendly_library_message


def test_friendly_device_name_trims_cuda_details():
    assert (
        _friendly_device_name("CUDA (NVIDIA GeForce RTX 4070 Ti SUPER, 16.0 GB VRAM, torch cuda 12.4)")
        == "NVIDIA GeForce RTX 4070 Ti SUPER"
    )


def test_friendly_device_name_maps_cpu_mode():
    assert _friendly_device_name("CPU (slow - install CUDA PyTorch for GPU acceleration)") == "CPU mode"


def test_friendly_library_message_when_models_exist():
    assert _friendly_library_message(2, 0) == "Library ready with 2 base models."
    assert _friendly_library_message(1, 3) == "Library ready with 1 base model and 3 LoRAs."


def test_friendly_library_message_when_no_models_exist():
    assert _friendly_library_message(0, 0) == (
        "No base models were found yet. Add one in Models or import another library in Settings."
    )
