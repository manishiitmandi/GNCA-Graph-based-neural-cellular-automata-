import torch

def display_gpu_memory(device=0):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    device = torch.device(f"cuda:{device}")
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"[GPU {device.index}] Allocated: {allocated:.2f} GiB | Reserved: {reserved:.2f} GiB")

def clear_gpu_memory(device=0):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    print("Before clearing GPU memory:")
    display_gpu_memory(device)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("After clearing GPU memory:")
    display_gpu_memory(device)

if __name__ == "__main__":
    clear_gpu_memory()
