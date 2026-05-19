#!/usr/bin/env python3
"""
Memory optimization utilities for GPU training.
"""

import torch
import gc


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return "No GPU available"

    info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        info.append(
            f"GPU {i}: {allocated:.2f}/{reserved:.2f}/{total:.2f} GB (allocated/reserved/total)"
        )

    return "\n".join(info)


def optimize_batch_size(model,
                        sample_input,
                        initial_batch_size=256,
                        device='cuda'):
    """Find optimal batch size for available GPU memory."""
    if not torch.cuda.is_available():
        return initial_batch_size // 4  # Conservative for CPU

    batch_size = initial_batch_size
    while batch_size > 1:
        try:
            clear_gpu_memory()
            # Try forward pass with current batch size
            test_batch = {
                k: v.repeat(batch_size, 1)
                for k, v in sample_input.items()
            }
            with torch.no_grad():
                _ = model(**test_batch)
            clear_gpu_memory()
            print(f"Batch size {batch_size} fits in memory")
            return batch_size
        except torch.cuda.OutOfMemoryError:
            batch_size = batch_size // 2
            print(f"Batch size too large, trying {batch_size}")

    return 1


class GradientAccumulator:
    """Helper for gradient accumulation to simulate larger batches."""

    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_step(self):
        """Check if optimizer should step."""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0

    def scale_loss(self, loss):
        """Scale loss for gradient accumulation."""
        return loss / self.accumulation_steps
