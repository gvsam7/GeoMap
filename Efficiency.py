import torch
import time
import numpy as np
from ptflops import get_model_complexity_info

from Main import networks, arguments  # uses your existing functions


def main():

    args = arguments()

    # Device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Instantiate model exactly as in training
    model = networks(
        architecture=args.architecture,
        in_channels=args.in_channels,
        num_classes=2,  # binary classification (cement / landcover)
        pretrained=args.pretrained,
        requires_grad=args.requires_grad,
        global_pooling=args.global_pooling,
        version=args.version
    ).to(device)

    model.eval()

    print(model)

    # -----------------------------
    # 1) Parameters
    # -----------------------------
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,} ({n_params/1e6:.2f} M)")

    # -----------------------------
    # 2) FLOPs / GFLOPs
    # -----------------------------
    macs, params = get_model_complexity_info(
        model,
        (args.in_channels, args.height, args.width),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"Computational complexity: {macs}")
    print(f"Parameters (ptflops): {params}")

    # -----------------------------
    # 3) Inference latency + memory
    # -----------------------------
    if torch.cuda.is_available():

        batch_size = args.batch_size
        dummy_input = torch.randn(
            batch_size,
            args.in_channels,
            args.height,
            args.width
        ).to(device)

        # Warmup
        for _ in range(50):
            _ = model(dummy_input)
            torch.cuda.synchronize()

        # Timed runs
        runs = 200
        times = []

        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(runs):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.time()

            times.append((end - start) * 1000)  # ms per batch

        mean_batch = np.mean(times)
        std_batch = np.std(times)

        ms_per_sample = mean_batch / batch_size
        throughput = 1000 / ms_per_sample

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2

        print(f"\nBatch size: {batch_size}")
        print(f"Latency: {ms_per_sample:.3f} ms/sample")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print(f"Peak GPU memory: {peak_memory:.2f} MB")

    else:
        print("\nCUDA not available — latency & memory not measured.")


if __name__ == "__main__":
    main()