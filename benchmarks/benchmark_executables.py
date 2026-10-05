import time
from src.tools.executables import list_path_executables


def run_benchmark(iterations: int = 10):
    print(f"Running benchmark over {iterations} iterations...")

    # Warmup / initial call
    t0 = time.perf_counter()
    list_path_executables(package_manager="none")
    t_first = time.perf_counter() - t0
    print(f"First call duration: {t_first * 1000:.3f} ms")

    # Repeated calls
    t_start = time.perf_counter()
    for _ in range(iterations):
        list_path_executables(package_manager="none")
    t_end = time.perf_counter()

    avg_time = (t_end - t_start) / iterations
    total_time = t_end - t_start
    print(f"Total time for {iterations} iterations: {total_time * 1000:.3f} ms")
    print(f"Average time per call: {avg_time * 1000:.3f} ms")

    return t_first, total_time, avg_time


if __name__ == "__main__":
    run_benchmark()
