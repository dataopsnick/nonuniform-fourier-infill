import asyncio
import numpy as np
import tritonclient.grpc.aio as grpcclient
import time

async def trigger_request(client, timestamps):
    """Fires a single request and measures round-trip latency."""
    inputs = [
        grpcclient.InferInput("TIMESTAMPS", timestamps.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(timestamps)
    
    start_time = time.perf_counter()
    result = await client.infer(model_name="nudft_imputer", inputs=inputs)
    latency = time.perf_counter() - start_time
    
    output = result.as_numpy("RECONSTRUCTED_SIGNAL")
    return latency, output

async def main():
    client = grpcclient.InferenceServerClient(url="localhost:8001")
    
    num_requests = 1000
    queries_per_request = 512  # Batch size of missing ticks to infill
    
    # Warmup
    dummy = np.zeros(queries_per_request, dtype=np.float32)
    for _ in range(10):
        await trigger_request(client, dummy)
        
    print(f"Blasting {num_requests} async requests to Triton...")
    
    # Generate non-uniform timestamps
    requests_data = [
        np.sort(np.random.uniform(0, 100, size=queries_per_request)).astype(np.float32)
        for _ in range(num_requests)
    ]
    
    tasks = [trigger_request(client, req) for req in requests_data]
    results = await asyncio.gather(*tasks)
    
    latencies = np.array([r[0] for r in results]) * 1000  # Convert to ms
    
    print("-" * 30)
    print("Latency Distribution (ms):")
    print(f"p50: {np.percentile(latencies, 50):.3f} ms")
    print(f"p95: {np.percentile(latencies, 95):.3f} ms")
    print(f"p99: {np.percentile(latencies, 99):.3f} ms")
    print(f"Max: {np.max(latencies):.3f} ms")
    print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())