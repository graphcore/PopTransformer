# Benchmark

- Sharding ChatGLM-6B to 32x Mk2 IPU
  - precision=FP16, topk=1

| input length | max length | Throughput(token/s) | Latency pre output token(ms/token) |
| --- | --- |--- | --- |
 32 | 256 | 312 | 3.205 |
 32 | 1024 | 286 | 3.494 |
 128 | 256 | 297 | 3.360 |
 128 | 1024 | 280 | 3.564 |
 512 | 1024 | 256 | 3.895 |

------
