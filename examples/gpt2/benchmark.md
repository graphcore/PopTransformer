# Benchmark

- Sharding GPT2-XL to 8x C600
  - max length=264, precision=FP16, topk=1

| batch size | Throughput(token/s) | Latency(ms/token) |
| --- | --- | --- |
 1 | 949 | 1.053 |
 8 | 3550 | 2.253 |
 16 | 4562 | 3.507 |
 24 | 5475 | 4.383 |

------

- Sharding GPT2-XL to 8x MK2
  - max length=264, precision=FP16, topk=1

| batch size | Throughput(token/s) | Latency(ms/token) |
| --- | --- | --- |
 1 | 840 | 1.190 |
 8 | 3139 | 2.548 |
 16 | 4043 | 3.957 |
 24 | 4858 | 4.940 |