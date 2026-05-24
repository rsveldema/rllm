This is an experimental LLM.
Just a vehicle to experiment with.

How to Build
===============

```bash
mkdir build
cd build
cmake ..
cmake --build .
```


How to use
=============

To train:

 ./build/rllm --train --filter simple --method window3 --epochs 10

Prompt mode:

 ./build/rllm


TODOs:
==============

more accuracy:
- multi-stage
- transformer arch
- check impact of more layers and/or wider layers

more features:
- check prompt mode

be faster:
- optimistic concurrency
- CI for testing various options overnight
- OpenCL instead of OpenMP
