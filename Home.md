# Recent feature additions and improvements in FBGEMM

We fully integrated FBGEMM with PyTorch. FBGEMM is the performant backend for quantized inference on servers for PyTorch. In addition, we have made significant improvements to FBGEMM to better serve the existing models as well as to prepare for up-and-coming use cases. These improvements include better 2D-and-3D groupwise convolutions, 64-bit integer GEMM operations for privacy aware ML use cases, and Sparse GEMMs/Convolutions for efficiently supporting sparsity in neural networks.  

Rest of the document highlights the major feature additions to FBGEMM. In addition, there were many other minor improvements and bug fixes.

## Improvements to Existing Features

*PyTorch Integration*: Quantization was a major feature in [version 1.3 release of PyTorch][1]. We implemented compute heavy operators such as Linear and convolution based on efficient FBGEMM kernels. FBGEMM's support for int8 GEMM with float input/output was very helpful at quickly implementing PyTorch dynamic quantization. During the same time, we also have added the following features for PyTorch integration:

* *Unified convolution interface*: Convolutions come in all shapes and sizes, i.e., regular convolution, depthwise convolutions, groupwise convolutions, etc. A high-performance implementation that works for depthwise is not the best way to implement regular convolutions due to the varying arithmetic intensity of these operations. To keep the PyTorch conv operator implementation simple in PyTorch, we added a unified interface for convolution in FBGEMM. This single API redirects to the most efficient available implementation in FBFEMM. Through this single interface, we can handle all types of 2D and 3D convolutions efficiently. 

* *Quantized/Nonquantized bias handling*: Bias is quantized using activation scale in Caffe2 models. For eager mode quantization in PyTorch, activation scale is not available until the operator execution time, so offline quantization of bias with activation scale is not possible. To handle this case, we added support for non-quantized bias in FBGEMM post-processing operations. Flexible design of post-processing operation in FBGEMM allowed us to handle this case without many changes and now we support both quantized and float bias in GEMM and convolution operations.


### Optimizing for more diverse convolution cases

* *Better Groupwise convolutions and 3D groupwise convolutions*: Groupwise convolutions with small number of channels per group (2, 4, 8 or 16 channels per group) don’t run efficiently with general implementation of convolutions. The reason is that smaller number of channels (reduction happens across this dimension) makes arithmetic intensity of such operations low. These convolutions are a significant portion of ResNext-101 models and models using Octave Convolutions. We created a specialized implementation for such convolutions which works on multiple groups at a time to take advantage of full SIMD width and use a different pre-packing of weights. This implementation was done recently and provided 4-10x speedups over existing implementation for stride = 2.   60x speedups were observed for 3D groupwise convolutions with respect to the base fused im2col based implementation. This new implementation also supports parallelization. For example, a typical 2D groupwise convolution shape is 3.4x faster with 4 threads in comparison to a single thread implementing. Caffe2 operators were also updated to make use of these 3D groupwise convolution. 

* *Dilated Convolutions*: Dilated convolutions are used in many networks in the wild and FBGEMM didn’t have support for dilated convolutions until recently because we didn’t have any quantized models using dilated convolutions. We added quantized dilated convolutions using an im2col based implementation in FBGEMM.
* *5x5 depthwise kernels*: 3x3 kernel is the most popular kernel size for depthwise convolutions but we noticed that a few important models were using 5x5 kernels for depthwise convolutions. We added an efficient implementation of 5x5 depthwise kernels as well*.*


### Better support for recent HWs

* *VNNI support*: A few operators were implemented to support VNNI instruction set. VNNI instruction set will be available in the most recent and upcoming generation of processors (e.g., Cooper Lake CPUs). 
* *AVX512 and AVX512-YMM FP16 GEMM*: We improved the performance FP16 FBGEMM on Intel Xeon Gold 6138 processor using AVX512 (the previous version was only using AVX2). We also added an option called avx512_ymm that uses AVX512 instructions but with YMM registers (32 YMM registers are still available rather than 16 in AVX2). This is for Skylake Xeon-D where we can only execute 1 instruction using ZMM per cycle, while the ZMM instruction slows down clock-frequency. On a recommendation model using FP16 kernels, we observed an end-to-end 10% improvement in latency.


### Better scalability

* *Multi-threading*: Latency sensitive use cases may want to use 2 or more threads for a single model for inference. We improved our multi-threading support by adding 2-dimensional parallelism and almost all the GEMMs and convolutions implementation now support multi-threading. For example, quantized Resnet-50 inference with a batch size of 1 takes ~130 ms on Intel Xeon D-2191 processor with 1 thread while the same model runs in ~38 ms with 4 threads.
* *Shared code cache*: FBGEMM generates code at runtime as needed and stores generated code in a code cache for reuse. Thus a kernel is generated on the first call and subsequent calls use the same code. We take care to keep the generated code minimal to keep the memory footprint of FBGEMM minimal. However, we didn’t share this code cache among threads. Each thread had its own private code cache. We unified the code cache and now all threads share a single code cache thus further minimizing the memory footprint in a mutli-threaded environment. 


### Miscellaneous Updates

* *Fast vectorized FP16/BF16 ↔ FP32 conversion routines*: We added fast vectorized FP16/BF16 ↔ FP32 conversion routines in AVX2/AVX512, which are used by recommendation models and resulted in memory and network bandwidth savings.

* *Fast transpose routines: * We added the edge-case handlers for the built-in vectorized transpose routines in FBGEMM, and provided 10%~20% speedups over MKL somatcopy routine in many cases.
* *Better CI/CD:* For OSS FBGEMM, we had limited build configurations and tests though each PR was only pushed if internal builds and tests ran fine. Recently we added Linux and MacOS builds as well as running of tests using GitHub Actions on each PR and push. The plan is expand this list to add more build configurations including builds and tests using MSVC. 

## New Feature Additions

*Embedding Operators for Recommendation Models: *Embedding operators like EmbeddingSpMDM (aka SparseLengthsSum in Caffe2 and EmbeddingBag in PyTorch) and (RowWise)SparseAdagrad are one of the most time consuming parts of recommendation models (link to DLRM). We added JIT code generation of these operators that specialize for data type, embedding dimension, the presence of weight, and so on. If we specialize for all these combinations (as in caffe2/perfkernels), the code size can be prohibitive. Even for cases specialized by caffe2/perfkernels, we observe FBGEMM’s JIT’ed EmbeddingSpMDM provides ~2x speedups over caffe2/perfkernels due to better control over register allocation. We tested our implementations on recommendation model in production settings and the implementation achieved an 18-25% speedup (average latency), and for another model achieved over 2% speedup (average latency).

*Sparse Operations:* Inference can be further optimized by sparse operations (sparse GEMMs and convolutions). Recently we added sparse GEMMs and convolutions to the FBGEMM library. These kernels in FBGEMM support AVX2, AVX512, and VNNI. They support INT8 and FP32 data types. These kernels fully inline non-zero values and positions of the sparse matrix allowing register blocking and much higher instruction-level parallelism. Depending on the structure (unstructured or block) and levels of sparsity, we can achieve speedups with specialized sparse kernels. We observe 2x-4x speedups with 90% sparsity.

*64-bit Matrix-Matrix Multiplications:* Privacy preserving machine learning is a fast growing area. The compute in this field happens on 64-bit integer data, hence, an efficient implementation of key 64-bit operation is essential for speeding up privacy preserving machine learning models. The compute workhorse in this area are 64-bit GEMMs and convolutions. Currently there exist no good high-performance implementation of 64-bit GEMMs on current generation of CPUs. Therefore, we have added an initial 64-bit GEMMs in FBGEMM and it achieves 10.5 GOPs/sec on  Intel Xeon Gold 6138 processor with turbo off. It is 3.5x faster than the existing implementation that runs at 3 GOps/sec. This is the first iteration of the 64-bit GEMM implementation. This initial implementation will be further improved via tuning of blocking parameters, optimization of remainder cases etc. 

## Other Usage/mentions of FBGEMM

[Recently released wav2letter inference framework][4] uses FBGEMM FP16 to deliver efficient performance for automatic speech recognition task. 

[A Neural Machine Translation (NMT) paper from Microsoft][2] uses FBGEMM as one of its baselines. In this paper, FBGEMM INT8 achieved 2x throughput compared to MKL FP32 for batch size 1 and 1.3x throughput compared to MKL FP32 for batch size 32. 

[NGEMM paper][3] cites FBGEMM and briefly mentions that there has been an increasing interest in developing high-performance low-precision libraries.

FBGEMM team would like to acknowledge and greatly appreciates the contributions of [@zlateski](https://github.com/zlateski) to sparse kernels and unified code cache, [@efiks](https://github.com/efiks) to FP16 performance optimizations and [@ykim362](https://github.com/ykim362) and [@peterjc123](https://github.com/peterjc123) to building and running FBGEMM with MSVC. [@shz0116](https://github.com/shz0116) is a new member of the team and working on making FBGEMM better including CI/CD among other things. 

## Summary

Overall, last few months have been really productive for FBGEMM. FBGEMM team fruitfully collaborated with the PyTorch quantization team to implement key quantized operators in PyTorch using FBGEMM’s efficient kernels and along the way added many features to FBGEMM as needed by the integration effort. Any further improvements in the FBGEMM library for speeding up server inference benefit PyTorch quantization. 

[1] https://github.com/pytorch/pytorch/releases/tag/v1.3.0
[2] https://www.microsoft.com/en-us/research/publication/from-research-to-production-and-back-ludicrously-fast-neural-machine-translation/
[3] https://arxiv.org/pdf/1910.00178.pdf
[4] https://ai.facebook.com/blog/online-speech-recognition-with-wav2letteranywhere/