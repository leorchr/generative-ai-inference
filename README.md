# Project Setup

## Prerequisites

Before starting, ensure you have the following installed:

* [Git](https://git-scm.com/downloads)
* [CMake](https://cmake.org/download/) (minimum required version `3.20`)
* MSVC Compiler (the project only works with MSVC Compiler)

## Download Required ONNX Model Files

Download the T5 ONNX model files from [this GitHub repository](https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/t5/model):

* `t5-decoder-with-lm-head-12.onnx`
* `t5-encoder-12.onnx`

And put the files into

```bash
/ressources/t5_base/
```

Download the M2M100 ONNX model files from [this HuggingFace repository](https://huggingface.co/optimum/m2m100_418M/tree/main):

* `encoder_model.onnx`
* `decoder_model.onnx`
* `decoder_with_past_model.onnx`

And put the files into

```bash
/ressources/m2m100_418M/
```

## Cloning the Repository

This project includes Git submodules. Clone the repository recursively:

```bash
git clone --recursive https://github.com/leorchr/generative-ai-inference.git
```
