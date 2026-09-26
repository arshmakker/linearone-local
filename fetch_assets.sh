#!/bin/sh
# Download the frozen encoder (int8 bge-small-en-v1.5, MIT licence) and build the three scorer files. Needs curl and python3 with numpy, onnxruntime, tokenizers.
set -e
cd "$(dirname "$0")"; A=snake/demo_assets; mkdir -p $A
B=https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main
[ -f $A/model_int8.onnx ] || curl -L -o $A/model_int8.onnx $B/onnx/model_quantized.onnx
[ -f $A/vocab.txt ] || curl -L -o $A/vocab.txt $B/vocab.txt
[ -f $A/tokenizer.json ] || curl -L -o $A/tokenizer.json $B/tokenizer.json
(cd snake && python3 make_snake_scorer.py)
(cd games && python3 make_flap_scorer.py && python3 make_flap_scorer.py cube)
