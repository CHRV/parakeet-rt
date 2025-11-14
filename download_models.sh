
curl -L "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.onnx" --output models/decoder_joint-model.onnx
curl -L "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx" --output models/encoder-model.onnx
curl -L "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx.data" --output models/encoder-model.onnx.data
curl -L "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/nemo128.onnx" --output models/nemo128.onnx
curl -L "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt" --output models/vocab.txt
curl -L "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip" --output models/vosk-model-en-us-0.22-lgraph.zip