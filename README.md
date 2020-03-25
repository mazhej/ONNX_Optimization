# ONNX_Optimization
### Intro
Convert torch models and yolo to onnx, and optimize them using both ONNX and onnxruntime.

ONNX provides a C++ library for performing arbitrary optimizations on ONNX models, as well as a growing list of prepackaged optimization passes. The library also provides a convenient in-memory representation that is much more convenient to manipulate than the raw protobuf structs, and converters to and from the protobuf format.

The primary motivation is to share work between the many ONNX backend implementations. Not all possible optimizations can be directly implemented on ONNX graphs - some will need additional backend-specific information - but many can, and our aim is to provide all such passes along with ONNX so that they can be re-used with a single function call.

You may be interested in invoking the provided passes, or in implementing new ones (or both). 
To see the original page https://github.com/onnx/onnx/blob/master/docs/Optimizer.md#onnx-optimizer

###  How to use
you first need to convert your models( resnet, mobilenet, inception,...) to onnx using torch_to_onnx.py ans then optimize them using optimize_onnx_models.py
