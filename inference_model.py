import os
import numpy as np
import onnxruntime as ort


# noinspection PyProtectedMember
class OnnxInferenceModel:
    def __init__(self, model_path: str = "", force_cpu: bool = False, model_name: str = "model.onnx"):
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.model_name = model_name

        if os.path.isdir(self.model_path):
            self.model_path = os.path.join(self.model_path, self.model_name)

        if not os.path.exists(self.model_path):
            raise Exception(f"Model path ({self.model_path}) does not exist")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" and not force_cpu \
            else ["CPUExecutionProvider"]

        self.model = ort.InferenceSession(self.model_path, providers=providers)
        self.metadata = self.model.get_modelmeta().custom_metadata_map

        if self.metadata:
            for key, value in self.metadata.items():
                setattr(self, key, value)

        if self.force_cpu:
            self.model.set_providers(["CPUExecutionProvider"])

        self.input_shape = self.model.get_inputs()[0].shape[1:]
        self.input_name = self.model._inputs_meta[0].name
        self.output_name = self.model._outputs_meta[0].name

