import time
import torch
import numpy as np
import onnxruntime as ort
from openvino.runtime import Core

# source : https://www.statology.org/seaborn-barplot-show-values/
def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


class PyTorchInference:
    def __init__(self, model):
        self.model = model
        self.device = next(self.model.parameters()).device

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            output = self.model(x)
        return output[0].cpu().numpy()

    def timeit(self, x, n=100):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        x = x.to(self.device)
        with torch.no_grad():
            start = time.time()
            for _ in range(n):
                output = self.model(x)
            end = time.time()
        return (end - start) / n

class ONNXInference:
    def __init__(self, session):
        self.session = session
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x):
        x = x.reshape(1, 1, 28, 28)
        ort_inputs = {self.input_name: x.astype(np.float32)}
        output = self.session.run(None, ort_inputs)
        return output[0][0]

    def timeit(self, x, n=100):
        x = x.reshape(1, 1, 28, 28) / 255.0
        ort_inputs = {self.input_name: x.astype(np.float32)}
        start = time.time()
        for _ in range(n):
            output = self.session.run(None, ort_inputs)
        end = time.time()
        return (end - start) / n

class OpenVINOInference:
    def __init__(self, model_ir, device):
        ie = Core()
        self.compiled_model_ir = ie.compile_model(model=model_ir, device_name=device)

        # Get input and output layers.
        self.output_layer_ir = self.compiled_model_ir.output(0)

    def predict(self, x):
        x = x.reshape(1, 1, 28, 28)
        # Run inference on the input image.
        output = self.compiled_model_ir([x])[self.output_layer_ir]
        return output[0]

    def timeit(self, x, n=100):
        x = np.array(x, dtype=np.float32).reshape(1, 1, 28, 28) / 255.0
        start = time.time()
        for _ in range(n):
            output = self.compiled_model_ir([x])[self.output_layer_ir]
        end = time.time()
        return (end - start) / n

class ModelsEngine:
    def __init__(self, path, formats=None):
        self.path = path
        self.models = self.load_models(formats)

    def load_models(self, formats):
        if 'pytorch' in formats:
            model_cpu = torch.load(self.path/'model.pt')
            model_cpu.eval()
            self.pytorch_cpu_inference = PyTorchInference(model_cpu)

            model_gpu = torch.load(self.path/'model.pt', map_location="cuda")
            model_gpu.eval()
            self.pytorch_gpu_inference = PyTorchInference(model_gpu)

        if 'torch_script' in formats:
            model_scripted_cpu = torch.jit.load(self.path/'model_scripted_cpu.pt')
            model_scripted_cpu.eval()
            self.torch_script_cpu_inference = PyTorchInference(model_scripted_cpu)

            model_scripted_gpu = torch.jit.load(self.path/'model_scripted_gpu.pt')
            model_scripted_gpu.eval()
            self.torch_script_gpu_inference = PyTorchInference(model_scripted_gpu)

        if 'onnx' in formats:
            session = ort.InferenceSession(str(self.path/'model_onnx.onnx'), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.onnx_cpu_inference = ONNXInference(session)
            self.onnx_gpu_inference = self.onnx_cpu_inference

        if 'openvino_onnx' in formats or 'openvino_ir' in formats:
            ie = Core()
            model_onnx = ie.read_model(model=self.path/'model_onnx.onnx')
            self.openvino_onnx_cpu_inference = OpenVINOInference(model_onnx, 'CPU')
            self.openvino_onnx_gpu_inference = OpenVINOInference(model_onnx, 'GPU')

            model_ir = ie.read_model(model=self.path/'openvino/model_onnx.xml')
            self.openvino_ir_cpu_inference = OpenVINOInference(model_ir, 'CPU')
            self.openvino_ir_gpu_inference = OpenVINOInference(model_ir, 'GPU')
        
    def get_model(self, name, device):
        model_name = f'{name}_{device}_inference'
        return getattr(self, model_name)