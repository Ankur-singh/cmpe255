import onnx
import torch
import argparse
import subprocess
from pathlib import Path

def main(args):
    path = Path(args['out_dir'])
    dummy_input = torch.randn(1, 1, 28, 28)
    model = torch.load(path/'model.pt', map_location="cpu")

    # TorchScript CPU
    model.cpu()
    model.eval()
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(path/'model_scripted_cpu.pt')
    print('TorchScript CPU model saved')

    # TorchScript GPU
    model.cuda()
    model.eval()
    traced_script_module = torch.jit.trace(model, dummy_input.cuda())
    traced_script_module.save(path/'model_scripted_gpu.pt')
    print('TorchScript GPU model saved')

    # ONNX : https://onnxruntime.ai/docs/get-started/with-python.html, 
    onnx_path = path/'model_onnx.onnx'
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        onnx_path,
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('ONNX model saved')

    ## OpenVINO IR : https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/102-pytorch-onnx-to-openvino/102-pytorch-onnx-to-openvino.ipynb
    # Construct the command for Model Optimizer.
    mo_command = f"""mo
                    --input_model "{onnx_path}"
                    --input_shape "[1,1,28,28]"
                    --data_type FP16
                    --output_dir "{onnx_path.parent}/openvino"
                    """
    mo_command = " ".join(mo_command.split())
    subprocess.run(mo_command, shell=True, check=True)
    # print(f"`{mo_command}`")
    # print("run the following Model Optimizer command to convert the ONNX model to OpenVINO:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', help='output directory', required=True)
    args = vars(parser.parse_args())
    print(args)
    main(args)