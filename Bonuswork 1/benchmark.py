import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from utils import ModelsEngine, show_values

def main(args):
    dummy_img = np.random.randn(28, 28)
    n_samples = args['n_samples']
    formats = ['pytorch', 'torch_script', 'onnx','openvino_onnx', 'openvino_ir']
    path = Path(args['models_dir'])

    # Load models.
    models = ModelsEngine(path, formats)

    model_dict = {
        'PyTorch model on CPU': models.pytorch_cpu_inference,
        'PyTorch model on GPU': models.pytorch_gpu_inference,
        'TorchScript model on CPU': models.torch_script_cpu_inference,
        'TorchScript model on GPU': models.torch_script_gpu_inference,
        'ONNX model on GPU': models.onnx_gpu_inference,
        'ONNX model in OpenVINO Runtime/CPU': models.openvino_onnx_cpu_inference,
        'ONNX model in OpenVINO Runtime/GPU': models.openvino_onnx_gpu_inference,
        'OpenVINO IR model in OpenVINO Runtime/CPU': models.openvino_ir_cpu_inference,
        'OpenVINO IR model in OpenVINO Runtime/GPU': models.openvino_ir_gpu_inference,
    }

    inf_times = {}
    for model_name, model in model_dict.items():
        time = model.timeit(dummy_img, n=n_samples)
        inf_times[model_name] = time/n_samples
        print(f"{model_name}: {time/n_samples:.3f} seconds per image, "
                f"FPS: {n_samples/time:.2f}")
        print('')

    print('Done!')

    # Plot results.
    df = pd.DataFrame.from_dict(inf_times, orient='index', columns=['Inference time (s)'])
    df['FPS'] = 1/df['Inference time (s)']
    df['percentage'] = df['FPS']/df.loc['PyTorch model on CPU', 'FPS']

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x=df.index, y=df['Inference time (s)'], )
    show_values(ax)
    plt.xticks(rotation=90)
    plt.ylabel('Inference time (s)')
    plt.title('Inference time per model')
    plt.tight_layout()
    plt.savefig('images/benchmark_inference_time.png')
    
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x=df.index, y=df['FPS'], )
    show_values(ax)
    plt.xticks(rotation=90)
    plt.ylabel('FPS')
    plt.title('FPS per model')
    plt.tight_layout()
    plt.savefig('images/benchmark_fps.png')

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x=df.index, y=df['percentage'], )
    show_values(ax)
    plt.xticks(rotation=90)
    plt.ylabel('percentage FPS compared to "PyTorch model on CPU"')
    plt.title('Percentage FPS per model compared to "PyTorch model on CPU"')
    plt.tight_layout()
    plt.savefig('images/benchmark_percentage.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models_dir", type=str, default='model_weights')
    parser.add_argument('-n', "--n_samples", type=int, default=200, help='number of inputs. Default : 200', required=False)
    args = vars(parser.parse_args())
    print(args)
    main(args)