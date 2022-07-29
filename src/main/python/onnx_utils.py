import torch
import torch.nn as nn
from typing import List


def transform2onnx(model: nn.Module,
                   dummy_input,
                   out_put_file_path: str,
                   input_names: List[str],
                   output_names: List[str],
                   opset_version: int = 12):

    if not out_put_file_path.endswith(".onnx"):
        out_put_file_path = out_put_file_path + ".onnx"

    torch.onnx.export(model,
                      dummy_input,
                      out_put_file_path,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=opset_version)
