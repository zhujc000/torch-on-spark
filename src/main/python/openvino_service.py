from openvino.inference_engine import IECore
import numpy as np
from typing import Dict


class OpenVinoService(object):
    def __init__(self,
                 model_path: str):
        self.ie = IECore()
        self.model_path = model_path
        self.net = self.ie.read_network(model=self.model_path)
        self.model = self.ie.load_network(network=self.net, device_name='CPU')

    def infer(self, in_put: Dict[str, object]):
        ipt = {}
        for i, j in in_put.items():
            ipt[i] = np.array(j)
        ort_outs = self.model.infer(in_put=ipt)
        return ort_outs
