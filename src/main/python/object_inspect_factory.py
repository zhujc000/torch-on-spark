from torch.utils.data import Dataset
import os
import importlib
import sys
import torch
from typing import Dict
import torch.nn as nn


class ObjectInspectFactory(object):
    def __init__(self,
                 env_path: str,
                 model_py_name: str, model_class_name: str, model_init: Dict[str, object],
                 train_dataset_py_name: str, train_dataset_class_name: str,
                 train_dataset_init: Dict[str, object],
                 eval_dataset_py_name: str = None, eval_dataset_class_name: str = None,
                 eval_dataset_init: Dict[str, object] = None,
                 optimizer_py_name: str = None, optimizer_class_name: str = None,
                 optimizer_init: Dict[str, object] = None,
                 scheduler_py_name: str = None, scheduler_class_name: str = None,
                 scheduler_init: Dict[str, object] = None,
                 collect_fn_py_name: str = None, collect_fn_name: str = None
                 ):
        self.env_path = env_path

        if not (os.path.exists(self.env_path) and os.path.isdir(self.env_path)):
            raise ValueError("env_path error")
        sys.path.append(self.env_path)

        self.model_py_name = model_py_name
        self.model_class_name = model_class_name
        self.model_init = model_init

        self.train_dataset_py_name = train_dataset_py_name
        self.train_dataset_class_name = train_dataset_class_name
        self.train_dataset_init = train_dataset_init

        if eval_dataset_py_name is not None:
            self.eval_dataset_py_name = eval_dataset_py_name
        else:
            self.eval_dataset_py_name = self.train_dataset_py_name

        if eval_dataset_class_name is not None:
            self.eval_dataset_class_name = eval_dataset_class_name
        else:
            self.eval_dataset_class_name = self.train_dataset_class_name
        self.eval_dataset_init = eval_dataset_init

        self.optimizer_py_name = optimizer_py_name
        self.optimizer_class_name = optimizer_class_name
        self.optimizer_init = optimizer_init

        self.scheduler_py_name = scheduler_py_name
        self.scheduler_class_name = scheduler_class_name
        self.scheduler_init = scheduler_init

        self.collect_fn_py_name = collect_fn_py_name
        self.collect_fn_name = collect_fn_name

    def __class_obj(self, py_name: str, class_name: str, init: Dict[str, object]):
        abs_path = self.env_path + os.sep + py_name
        package_name = py_name.strip(".py")
        spec = importlib.util.spec_from_file_location(package_name, abs_path)
        spec.loader.load_module()
        mo = importlib.import_module(package_name, package=package_name)
        if not hasattr(mo, class_name):
            raise ValueError(package_name + " not in " + abs_path)

        class_instance = getattr(mo, class_name)
        class_obj = class_instance(**init)
        return class_obj

    def __func_obj(self, func_py_name: str, func_name: str):
        abs_path = self.env_path + os.sep + func_py_name
        package_name = func_py_name.strip(".py")
        spec = importlib.util.spec_from_file_location(package_name, abs_path)
        spec.loader.load_module()
        mo = importlib.import_module(package_name, package=package_name)
        if not hasattr(mo, func_name):
            raise ValueError(package_name + " not in " + abs_path)

        func_instance = getattr(mo, func_name)
        return func_instance

    def build_model(self) -> nn.Module:
        return self.__class_obj(self.model_py_name,
                                self.model_class_name,
                                self.model_init)

    def build_train_dataset(self) -> Dataset:
        return self.__class_obj(self.train_dataset_py_name,
                                self.train_dataset_class_name,
                                self.train_dataset_init)

    def build_eval_dataset(self) -> Dataset:
        return self.__class_obj(self.eval_dataset_py_name,
                                self.eval_dataset_class_name,
                                self.eval_dataset_init)

    def build_optimizer(self) -> torch.optim.Optimizer:
        return self.__class_obj(self.optimizer_py_name,
                                self.optimizer_class_name,
                                self.optimizer_init)

    def build_scheduler(self):
        return self.__class_obj(self.scheduler_py_name,
                                self.scheduler_class_name,
                                self.scheduler_init)

    def build_collect_fn(self):
        return self.__func_obj(self.collect_fn_py_name,
                               self.collect_fn_name)
