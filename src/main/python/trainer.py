import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
from typing import List, Dict, Optional, Tuple, Callable, Any
import os
import torch.distributed as dist
import json


OPTIMIZER_FILE_NAME = "optimizer.pt"
SCHEDULER_FILE_NAME = "scheduler.pt"
MODEL_FILE_PATH = "model.pth"


def default_collect_f(x):
    if len(x) <= 0:
        return {}

    if not isinstance(x[0], dict):
        raise TypeError("default data type must be dict")

    keys = [i for i in x[0].keys()]
    for key in keys:
        if not isinstance(x[0][key], list):
            raise TypeError("value type must be list")

    res = x[0]
    for i in range(1, len(x)):
        for key in keys:
            res[key] = res[key] + x[i][key]

    for key in keys:
        res[key] = torch.Tensor(res[key])

    return res


def get_linear_schedule_with_warmup(optimizer, num_of_warmup_steps, num_of_train_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_of_warmup_steps:
            return float(current_step) / float(max(1, num_of_warmup_steps))
        return max(
            0.0, float(num_of_train_steps - current_step) / float(max(1, num_of_train_steps - num_of_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class TrainingArgument(object):
    def __init__(self,
                 num_train_epoch: int,
                 train_batch_size: int,
                 learning_rate: float,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.0,
                 device: torch.device = torch.device("cpu"),
                 out_put_dir: str = ".",
                 arg_mapping: Dict[str, object] = None,
                 rank: int = -1,
                 local_rank: int = -1,
                 backend: str = "nccl",
                 init_method: str = None,
                 world_size: int = 0,
                 do_train: bool = False,
                 do_eval: bool = False,
                 eval_batch_size: int = 1
                 ):
        self.device = device
        self.num_train_epoch = num_train_epoch
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.out_put_dir = out_put_dir
        self.arg_mapping = arg_mapping
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.do_train = do_train
        self.do_eval = do_eval
        self.eval_batch_size = eval_batch_size

        if self.rank != -1 and self.local_rank != -1 and self.backend == 'nccl':
            self.device = torch.device("cuda:" + str(self.local_rank))

        str_map = {"device": self.device,
                   "num_train_epoch": self.num_train_epoch,
                   "train_batch": self.train_batch_size,
                   "learning_rate": self.learning_rate,
                   "warmup_steps": self.warmup_steps,
                   "weight_decay": self.weight_decay,
                   "out_put_dir": self.out_put_dir,
                   "arg_mapping": self.arg_mapping,
                   "rank": self.rank,
                   "local_rank": self.local_rank,
                   "backend": self.backend,
                   "init_method": self.init_method,
                   "world_size": self.world_size,
                   "do_train": self.do_train,
                   "do_eval": self.do_eval,
                   "eval_batch": self.eval_batch_size}
        self.str = json.dumps(str_map, ensure_ascii=False)

    def __str__(self):
        return self.str


class Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 training_arg: TrainingArgument,
                 train_dataset: Optional[Dataset],
                 eval_dataset: Optional[Dataset] = None,
                 collect_fn: Callable[[List[Any]], Any] = default_collect_f,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None):
        self.training_arg = training_arg
        self.model = model.to(self.training_arg.device)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collect_fn = collect_fn
        self.optimizers = optimizers

        self.current_epoch = 0
        self.current_step = 0
        self.total_step_per_epoch = 0

        self.current_loss = 0
        self.loss_from_training_start = []

        self.distributed_train_mode = False
        if self.training_arg.local_rank != -1 and \
                self.training_arg.rank != -1 and \
                self.training_arg.world_size > 0 and \
                self.training_arg.init_method is not None and \
                self.training_arg.backend is not None:
            self.distributed_train_mode = True

    def get_training_metric(self) -> Dict[str, object]:
        metric = {
            "current_epoch": self.current_epoch,
            "total_epoch": self.training_arg.num_train_epoch,

            "current_step": self.current_epoch,
            "total_step_per_epoch": self.total_step_per_epoch,

            "current_loss": self.current_loss,

            "train_data_size": len(self.train_dataset),

            "training_arg": self.training_arg,
            "loss_from_training_start": self.loss_from_training_start
        }

        return metric

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")

        if self.training_arg.local_rank != -1 and self.training_arg.rank != -1:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            train_sampler = DistributedSampler(self.train_dataset)

        data_loader = DataLoader(dataset=self.train_dataset,
                                 batch_size=self.training_arg.train_batch_size,
                                 sampler=train_sampler,
                                 collate_fn=self.collect_fn,
                                 drop_last=False)
        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset]) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a eval_dataset")

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if self.training_arg.local_rank != -1 and self.training_arg.rank != -1:
            eval_sampler = DistributedSampler(eval_dataset, shuffle=True)
        else:
            eval_sampler = DistributedSampler(eval_dataset)

        data_loader = DataLoader(dataset=eval_dataset,
                                 batch_size=self.training_arg.eval_batch_size,
                                 sampler=eval_sampler,
                                 collate_fn=self.collect_fn,
                                 drop_last=False)
        return data_loader

    def get_optimizers(self, num_of_train_steps: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        if self.optimizers is not None:
            return self.optimizers

        weight_decay = float(self.training_arg.weight_decay)
        lr = float(self.training_arg.learning_rate)
        num_of_warmup_steps = self.training_arg.warmup_steps

        optm = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        schedular = get_linear_schedule_with_warmup(optimizer=optm,
                                                    num_of_warmup_steps=num_of_warmup_steps,
                                                    num_of_train_steps=num_of_train_steps)
        self.optimizers = optm, schedular
        return self.optimizers

    def train(self, model_path: Optional[str] = None):

        if self.distributed_train_mode:
            dist.init_process_group(backend=self.training_arg.backend,
                                    init_method=self.training_arg.init_method,
                                    world_size=self.training_arg.world_size,
                                    rank=self.training_arg.rank)

        train_dataloader = self.get_train_dataloader()
        t_total = len(train_dataloader) * self.training_arg.num_train_epoch

        if self.distributed_train_mode:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.training_arg.local_rank, ],
                                                                   output_device=self.training_arg.local_rank,
                                                                   find_unused_parameters=True)
        optimizer, scheduler = self.get_optimizers(t_total)

        model = self.model
        train_epoch_start = 0
        train_epoch_iter = range(train_epoch_start, self.training_arg.num_train_epoch)

        model.train()

        train_sampler = train_dataloader.sampler
        need_set_epoch = hasattr(train_sampler, "set_epoch")

        self.total_step_per_epoch = len(train_dataloader)
        epoch_loss_sum = 0
        for epoch in train_epoch_iter:

            if need_set_epoch:
                train_sampler.set_epoch(epoch)
            self.current_epoch = epoch

            for step, input_v in enumerate(train_dataloader):
                self.current_step = step
                for k, v in input_v.items():
                    input_v[k] = v.to(self.training_arg.device)

                optimizer.zero_grad()
                output = model(**input_v)
                step_loss = output[0]
                step_loss.backward()
                optimizer.step()

                self.current_loss = float(step_loss.cpu().data)
                epoch_loss_sum = epoch_loss_sum + step_loss.data

            epoch_loss = epoch_loss_sum / self.total_step_per_epoch
            epoch_loss_sum = 0
            self.loss_from_training_start.append(float(epoch_loss.data))
            scheduler.step()

        output_path = self.training_arg.out_put_dir
        if output_path is None or len(output_path) == 0:
            return
        if not self.distributed_train_mode:
            self.save_model(output_path)
        else:
            if self.training_arg.rank == 0:
                self.save_model(output_path)

    def save_model(self, output_dir: str):
        if os.path.isfile(output_dir):
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.optimizers[0].state_dict(), output_dir + os.sep + OPTIMIZER_FILE_NAME)
        torch.save(self.optimizers[1].state_dict(), output_dir + os.sep + SCHEDULER_FILE_NAME)
        if self.distributed_train_mode:
            torch.save(self.model.module.state_dict(), output_dir + os.sep + OPTIMIZER_FILE_NAME)
        else:
            torch.save(self.model.state_dict(), output_dir + os.sep + OPTIMIZER_FILE_NAME)
