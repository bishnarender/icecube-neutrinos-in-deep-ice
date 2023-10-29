import polars as pl
import pandas as pd
from src.fastai_fix import *
from tqdm.notebook import tqdm
from src.dataset import (
    RandomChunkSampler,
    LenMatchBatchSampler,
    IceCubeCache,
    DeviceDataLoader,
)
from src.loss import loss, loss_vms
from fastxtend.vision.all import EMACallback
from tqdm import tqdm
from src.utils import seed_everything, WrapperAdamW
import argparse
import json
import torch
import os, sys
from src.models import (
    DeepIceModel,
    EncoderWithDirectionReconstructionV22,
    EncoderWithDirectionReconstructionV23,
)
from pdb import set_trace
from IPython.display import display

import graphviz
graphviz.set_jupyter_format('png')
from torchview import draw_graph

def read_config_file(file_path):
    with open(file_path, "r") as f:
        config_data = json.load(f)
    return config_data


def train(config):
    ds_train = IceCubeCache(
        config["PATH"],
        mode="train",
        L=config["L"],
        selection=config["SELECTION"],
        reduce_size=0.125,
    )
    ds_train_len = IceCubeCache(
        config["PATH"],
        mode="train",
        L=config["L"],
        selection=config["SELECTION"],
        reduce_size=0.125,
        mask_only=True,
    )
    
    # ds_train.chunks => [200000, 200000, 200000, 200000, 200000, ... ]
    
    sampler_train = RandomChunkSampler(ds_train_len, chunks=ds_train.chunks)
    
    len_sampler_train = LenMatchBatchSampler(
        sampler_train, batch_size=config["BS"], drop_last=True
    )
    
    dl_train = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_train,
            batch_size=config["BS"], #1, # config["BS"] my_
            batch_sampler=None, #len_sampler_train, # None my_
            num_workers=0,  #config["NUM_WORKERS"], # 0 my_
            persistent_workers=False, #True, # False my_
            shuffle=False,
        )
    )

    
    ds_val = IceCubeCache(
        config["PATH"], mode="eval", L=config["L_VALID"], selection=config["SELECTION"]
    )
    ds_val_len = IceCubeCache(
        config["PATH"],
        mode="eval",
        L=config["L_VALID"],
        selection=config["SELECTION"],
        mask_only=True,
    )
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(
        sampler_val, batch_size=config["BS_VALID"], drop_last=False
    )
    
    dl_val = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_val, batch_sampler=len_sampler_val, num_workers=0
        )
    )

    # DataLoaders => <class 'fastai.data.core.DataLoaders'>

    data = DataLoaders(dl_train, dl_val)
    # config["MODEL_KWARGS"] => {'dim': 384, 'dim_base': 128, 'depth': 12, 'head_size': 32}

    model = config["MODEL"](**config["MODEL_KWARGS"])
    if config["WEIGHTS"]:
        print("Loading weights from ...", config["WEIGHTS"])
        model.load_state_dict(torch.load(config["WEIGHTS"]))
    model = nn.DataParallel(model)
    model = model.cuda()
    cbs = [
        GradientClip(3.0),
        CSVLogger(),
        SaveModelCallback(monitor="loss", comp=np.less, every_epoch=True),
        GradientAccumulation(n_acc=4096 // config["BS"]),
    ]
    if config["EMA"]:
        cbs.append(EMACallback())

    learn = Learner(
        data,
        model,
        cbs=cbs, # list of callbacks.
        path=config["OUT"],
        
        # if no loss function is specified, the default loss function for the data loaders object is used. The dataloader object selects an appropriate loss function based on the type of the target.
        loss_func=config["LOSS_FUNC"],
        metrics=[config["METRIC"]],
        opt_func=partial(WrapperAdamW, eps=1e-7),
    ).to_fp16()
    
    # Learner => <class 'fastai.learner.Learner'>
    # learn.dls => <fastai.data.core.DataLoaders object at 0x7f44a670a5b0>
    
#     for x in learn.dls:
#         for z in x:            
#             draw_graph(learn.model, input_data = [z[0]], expand_nested=True, save_graph=True).visual_graph                         
#             break
#         break
#     return 1

    # fit self.model for n_epoch using the 1cycle policy.                 (https://fastai1.fast.ai/callbacks.one_cycle.html)
    learn.fit_one_cycle( 
        n_epoch=8,
        lr_max=1e-5,
        wd=0.05,
        pct_start=0.01,
        div=config["DIV"], # default: 25
        div_final=config["DIV_FINAL"], # default: 25
        moms=tuple(config["MOMS"]) if config["MOMS"] else None,
    )
    # if moms = [0.95,0.85]: the momentum (moms) is the first beta in Adam (or the momentum in SGD/RMSProp). When you pass along (0.95,0.85) it means going from 0.95 to 0.85 during the warmup then from 0.85 to 0.95 in the annealing, but it only changes the first beta in Adam, yes.


def main():
    parser = argparse.ArgumentParser(
        description="Create a model from a JSON config file."
    )
    parser.add_argument("config_file", type=str, help="Path to the JSON config file.")
    parser.add_argument(
        "configs",
        nargs="*",
        metavar=("KEY", "VALUE"),
        help="The JSON config key to override and its new value.",
    )

    args = parser.parse_args()
    config_file_path = args.config_file

    config_data = read_config_file(config_file_path)

    if args.configs:
        for config_key, config_value in zip(args.configs[::2], args.configs[1::2]):
            keys = config_key.split(".")
            last_key = keys.pop()

            current_data = config_data
            for key in keys:
                current_data = current_data[key]

            try:
                value = json.loads(config_value)
            except json.JSONDecodeError:
                value = config_value

            current_data[last_key] = value

    print("Training with the following configuration:")
    print(json.dumps(config_data, indent=4))
    print("_______________________________________________________")

    config_data["MODEL"] = getattr(sys.modules[__name__], config_data["MODEL"])
    config_data["LOSS_FUNC"] = getattr(sys.modules[__name__], config_data["LOSS_FUNC"])
    config_data["METRIC"] = getattr(sys.modules[__name__], config_data["METRIC"])

    seed_everything(config_data["SEED"])
    os.makedirs(config_data["OUT"], exist_ok=True)
    
    train(config_data)


if __name__ == "__main__":
    main()
