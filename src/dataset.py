import polars as pl
import pandas as pd
import gc, os, random, math, sys
import numpy as np
import pickle5 as pickle
from tqdm.notebook import tqdm
from collections import OrderedDict
from bisect import bisect_right
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import (
    Iterator,
    Optional,
    Sized,
)
from IPython.display import display

def prepare_sensors(path):
    # pd.read_csv(os.path.join(path, "sensor_geometry.csv")).head(3) =>
    #    sensor_id       x       y       z
    # 0          0 -256.14 -521.08  496.03
    # 1          1 -256.14 -521.08  479.01
    # 2          2 -256.14 -521.08  461.99        
    
    sensors = pd.read_csv(os.path.join(path, "sensor_geometry.csv")).astype(
        {
            "sensor_id": np.int16,
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
        }
    )
    sensors["string"] = 0
    sensors["qe"] = 0  # 1

    # len(sensors), len(sensors) // 60) => 5160, 86 (strings)
    
    for i in range(len(sensors) // 60):
        start, end = i * 60, (i * 60) + 60                  # 0,60 # 60,120 # 120,180 ...
        sensors.loc[start:end, "string"] = i

        # High Quantum Efficiency is in the lower 50 DOMs - https://arxiv.org/pdf/2209.03042.pdf (Figure 1)
        if i in range(78, 86):
            start_veto, end_veto = i * 60, (i * 60) + 10   # 0,10 # 60,70 # 120,130  ...
            start_core, end_core = end_veto + 1, (i * 60) + 60  # 11,60 # 71,120 # 121,180 ...
            sensors.loc[start_core:end_core, "qe"] = 1  # 1.35

    # https://github.com/graphnet-team/graphnet/blob/b2bad25528652587ab0cdb7cf2335ee254cfa2db/src/graphnet/models/detector/icecube.py#L33-L41
    # Assume that "rde" (relative dom efficiency) is equivalent to QE
    sensors["x"] /= 500
    sensors["y"] /= 500
    sensors["z"] /= 500

    return sensors


def ice_transparency(path, datum=1950):
    # Data from page 31 of https://arxiv.org/pdf/1301.5361.pdf
    # Datum is from footnote 8 of page 29
    
    df = pd.read_csv(os.path.join(path, "ice_transparency.txt"), delim_whitespace=True)
    # df.head(3) =>
    #         depth  scattering_len  absorption_len
    # 0  1398.4            13.2            45.1
    # 1  1408.4            14.0            48.6
    # 2  1418.4            14.7            53.2

    df["z"] = df["depth"] - datum
    df["z_norm"] = df["z"] / 500
    
    df[["scattering_len_norm", "absorption_len_norm"]] = RobustScaler().fit_transform(
        df[["scattering_len", "absorption_len"]]
    )

    # These are both roughly equivalent after scaling
    # interp1d(x, y,..) => interpolate a 1-D function. x and y are arrays of values used to approximate some function f: y = f(x).
    f_scattering = interp1d(df["z_norm"], df["scattering_len_norm"])
    f_absorption = interp1d(df["z_norm"], df["absorption_len_norm"])
    return f_scattering, f_absorption


class IceCubeCache(Dataset):
    def __init__(
        self,
        path,
        mode="test",
        selection="total",
        L=128,
        cache_size=4,
        reduce_size=-1,
        mask_only=False,
    ):
        val_fnames = [
            "batch_655.parquet",
            "batch_656.parquet",
            "batch_657.parquet",
            "batch_658.parquet",
            "batch_659.parquet",
        ]
        
        with open(os.path.join(path, "Nevents.pickle"), "rb") as f:
            Nevents = pickle.load(f)
            
        self.mode, self.reduce_size, self.mask_only = mode, reduce_size, mask_only
        # put data, meta, and ice properties at the same folder
        path_ice_properties = path
        self.path_meta = os.path.join(path, "train_meta") if mode != "test" else None
        self.selection = selection

        if mode == "train" or mode == "eval":
            self.path = os.path.join(path, "train")
            self.files = [p for p in sorted(os.listdir(self.path))]
            if mode == "train":
                self.files = sorted(set(self.files) - set(val_fnames))
            else:
                self.files = val_fnames
            
            self.chunks = [Nevents[fname][selection] for fname in self.files]

        elif mode == "test":
            self.path = os.path.join(path, "test")
            self.files = [p for p in sorted(os.listdir(self.path))]

            # make sure that all files are considered regardless the number of events
            self.chunks = []
            for fname in self.files:
                ids = (
                    pl.read_parquet(os.path.join(self.path, fname))
                    .select(["event_id"])
                    .unique()
                    .to_numpy()
                    .reshape(-1)
                )
                self.chunks.append(len(ids))
            gc.collect()
        else:
            raise NotImplementedError
        
        # self.chunks[:5] => [200000, 200000, 200000, 200000, 200000]
        self.chunk_cumsum = np.cumsum(self.chunks)
        # self.chunk_cumsum[:5], self.chunk_cumsum[-1:] => [ 200000  400000  600000  800000 1000000], [130953924]
    
        self.cache, self.meta = None, None
        self.L, self.cache_size = L, cache_size
        # path => data/
            
        sensors = prepare_sensors(path)
        self.geometry = torch.from_numpy(
            sensors[["x", "y", "z"]].values.astype(np.float32)
        )
        self.qe = sensors["qe"].values
        
        # path_ice_properties => data/
        self.ice_properties = ice_transparency(path_ice_properties)
        # type(self.ice_properties) => <class 'tuple'>


    def __len__(self):
        return (
            self.chunk_cumsum[-1]
            if self.reduce_size < 0
            else int(self.reduce_size * self.chunk_cumsum[-1])
        )

    def load_data(self, fname):
        if self.cache is None:
            self.cache = OrderedDict()
        if fname not in self.cache:
            df = pl.read_parquet(os.path.join(self.path, fname))
            # df.head(3) =>
            #   sensor_id ┆ time ┆ charge ┆ auxiliary ┆ event_id │
            # │ ---       ┆ ---  ┆ ---    ┆ ---       ┆ ---      │
            # │ i16       ┆ i64  ┆ f64    ┆ bool      ┆ i64      │
            # ╞═══════════╪══════╪════════╪═══════════╪══════════╡
            # │ 3918      ┆ 5928 ┆ 1.325  ┆ true      ┆ 24       │
            # │ 4157      ┆ 6115 ┆ 1.175  ┆ true      ┆ 24       │
            # │ 3520      ┆ 6492 ┆ 0.925  ┆ true      ┆ 24       │
            # └───────────┴──────┴────────┴───────────┴────────            
            
            
            df = df.groupby("event_id").agg(
                [
                    pl.count(),
                    pl.col("sensor_id"),#.list(),
                    pl.col("time"),#.list(),
                    pl.col("charge"),#.list(),
                    pl.col("auxiliary"),#.list(),
                ]
            )
            
            # df.head(3) =>
            #   event_id ┆ count ┆ sensor_id     ┆ time          ┆ charge    ┆ auxiliary     │
            # │ ---      ┆ ---   ┆ ---           ┆ ---           ┆ ---       ┆ ---           │
            # │ i64      ┆ u32   ┆ list[i16]     ┆ list[i64]     ┆ list[f64] ┆ list[bool]    │
            # ╞══════════╪═══════╪═══════════════╪═══════════════╪═══════════╪═══════════════╡
            # │ 566904   ┆ 50    ┆ [2375, 1447,  ┆ [5904, 6019,  ┆ [0.875,   ┆ [true, true,  │
            # │          ┆       ┆ … 2855]       ┆ … 15816]      ┆ 1.225, …  ┆ … true]       │
            # │          ┆       ┆               ┆               ┆ 1.125]    ┆               │
            # │ 2257296  ┆ 73    ┆ [1757, 1165,  ┆ [6468, 6587,  ┆ [0.525,   ┆ [true, true,  │
            # │          ┆       ┆ … 3607]       ┆ … 18054]      ┆ 0.825, …  ┆ … true]       │
            # │          ┆       ┆               ┆               ┆ 1.225]    ┆               │
            # │ 2454648  ┆ 75    ┆ [2836, 1135,  ┆ [7164, 7955,  ┆ [1.025,   ┆ [true, true,  │
            # │          ┆       ┆ … 3410]       ┆ … 18850]      ┆ 0.375, …  ┆ … true]       │
            # │          ┆       ┆               ┆               ┆ 0.925]    ┆               │            

            
            if self.selection == "short":
                df = df.filter(pl.col("count") < 64)
            elif self.selection == "medium":
                df = df.filter((pl.col("count") >= 64) & (pl.col("count") < 192))
            elif self.selection == "long":
                df = df.filter(pl.col("count") >= 192)
                
            self.cache[fname] = df.sort("event_id")
            # self.cache[fname].head(3) =>
            # ┌──────────┬───────┬───────────────┬───────────────┬───────────┬───────────────┐
            # │ event_id ┆ count ┆ sensor_id     ┆ time          ┆ charge    ┆ auxiliary     │
            # │ ---      ┆ ---   ┆ ---           ┆ ---           ┆ ---       ┆ ---           │
            # │ i64      ┆ u32   ┆ list[i16]     ┆ list[i64]     ┆ list[f64] ┆ list[bool]    │
            # ╞══════════╪═══════╪═══════════════╪═══════════════╪═══════════╪═══════════════╡
            # │ 24       ┆ 61    ┆ [3918, 4157,  ┆ [5928, 6115,  ┆ [1.325,   ┆ [true, true,  │
            # │          ┆       ┆ … 104]        ┆ … 19031]      ┆ 1.175, …  ┆ … true]       │
            # │          ┆       ┆               ┆               ┆ 0.875]    ┆               │
            # │ 41       ┆ 51    ┆ [458, 4107, … ┆ [6469, 6975,  ┆ [0.925,   ┆ [true, true,  │
            # │          ┆       ┆ 1287]         ┆ … 17230]      ┆ 1.175, …  ┆ … true]       │
            # │          ┆       ┆               ┆               ┆ 0.425]    ┆               │
            # │ 59       ┆ 36    ┆ [4685, 3925,  ┆ [6601, 6896,  ┆ [1.225,   ┆ [true, true,  │
            # │          ┆       ┆ … 1548]       ┆ … 15362]      ┆ 0.775, …  ┆ … true]       │
            # │          ┆       ┆               ┆               ┆ 0.675]    ┆               │
            # └──────────┴───────┴───────────────┴───────────────┴───────────┴───────────────┘
            
            # len(self.cache), self.cache_size => 1, 4
            # self.cache.keys() => odict_keys(['batch_1.parquet'])
            # list(self.cache.keys()) => ['batch_1.parquet']
            
            if len(self.cache) > self.cache_size:
                del self.cache[list(self.cache.keys())[0]]

        if self.path_meta is None:
            return
        if self.meta is None:
            self.meta = OrderedDict()
        if fname not in self.meta:
            fidx = fname.split(".")[0].split("_")[-1]
            df = pl.read_parquet(
                os.path.join(self.path_meta, f"train_meta_{fidx}.parquet")
            )
            # df.head(3) =>
            # ──────────┬───────────────┬──────────────┬──────────┬──────────┬──────────────┐
            # │ event_id ┆ first_pulse_i ┆ last_pulse_i ┆ azimuth  ┆ zenith   ┆ __index_leve │
            # │ ---      ┆ ndex          ┆ ndex         ┆ ---      ┆ ---      ┆ l_0__        │
            # │ i64      ┆ ---           ┆ ---          ┆ f64      ┆ f64      ┆ ---          │
            # │          ┆ i64           ┆ i64          ┆          ┆          ┆ i64          │
            # ╞══════════╪═══════════════╪══════════════╪══════════╪══════════╪══════════════╡
            # │ 24       ┆ 0             ┆ 60           ┆ 5.029555 ┆ 2.087498 ┆ 0            │
            # │ 41       ┆ 61            ┆ 111          ┆ 0.417742 ┆ 1.549686 ┆ 1            │
            # │ 59       ┆ 112           ┆ 147          ┆ 1.160466 ┆ 2.401942 ┆ 2            │
            # └──────────┴───────────────┴──────────────┴──────────┴──────────┴──────────────┘            
            
            # df.columns => ['event_id', 'first_pulse_index', 'last_pulse_index', 'azimuth', 'zenith', '__index_level_0__']
            
            
            df = df.filter(
                pl.col("event_id").is_in(self.cache[fname]["event_id"])
            ).sort("event_id")
            
            self.meta[fname] = df
            if len(self.meta) > self.cache_size:
                del self.meta[list(self.meta.keys())[0]]

    def __getitem__(self, idx0):
        # idx0 => 0
        
        # self.chunk_cumsum[:5], self.chunk_cumsum[-1:] => [ 200000  400000  600000  800000 1000000], [130953924]
        fidx = bisect_right(self.chunk_cumsum, idx0)
        # fidx => 0
        
        # self.files[:5] => ['batch_1.parquet', 'batch_10.parquet', 'batch_100.parquet', 'batch_101.parquet', 'batch_102.parquet']
        
        fname = self.files[fidx]        

        # idx0, fidx => 200000, 1                               (when fidx=1)
        # self.chunk_cumsum[fidx - 1] => 200000                 (when fidx=1) 
           
        
        idx = int(idx0 - self.chunk_cumsum[fidx - 1]) if fidx > 0 else idx0

        self.load_data(fname)
        df = self.cache[fname][idx]
        
        # df =>                (idx = 0)
        #  ──────────┬───────┬───────────────┬───────────────┬───────────┬───────────────┐
        # │ event_id ┆ count ┆ sensor_id     ┆ time          ┆ charge    ┆ auxiliary     │
        # │ ---      ┆ ---   ┆ ---           ┆ ---           ┆ ---       ┆ ---           │
        # │ i64      ┆ u32   ┆ list[i16]     ┆ list[i64]     ┆ list[f64] ┆ list[bool]    │
        # ╞══════════╪═══════╪═══════════════╪═══════════════╪═══════════╪═══════════════╡
        # │ 24       ┆ 61    ┆ [3918, 4157,  ┆ [5928, 6115,  ┆ [1.325,   ┆ [true, true,  │
        # │          ┆       ┆ … 104]        ┆ … 19031]      ┆ 1.175, …  ┆ … true]       │
        # │          ┆       ┆               ┆               ┆ 0.875]    ┆               │
        # └──────────┴───────┴───────────────┴───────────────┴───────────┴───────────────┘        

        
        sensor_id = df["sensor_id"][0].to_numpy()#.item().to_numpy()        
                
        if self.mask_only:
            mask = torch.ones(min(len(sensor_id), self.L), dtype=torch.long)
            # mask.shape => torch.Size([61])
            return {"mask": mask}, {}
        
        time = df["time"][0].to_numpy()#.item().to_numpy()
        charge = df["charge"][0].to_numpy()#.item().to_numpy()
        auxiliary = df["auxiliary"][0].to_numpy()#.item().to_numpy()
        event_idx = df["event_id"].item()
        
        # time[:5] => [5928 6115 6492 6665 8054]
        
        # In the real detector data, the 0-time-reference is not available, and the time might be considered with the reference to the first detected event only. For models trained using 0-time-reference, such experimental data should be shifted by the average first detection time. However, in this case, the performance may drop by 150+ bps, comparable to the results of the model trained without the 0-time leak.
        # time = time - time.min() + 6000.0
        time = (time - 1e4) / 3e4
        # time[:5] => [-0.13573333 -0.1295     -0.11693333 -0.11116667 -0.06486667]        
        
        charge = np.log10(charge) / 3.0

        L = len(sensor_id)  
        L0 = L
        if L < self.L:
            sensor_id = np.pad(sensor_id, (0, max(0, self.L - L)))
            time = np.pad(time, (0, max(0, self.L - L)))
            charge = np.pad(charge, (0, max(0, self.L - L)))
            auxiliary = np.pad(auxiliary, (0, max(0, self.L - L)))
        else:
            # idx, L, self.L => 6, 316, 192
            
            # .randperm(n) => Returns a random permutation of integers from 0 to n - 1.
            ids = torch.randperm(L).numpy()
            # ids[:5] => [ 58 154  81 111 158]
            
            auxiliary_n = np.where(~auxiliary)[0] # return indices where ~auxiliary is True.
            auxiliary_p = np.where(auxiliary)[0]
                        
            # len(ids), len(auxiliary_n), len(auxiliary_p) => 316, 231, 85
            # min(self.L, len(auxiliary_n)) => 192
            ids_n = ids[auxiliary_n][: min(self.L, len(auxiliary_n))]
            # min(self.L - len(ids_n), len(auxiliary_p)) => 0

            ids_p = ids[auxiliary_p][: min(self.L - len(ids_n), len(auxiliary_p))]
            
            ids = np.concatenate([ids_n, ids_p])
            ids.sort()
            
            sensor_id = sensor_id[ids]
            time = time[ids]
            charge = charge[ids]
            auxiliary = auxiliary[ids]
            L = len(ids)
            # L => 192

        attn_mask = torch.zeros(self.L, dtype=torch.bool)        
        attn_mask[:L] = True
        
        sensor_id = torch.from_numpy(sensor_id).long()
        pos = self.geometry[sensor_id]
        # pos.shape => torch.Size([192, 3])       
        pos[L:] = 0
        
        qe = self.qe[sensor_id]
        qe[L:] = 0
        
        # type(self.ice_properties[0]) =>  <class 'scipy.interpolate._interpolate.interp1d'>
        
        ice_properties = np.stack(
            [self.ice_properties[0](pos[:L, 2]), self.ice_properties[1](pos[:L, 2])], -1
        )
        # ice_properties.shape => (61, 2)
        ice_properties = np.pad(ice_properties, ((0, max(0, self.L - L)), (0, 0)))
        # ice_properties.shape => (192, 2)
        ice_properties = torch.from_numpy(ice_properties).float()

        time = torch.from_numpy(time).float()
        # if self.mode == 'train': time += 0.1*torch.randn(1)

        if self.mode != "test":
            meta = self.meta[fname][idx]            
            # meta.head()
            # ┌──────────┬───────────────┬──────────────┬──────────┬──────────┬──────────────┐
            # │ event_id ┆ first_pulse_i ┆ last_pulse_i ┆ azimuth  ┆ zenith   ┆ __index_leve │
            # │ ---      ┆ ndex          ┆ ndex         ┆ ---      ┆ ---      ┆ l_0__        │
            # │ i64      ┆ ---           ┆ ---          ┆ f64      ┆ f64      ┆ ---          │
            # │          ┆ i64           ┆ i64          ┆          ┆          ┆ i64          │
            # ╞══════════╪═══════════════╪══════════════╪══════════╪══════════╪══════════════╡
            # │ 24       ┆ 0             ┆ 60           ┆ 5.029555 ┆ 2.087498 ┆ 0            │
            # └──────────┴───────────────┴──────────────┴──────────┴──────────┴──────────────┘
            azimuth = meta["azimuth"].item()
            zenith = meta["zenith"].item()
            target = np.array([azimuth, zenith]).astype(np.float32)
            
            # target.shape => (2,)
            target = torch.from_numpy(target)
        else:
            target = df["event_id"].item()

        return {
            "sensor_id": sensor_id,
            "time": time,
            "charge": torch.from_numpy(charge).float(),
            "pos": pos,
            "mask": attn_mask,
            "idx": event_idx,
            "auxiliary": torch.from_numpy(auxiliary).long(),
            "qe": qe,
            "ice_properties": ice_properties,
            "L0": L0,
        }, {"target": target}


class RandomChunkSampler(torch.utils.data.Sampler[int]): # sampler will yield integers (typically indices).
    data_source: Sized
    replacement: bool # boolean flag indicating whether sampling is done with or without replacement.

    def __init__(
        self,
        data_source: Sized,
        chunks,
        num_samples: Optional[int] = None,
        generator=None,
        **kwargs,
    ) -> None:
        # chunks - a list of chunk sizes
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.chunks = chunks

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:

        n = len(self.data_source)
        cumsum = np.cumsum(self.chunks) # useful for determining the boundaries between the chunks.
        # cumsum[:5], cumsum[-1] => [ 200000  400000  600000  800000 1000000], 130953924
        
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        chunk_list = torch.randperm(len(self.chunks), generator=generator).tolist()
        # len(self.chunks), len(chunk_list), chunk_list[:5] => 655, 655, [44, 36, 132, 599, 167]                
        
        # sample indexes chunk by chunk
        yield_samples = 0
        for i in chunk_list:
            chunk_len = self.chunks[i]
            offset = cumsum[i - 1] if i > 0 else 0
            # offset => 8800000

            samples = (offset + torch.randperm(chunk_len, generator=generator)).tolist()
            # samples[:5] => 200000, [8823127, 8842653, 8819382, 8864056, 8876650]
            
            if len(samples) <= self.num_samples - yield_samples:
                yield_samples += len(samples)
            else:
                samples = samples[: self.num_samples - yield_samples]
                yield_samples = self.num_samples
            
            # len(samples), samples[:5] => 200000, [8823127, 8842653, 8819382, 8864056, 8876650]

            yield from samples

    def __len__(self) -> int:
        return self.num_samples


class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0
        # buckets => [[], [], [], [], [], [] ...]      
        
        # self.sampler =>
        # <src.dataset.RandomChunkSampler object at 0x7f9dcdc28850>                          (during train)
        # <torch.utils.data.sampler.SequentialSampler object at 0x7f9d9c09bdf0>              (during test)
            
        for idx in self.sampler:
            
            # self.sampler.data_source => <src.dataset.IceCubeCache object at 0x7fb642a289d0>
            # self.sampler.data_source.mask_only => True
            s = self.sampler.data_source[idx]           
            
            if isinstance(s, tuple):
                # s[0]["mask"] => tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...])
                L = s[0]["mask"].sum()
                # L => 61
            else:
                L = s["mask"].sum()
            
            L = max(1, L // 16) #  61//16 => 3
            
            if len(buckets[L]) == 0:
                buckets[L] = []
                
            buckets[L].append(idx)
            # buckets => [[], [], [], [8823127], [], ...]
                        
            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                # batch => [8823127, 8864056]                (self.batch_size = 2)          
                yield batch
                yielded += 1
                buckets[L] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1            
            yield batch

        # assert len(self) == yielded,\
        #  "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)


def dict_to(x, device="cuda"):
    return {k: x[k].to(device) for k in x}


def to_device(x, device="cuda"):
    return tuple(dict_to(e, device) for e in x)


class DeviceDataLoader:
    def __init__(self, dataloader, device="cuda"):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)
