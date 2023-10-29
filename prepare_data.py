import polars as pl
import pickle
import os
from tqdm import tqdm
import sys
import argparse
import json


def read_config_file(file_path):
    with open(file_path, "r") as f:
        config_data = json.load(f)    
    return config_data

def run(config):
    Nevents = {}
    for fname in tqdm(sorted(os.listdir(os.path.join(config["PATH"], "train")))):
        # fname => batch_1.parquet
        
        path = os.path.join(config["PATH"], "train", fname)
        df = pl.read_parquet(path)
        # df.head(3)
        #  sensor_id ┆ time ┆ charge ┆ auxiliary ┆ event_id │
        # │ ---       ┆ ---  ┆ ---    ┆ ---       ┆ ---      │
        # │ i16       ┆ i64  ┆ f64    ┆ bool      ┆ i64      │
        # ╞═══════════╪══════╪════════╪═══════════╪══════════╡
        # │ 3918      ┆ 5928 ┆ 1.325  ┆ true      ┆ 24       │
        # │ 4157      ┆ 6115 ┆ 1.175  ┆ true      ┆ 24       │
        # │ 3520      ┆ 6492 ┆ 0.925  ┆ true      ┆ 24          
        
        # count of elements/rows in each event_id.
        df = df.groupby("event_id").agg([pl.count()])
        # df.head() => 
        #  event_id ┆ count │
        # │ ---      ┆ ---   │
        # │ i64      ┆ u32   │
        # ╞══════════╪═══════╡
        # │ 3224960  ┆ 88    │
        # │ 619528   ┆ 82    │
        # │ 2041056  ┆ 101   │
        # │ 2709032  ┆ 44    │
        # │ 825096   ┆ 41    │    

        Nevents[fname] = {}
        Nevents[fname]["total"] = len(df)
        Nevents[fname]["short"] = len(df.filter(pl.col("count") < 64))
        Nevents[fname]["medium"] = len( df.filter((pl.col("count") >= 64) & (pl.col("count") < 192)) )
        Nevents[fname]["long"] = len(df.filter(pl.col("count") >= 192))

    with open(os.path.join(config["PATH"], "Nevents.pickle"), "wb") as f:
        pickle.dump(Nevents, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(
        description="Create a model from a JSON config file."
    )
    parser.add_argument("config_file", type=str, help="Path to the JSON config file.")
    args = parser.parse_args()
    config_file_path = args.config_file
    config_data = read_config_file(config_file_path)
    run(config_data)
    
if __name__ == "__main__":
    main()
