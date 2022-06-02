Graph-ICF:  Item-based Collaborative Filtering based on Graph Neural Network. Knowledge-based systerms


## Enviroment Requirement

`pip install -r requirements.txt`

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command/model中包含三个模型，lgn（LightGCN),icf(Graph-ICF),graph-icf(尝试将icf与lgn结合起来的版本，未完成). 因此icf_ratio在模型lgn与icf下无效。

` cd code && CUDA_VISIBLE_DEVICES=3 python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="ml-1m" --topks="[10, 20]" --recdim=64 --icf_ratio=0.5 --model=icf --comment=''`


#### Update

1. Change the print format of each epoch
2. Add Cpp Extension in  `code/sources/`  for negative sampling. To use the extension, please install `pybind11` and `cppimport` under your environment
