# PathletRL
A deep learning model based on reinforcement learning to construct pathlet dictionaries.

## Requirements
The code was run in Ubuntu machines, but it can also be run on Macs (based on the packages listed in ```requirements.txt```; for example ```tensorflow-macos``` is specifically for the Mac). In general you can install most of the packages using the following command. 

```
pip install -r requirements.txt
```

It can be noted that this is mostly built with ```Tensorflow``` and specifically ```tf-agents``` specifically for RL models. 

## Datasets
In our experiments, we used the ```Toronto``` dataset, which we provide the link on our shared Google Drive (we could not share it through Github due to size). Moreover, the ```Rome``` dataset is publicly available and is being hosted on ```Crawdad```: https://crawdad.org/. In case you want to use your own datasets, here are the specifics for curating them. Grab your dataset and preprocess it as follows:

1. Have a ```.json``` file called ```traj_edge_dict.json```. This dictionary should have the following keys and values formatting:

| Keys | Values |
| --- | --- |
| ```1``` | ```[[273, 272], [272, 347], [347, 321], [321, 379], [379, 385], [385, 320]]``` |
| ```6``` | ```[[1362, 1558]]``` |
| ```11``` | ```[[115, 349], [349, 348], [348, 605]]``` |
| ... | ... |

Here, each trajectory (keys are the trajectory IDs) is mapped to a list of edges (road segments) in the road network (i.e., ```[u, v]``` is an edge in the road network where ```u``` and ```v``` are node/road intersection IDs). It is important to note that all trajectories have to be continuous (no gaps), which means that the "end node" of the previous edge is the same as the "start node" of the following edge. Ensure this, otherwise you could run into errors later in the long run.

2. Now also have ```traj_coords_dict.json```, which is literally the same file as above but we replace every road intersection ID with a length-```2``` list the consists of the longitude and latitude coordinates of the node in the road network.

3. Now curate ```traj_pathlets_dict.json``` (i.e., the pathlet-based representation of trajectories) by mapping each edge in ```traj_edge_dict.json``` to a pathlet ID (you can curate a ```pathlet_dict.json``` later that does this. See Step 4). Each ID will be a string. Here is an example.

| Keys | Values |
| --- | --- |
| ```1``` | ```['12267', '12398', '12478', '12479', '12484', '12915']``` |
| ```6``` | ```['60845']``` |
| ```11``` | ```['62672', '12400', '46585']``` |
| ... | ... |

This means that edge ```[273, 272]``` has pathlet ID ```'12267'```. It also means that some edge ```[272, 273]``` maps to the same pathlet ID (under this circumstance where the road network is undirected). Also, you can think of the pathlets as "edges" initially since all pathlets are length-```1```, which are simply just the edges of the road network.

4. Now curate ```pathlet_dict.json```, which you can do before Step 3.

| Keys | Values |
| --- | --- |
| ```'321'``` | ```[0, 1]``` |
| ```'48612'``` | ```[1364, 0]``` |
| ```'49547'``` | ```[687, 1]``` |
| ... | ... |

## Running the Model

1. Place all the datasets to be used under a directory called ```/data/```

2. Run the PathletRL using the following command:

```
python main_rl.py
```

## Citation

If you like our work or if you plan to use it, please cite our work with the following Bibtex format:

```
@INPROCEEDINGS{alix2023pathletrl,
  author={Alix, Gian and Papagelis, Manos},
  booktitle={2023 Proceedings of the 31st ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems}, 
  title={PathletRL: Trajectory Pathlet Dictionary Construction using Reinforcement Learning}, 
  year={2023},
  volume={},
  number={},
  pages={},
  doi={}
}

```

Or you can also use this citation:

> Gian Alix and Manos Papagelis. 2023. PathletRL: Trajectory Pathlet Dictionary Construction using Reinforcement Learning. In Proceedings of Proceedings of the 31st ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (SIGSPATIAL ’23). ACM, New York, NY, USA, 12 pages. https://doi.org/XXXXXXX.XXXXXXX

#### Contact

Please contact me gcalix@eecs.yorku.ca for any bugs/issues/questions you may have found on the code.
