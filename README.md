# Organization

The various Python scripts assume a shared organizational structure such that the output from one script can easily be used as input to another. This is true for both preprocessed data as well as experiments which make use of the datasets.

##### Data Layout

The DeepSDF code allows for pre-processing of meshes from multiple datasets and stores them in a unified data source. It also allows for separation of meshes according to class at the dataset level. The structure is as follows:

```
<data_source_name>
|-- .data_sources.json 
|-- <dataset_name>
    |-- <class_name>
        |-- <instance_name>.npz
```

Subsets of the unified data source can be reference using split files, which are stored in a simple JSON format. For examples, see `examples/splits/`. 

The file `data_sources.json` stores a mapping from named datasets to paths indicating where the data came from. This file is referenced again during evaluation to compare against ground truth meshes (see below), so if this data is moved this file will need to be updated accordingly.

##### Experiment Layout

Each DeepSDF experiment is organized in an "experiment directory", which collects all of the data relevant to a particular experiment. The structure is as follows:

```
<experiment_name>
|-- specs.json
|-- Logs.pth
|-- LatentCodes
|   |-- <Epoch>.pth
|-- ModelParameters
|   |-- <Epoch>.pth
|-- OptimizerParameters
|   |-- <Epoch>.pth
|-- Reconstructions
|   |-- <Epoch>
|       |-- Codes
|       |   |-- <MeshId>.pth
|       |-- Meshes
|           |-- <MeshId>.pth
|-- Evaluations
    |-- Chamfer
    |   |-- <Epoch>.json
    |-- EarthMoversDistance
    |   |-- <Epoch>.json
    |-- Reconstruction
        |-- <Epoch>.json
```

The only file that is required to begin an experiment is 'specs.json', which sets the parameters, network architecture, and data to be used for the experiment.

# How to Use DeepSDF

### Pre-processing the Data

TODO

### Training a Model

TODO

##### Visualizing Progress

To visualize the progress of a model during training, run:

```
python plot_log.py -e <experiment_directory>
```

##### Continuing from a Saved Optimization State

TODO

### Reconstructing Meshes

To use a trained model to reconstruct explicit mesh representations of shapes from the test set, run:

```
python reconstruct.py -e <experiment_directory>
```

This will use the latest model parameters to reconstruct all the meshes in the split. To specify a particular checkpoint to use for reconstruction, use the ```--checkpoint``` flag followed by the epoch number.

### Shape Completion

TODO

### Evaluating Reconstructions

TODO

# Examples

```
# create a home for the data
mkdir <...>/DeepSdf

# pre-process a selection of ShapeNetV2 classes
python append_data.py -s <...>/ShapeNetCore.v2/ -d <...>/DeepSdf/ --name ShapeNetV2 --skip --classes 02691156 03001627 03636649 04256520 04379243

# train the model
python train_deep_sdf.py -e <...>/examples/chair

# reconstruct meshes from the test split
python reconstruct.py -e <...>/examples/chair --split <...>/examples/splits/sv2_chairs_test.json --skip

# evaluate the reconstructions

```

TODO
