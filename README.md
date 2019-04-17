# Organization

The various Python scripts assume a shared organizational structure such that the output from one script can easily be used as input to another. This is true for both preprocessed data as well as experiments which make use of the datasets.

##### Data Layout

TODO

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
    |-- <Epoch>
        |-- Codes
        |   |-- <MeshId>.pth
        |-- Meshes
            |-- <MeshId>.pth
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
python plot_log.py -e <experiment_directory>
```

This will use the latest model parameters

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

...

```

TODO
