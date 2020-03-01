# Misbehaviour Prediction for Autonomous Driving Systems

This repository contains the code attached to the paper "Misbehaviour Prediction for Autonomous Driving Systems" by A. Stocco, M. Weiss, M. Calzana, P. Tonella, to be published in the proceedings of the 42nd International Conference in Software Engineering (ICSE 2020).
See LICENSE.md for more information if you want to use the code.

## Dependencies

**Software setup:**
We adopted the [PyCharm](https://www.jetbrains.com/pycharm/) Professional 2019.3.1, a Python IDE by JetBrains.

If you have [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) installed on your machine, you can create and install all dependencies on a dedicated virtual environment, by running the following command:

```python
# Use TensorFlow with GPU
conda env create -f code-sdc/conda-env.yml
```

Alternatively you can manually install the required libraries (see the contents of the conda-env.yml files) using ```pip```.



**Hardware setup:** Training the DNN models (self-driving cars and anomaly detectors) on our datasets is computationally expensive. Therefore we recommend using a machine powered by a GPU. In our setting, we ran our experiments on a machine equipped with a i9 processor, 32 GB of memory, and an Nvidia GPU GeForce RTX 2080 Ti with 11GB of dedicated memory. 

## Repository Structure

The repository is structured as follows:

<pre/>
- code-predictors
  | This folder contains all the code to run the training and evaluation of the misbehavior predictors.
- code-sdc
  | This folder contains all the code to train the self-driving car models and record them when they are driving.
- evaluation-results
  | This folder contains sqlite databases containing the results of our evaluation
</pre>

### Other Artefacts

We made the following artifacts available as a torrent file [here](https://academictorrents.com/details/221c3c71ac0b09b1bb31698534d50168dc394cc7). The files have a combined size of 24.9 GB, and the torrent contains:

- Trained SDC Models
- Trained Failure Predictor Models
- Training Dataset
- Evaluation Dataset

## Usage

### Reproduce the results

```python
python code-predictors/evaluation_runner.py
```

*Note:* You do not need the simulator to reproduce the results.

### Run the pretrained SDC models

Start up our Udacity self-driving simulator, choose a scene, and press the Autonomous Mode button. Then, run the model as follows:

```python
python code-sdc/drive.py <sdc-model>.h5
```

### Train a SDC model

You'll need the data folder which contains the training images.

```python
python code-sdc/train_self_driving_car.py
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.


### Train an anomaly detector


You'll need the data folder which contains the training images.

```python
python code-predictors/training_runner.py
```

## Improved simulator

Our improved Udacity simulator is available as binary file for Windows and macOS platforms [here](https://drive.google.com/drive/folders/1i4naoN9Wermz5LSW_RNeXdN2zhDMKIDJ). 

## Demo video of the simulator

[![Watch the video](https://youtu.be/r4oiX6UBJPI/maxresdefault.jpg)](https://youtu.be/r4oiX6UBJPI)

## License 
See the [LICENSE.md](https://github.com/testingautomated-usi/selforacle/LICENSE.md) file.

## Contacts

For any questions, feel free to contact Andrea Stocco ([andrea.stocco@usi.ch](mailto:andrea.stocco@usi.ch)) or Michael Weiss ([michael.weiss@usi.ch](mailto:michael.weiss@usi.ch)).