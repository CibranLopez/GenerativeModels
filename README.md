# Graph equivariant diffusion models: the crystal material generation testcase

This project introduces a novel framework for generating new crystal materials using equivariant diffusion models based on a graph-like representation of data. This offers a powerful approach to surpass previous implementations, as it considers interactions and similarities between close points (which are more likely to interact). As well, aligned with previous implementations, ours also allows maximizing arbitrary targets such as conductivity, absorption, and more, or looking for materials with specific properties (such as diffraction patterns or n-order transitions). The generated crystals demonstrate remarkable thermodynamic stability (convex-hull approach), compared against the Materials Project database. 

The core technology behind this framework is based on deep convolutional layers and graph-like representation of data, where the diffusion process is achieved through the use of Markov chains. The denoising aspect of the model is implemented using convolutional graph neural networks, ensuring high-quality results, with which the noise of graphs is predicted and extracted, allowing the generation of an arbitrary number of novel, independent materials.

This technology is suitable for different applications: from discovering improved ionic conductors beyond current databases to generating molecules for efficient water splitting. Moreover, the model itself can be applied to a variety of problems (concretely, any problem which can be formulated in terms of graphs), such as proposing enhanced distributions in social networks or traffic. Then, although applied to crystal material generation, this repository is divided into two independent functionalities:

The main feature is the generation and interpolation of novel graphs given a reference database of them. However, this architecture can be directly applied to any other problem whose data is graph-like.

A quick discussion of all these topics can be found in our [paper](https://www.overleaf.com/read/cjxhknmhpfpg#d4cb5f).

## Features

Within this repository you will find the following scripts:

- Graph database generation:
  - [`database-generation.ipynb`](database-generation.ipynb): generates a graph database from a set of crystals or molecules.
  - [`molecule-to-POSCAR.ipynb`](molecule-to-POSCAR.ipynb): script for converting a molecule database into a POSCAR-like one.

- Property prediction:
  - [`GNN.ipynb`](GNN.ipynb): prediction of band-gaps in molecules and crystals via graph neural networks.

- Material generation:
  - [`basic-example.ipynb`](basic-example.ipynb): simplified script for checking diffusion and its parameters.
  - [`model-training.ipynb`](model-training.ipynb): main code for training the diffusion models.
  - [`graph-interpolation.ipynb`](graph-interpolation.ipynb): interpolates pairs of graphs based on some target property.
  - [`graph-prediction.ipynb`](graph-prediction.ipynb): generates graphs based on some target property.
  - [`graph-to-POSCAR.ipynb`](graph-to-POSCAR.ipynb): conversion of a graph structure into a POSCAR representing a molecule or a crystal.

with **libraries** containing all main functionalities and **tests** some test functions. 

Please be aware that the code is under active development, bug reports are welcomed in the GitHub issues!

## Installation

To download the repository and install the dependencies:

```bash
git clone https://github.com/CibranLopez/GenerativeModels.git
cd GenerativeModels
pip3 install -r requirements.txt
```

## Execution

A set of user-friendly jupyter notebook have been developed, which can be run locally with pytorch and pymatgen dependencies. It generates a graph-like database (from the Materials Project database or any other source) and trains the generative model to best reproduce those materials (and enhance some desired target, if desired).

## Authors

This project is being developed by:

- **Cibrán López Álvarez** - Lead Developer and Researcher
- **Jacobo Osorio Ríos** - Developer

## Contact, questions and contributing

For any questions, issues, or contributions, feel free to contact:

- Cibrán López Álvarez: [cibran.lopez@upc.edu](mailto:cibran.lopez@upc.edu)

Feel free to open issues or submit pull requests for bug fixes, improvements, or feature suggestions.