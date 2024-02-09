# Graph equivariant diffusion models: the crystal material generation testcase

This project introduces a novel framework for generating new crystal materials using equivariant diffusion models based on a graph-like representation of data. This offers a powerful approach to surpass previous implementations, as it considers interactions and similarities between close points (which are more likely to interact). As well, aligned with previous implementations, ours also allows maximizing arbitrary targets such as conductivity, absorption, and more, or looking for materials with specific properties (such as diffraction patterns or n-order transitions). The generated crystals demonstrate remarkable thermodynamic stability (convex-hull approach), compared against the Materials Project database. 

The core technology behind this framework is based on deep convolutional layers and graph-like representation of data, where the diffusion process is achieved through the use of Markov chains. The denoising aspect of the model is implemented using convolutional graph neural networks, ensuring high-quality results, with which the noise of graphs is predicted and extracted, allowing the generation of an arbitrary number of novel, independent materials.

This technology is suitable for different applications: from discovering improved ionic conductors beyond current databases to generating molecules for efficient water splitting. Moreover, the model itself can be applied to a variety of problems (concretely, any problem which can be formulated in terms of graphs), such as proposing enhanced distributions in social networks or traffic. Then, although applied to crystal material generation, this repository is divided into two independent functionalities:

- Database generation: Generation of graph database from a set of crystal material structure files. Here we implemented the cut-off and Voronoi tesselation strategies.
- New materials generation: Generation and interpolation of novel graphs given a database of graphs (which can in fact be applied to any other problem whose data is graph-like).

A quick discussion of all these topics can be found in our [paper](https://www.overleaf.com/read/cjxhknmhpfpg#d4cb5f).

## Features

- Generation of new crystal materials beyond the limitations of existing databases
- Optimization of arbitrary targets such as conductivity, light absorption, and more
- Evaluation of thermodynamic stability using the convex-hull approach
- Graph-based representation of data for efficient modeling
- Denoising process utilizing convolutional graph neural networks
- High stability and reliability of generated crystals

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

 - Cibrán López Álvarez
 - Jacobo Osorio Ríos

## Contact, questions and contributing

If you have questions, please don't hesitate to reach out at: cibran.lopez@upc.edu
