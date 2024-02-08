# Graph equivariant diffusion models: the crystal material generation testcase

This project introduces a novel framework for generating new crystal materials using equivariant diffusion models through a graph-like representation of data. This offers a powerful approach to surpass the limitations of existing databases by maximizing arbitrary targets such as conductivity, absorption, and more, thanks to enhancing local features of particles. The generated crystals demonstrate remarkable thermodynamic stability, compared against the Materials Project database, utilizing the convex-hull approach. 

The core technology behind this framework is based on a graph-like representation of data, where the diffusion process is achieved through the use of Markov chains. The denoising aspect of the model is implemented using convolutional graph neural networks, ensuring high-quality results, with which the noise of graphs is predicted.

This technology is suitable for different applications: from discovering improved ionic conductors beyond current databases to generating molecules for efficient water splitting.

Although applied to crystal material generation, this repository is divided into two independent functionalities:

- Database generation: Generation of graph database from a set of crystal material structure files. Here we implemented the cut-off and Voronoi tesselation strategies.
- New materials generation: Generation and interpolation of novel graphs given a database of graphs (which can in fact be applied to any other problem whose data is graph-like).

As well, a quick discussion might be found in our [paper](https://www.overleaf.com/read/cjxhknmhpfpg#d4cb5f).

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

A set of user-friendly jupyter notebook have been developed, which can be run locally with pytorch and pymatgen dependencies. It generates a graph-like database (from the Materials Project database) and trains the generative model to best reproduce those materials and enhance some desired target.

## Authors

This project is being developed by:

 - Cibrán López Álvarez
 - Jacobo Osorio Ríos

## Contact, questions and contributing

If you have questions, please don't hesitate to reach out at: cibran.lopez@upc.edu
