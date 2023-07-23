# Equivariant Diffusion Models for Crystal Material Generation

This project introduces a novel framework for generating new crystal materials using equivariant diffusion models. The framework offers a powerful approach to surpass the limitations of existing databases by maximizing arbitrary targets such as conductivity, absorption, and more. The generated crystals demonstrate remarkable thermodynamic stability, compared against the Materials Project database, utilizing the convex-hull approach. 

The core technology behind this framework is based on a graph-like representation of data, where the diffusion process is achieved through the use of Markov chains. The denoising aspect of the model is implemented using convolutional graph neural networks, ensuring high-quality results, with which the noise of graphs is predicted.

This technology is suitable for different applications: from discovering improved ionic conductors beyond current databases to generating molecules for efficient water splitting.

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

An user-friendly jupyter notebook has been developed, which can be run locally with pytorch dependencies. It generates a graph-like database (from the Materials Project database) and trains the generative model to better reproduce those materials and enhance a desired target.

## Authors

This project is being developed by:

 - Cibrán López Álvarez

## Contact, questions and contributing

If you have questions, please don't hesitate to reach out at: cibran.lopez@upc.edu
