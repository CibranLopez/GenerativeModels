Project: Generative library

Cibran: Convert graph into POSCAR (not working properly right now).

Cibran: Check other approaches for edge attribute prediction.

Cibran: When introducing graph-level embeddings, explore convoluting node and graph-level separately and then convolving those latent representations together. Explore how these approaches have been implemented before in AI.

Cibran: GPU enabled, check that everything is being predicted and assigned correctly.

Cibran: Training is being performed over noise prediction (although I think this is better than grah itself, it is worth checking it).

Cibran: Given a batch of nodes, i want to denoise a step in batch and then do that over t-steps. These functions have to be improved.