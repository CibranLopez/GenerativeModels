Project: Generative library

Cibran: Check other approaches for edge attribute prediction.

Cibran: When introducing graph-level embeddings, explore convoluting node and graph-level separately and then convolving those latent representations together. Explore how these approaches have been implemented before in AI.

Cibran: Training is being performed over noise prediction (although I think this is better than graph itself, it is worth checking it).

Cibran: Check conditional generation approaches (ILVR has been discussed in the draft).

Jacobo: Check schedule of weights in model backpropagations.

Cibran: Losses on nodes are being averaged, although it is interesting to save them separately.

Cibran: We first add t-step information allowing transfer learning from unconditional to conditional learning.