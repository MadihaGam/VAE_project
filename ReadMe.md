# Variational Autoencoder (VAE) Project

This project explores the implementation and comparison of Variational Autoencoders (VAEs) using three different latent space distributions: **Gaussian**, **Dirichlet**, and **Continuous Categorical**. The results showcase the models' capabilities in learning structured latent representations and generating high-quality image reconstructions.



## Dependencies

To run the code and notebook, the following Python libraries are required:

- `numpy`
- `torch`
- `sklearn`
- `matplotlib`
- `seaborn`
- `umap-learn`



## Usage

Train Models:

Use the provided Python scripts to train the models:

```
python GVAE.py  # Train Gaussian VAE
python DirVAE.py # Train Dirichlet VAE
python ccVAE.py # Train CC VAE
```

