# VAE
Exploring various questions about VAEs

--- 

- [ ] batch norm or no batch norm in encoder-decoder
- [ ] beta-VAE effect
- [ ] taking expectation of nll over image / using normalised-beta
- [ ] image statistics based regularisation term

---

### Base Model

The base model uses the encoder-decoder architecture from the [VQ-VAE paper]((https://arxiv.org/pdf/1711.00937) and the ELBO from the original VAE paper. 

Performance is evaluated on a reserved test set in each case. No hyper-parameter tuning is performed. 

| Data |  NLL | bits/dim | Example Images (top: input; bottom: output) |
| :---:   |  :---: | :---: | :---: |
| MNIST |   | |  |
| CIFAR10 |   | |  |
| RGZ |   | |  |

---
