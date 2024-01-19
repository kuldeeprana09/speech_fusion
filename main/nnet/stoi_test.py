import torch
from torch import nn
from torch_stoi import NegSTOILoss

sample_rate = 16000
loss_func = NegSTOILoss(sample_rate=sample_rate)
# Your nnet and optimizer definition here
nnet = nn.Module()

noisy_speech = torch.randn(2, 16000)
clean_speech = torch.randn(2, 16000)
# Estimate clean speech
est_speech = nnet(noisy_speech)
# Compute loss and backward (then step etc...)
loss_batch = loss_func(est_speech, clean_speech)
loss_batch.mean().backward()