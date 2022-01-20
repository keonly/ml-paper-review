"""DRO and DORO Training Algorithms

Reference:
[1] Hashimoto et al., Fairness Without Demographics in Repeated
      Loss Minimization, ICML 2018.
"""

import math
import scipy.optimize as sopt

import torch
import torch.nn
from torch import optim
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F


def erm(model: Module, loader: DataLoader, optimizer: optim.Optimizer,
        criterion, device: str):
  """Empirical Risk Minimization (ERM)"""

  model.train()
  for _, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def cvar(model: Module, loader: DataLoader, optimizer: optim.Optimizer,
         criterion, device: str, alpha: float):
  """Original CVaR"""
    raise NotImplementedError


def cvar_doro(model: Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion, device: str, alpha: float, eps: float):
  """CVaR DORO"""
    raise NotImplementedError



def chisq(model: Module, loader: DataLoader, optimizer: optim.Optimizer,
          criterion, device: str, alpha: float):
  """Chi^2-DRO"""

  model.train()
  max_l = 10.
  C = math.sqrt(1 + (1 / alpha - 1) ** 2)
  for _, (inputs, targets) in enumerate(loader):
    raise NotImplementedError
    


def chisq_doro(model: Module, loader: DataLoader, optimizer: optim.Optimizer,
                 criterion, device: str, alpha: float, eps: float):
  """Chi^2-DORO"""

  model.train()
  max_l = 10.
  C = math.sqrt(1 + (1 / alpha - 1) ** 2)
  for _, (inputs, targets) in enumerate(loader):
    raise NotImplementedError


def train(alg: str, model: Module, loader: DataLoader,
          optimizer: optim.Optimizer, criterion, device: str,
          alpha: float, eps: float):
  """Train one epoch using the specified algorithm"""

  if alg == 'erm':
    erm(model, loader, optimizer, criterion, device)
  elif alg == 'cvar':
    cvar(model, loader, optimizer, criterion, device, alpha)
  elif alg == 'cvar_doro':
    cvar_doro(model, loader, optimizer, criterion, device, alpha, eps)
  elif alg == 'chisq':
    chisq(model, loader, optimizer, criterion, device, alpha)
  elif alg == 'chisq_doro':
    chisq_doro(model, loader, optimizer, criterion, device, alpha, eps)
  else:
    raise NotImplementedError