#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import sys
sys.dont_write_bytecode = True

from behavior_benchmarks.applications.VAME.vame.model.create_training import create_trainset
from behavior_benchmarks.applications.VAME.vame.model.dataloader import SEQUENCE_DATASET
from behavior_benchmarks.applications.VAME.vame.model.rnn_vae import train_model
from behavior_benchmarks.applications.VAME.vame.model.evaluate import evaluate_model

