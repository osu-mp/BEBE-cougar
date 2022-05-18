import yaml
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchmetrics
from behavior_benchmarks.models.wicc_utils import BEHAVIOR_DATASET
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from behavior_benchmarks.models.model_superclass import BehaviorModel
from behavior_benchmarks.applications.S4.S4 import S4
from sklearn.mixture import GaussianMixture

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class wicc(BehaviorModel):
  def __init__(self, config):
    super(wicc, self).__init__(config)
    print(f"Using {device} device")
    
    ##
    self.downsizing_factor = self.model_config['downsizing_factor']
    self.lr = self.model_config['lr']
    self.weight_decay = self.model_config['weight_decay']
    self.n_epochs = self.model_config['n_epochs']
    self.hidden_size = self.model_config['hidden_size']
    self.n_s4_blocks = self.model_config['num_layers'] ## Total layers is num_layers * 3 tiers
    self.temporal_window_samples = self.model_config['temporal_window_samples']
    self.batch_size = self.model_config['batch_size']
    self.dropout = self.model_config['dropout']
    self.blur_scale = self.model_config['blur_scale']
    self.jitter_scale = self.model_config['jitter_scale']
    self.state_size = self.model_config['state_size']
    self.downsample_rate = self.model_config['downsample_rate']
    self.n_clusters = self.config['num_clusters']
    self.context_window_samples = self.model_config['context_window_samples']
    self.n_pseudolabels = self.n_clusters // 2
    self.max_iter_gmm = self.model_config['max_iter_gmm']
    self.tau_init = self.model_config['tau_init']
    self.tau_decay_rate = self.model_config['tau_decay_rate']
    ##
    
    # cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    # self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_features = len(self.cols_included)
    
    self.encoder =  Encoder(self.n_features,
                            self.n_clusters,
                            hidden_size = self.hidden_size,
                            state_size = self.state_size,
                            n_s4_blocks = self.n_s4_blocks,
                            downsample_rate = self.downsample_rate,
                            dropout = self.dropout,
                            blur_scale = self.blur_scale,
                            jitter_scale = self.jitter_scale).to(device)
    
    self.decoder = Decoder(self.n_clusters,
                           self.n_pseudolabels,
                           self.context_window_samples).to(device)
    
    print('Encoder parameters:')
    print(_count_parameters(self.encoder))
  
  def load_pseudolabels(self, filename):
    filepath = os.path.join(self.pseudolabel_dir, filename)
    labels = np.load(filepath).astype(int)
    return labels
  
  def generate_pseudolabels(self):
    ## Generate pseudo-labels
    print("Training GMM to produce pseudo-labels")
    
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
      test_fps = self.config['test_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
      test_fps = self.config['test_data_fp']
    
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in dev_fps]
    dev_data = np.concatenate(dev_data, axis = 0)
    
    gmm = GaussianMixture(n_components = self.n_pseudolabels, verbose = 2, max_iter = self.max_iter_gmm, n_init = 1)
    gmm.fit(dev_data)
    
    self.pseudolabel_dir = os.path.join(self.config['temp_dir'], 'pseudolabels')
    if not os.path.exists(self.pseudolabel_dir):
      os.makedirs(self.pseudolabel_dir)
    
    print("Generating pseudo-labels for dev and test data")
    for fp in tqdm.tqdm(dev_fps + test_fps):
      data = self.load_model_inputs(fp, read_latents = self.read_latents)
      pseudolabels = gmm.predict(data)
      target = os.path.join(self.pseudolabel_dir, fp.split('/')[-1])
      np.save(target, pseudolabels)
    
  def fit(self):
    self.generate_pseudolabels()
    
    ## get data. assume stored in memory for now
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
      test_fps = self.config['test_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
      test_fps = self.config['test_data_fp']
    
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in dev_fps]
    test_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in test_fps]
    
    ## Load up pseudo-labels
    
    dev_pseudolabels = [self.load_pseudolabels(fp) for fp in self.config['dev_file_ids']]
    test_pseudolabels = [self.load_pseudolabels(fp) for fp in self.config['test_file_ids']]
    
    dev_dataset = BEHAVIOR_DATASET(dev_data, dev_pseudolabels, True, self.temporal_window_samples, self.context_window_samples)
    dev_dataloader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)

    test_dataset = BEHAVIOR_DATASET(test_data, test_pseudolabels, False, self.temporal_window_samples, self.context_window_samples)
    num_examples_test = len(list(range(0, len(test_dataset), self.downsizing_factor)))
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), self.downsizing_factor)))
    print("Number windowed test examples after subselecting: %d" % len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
    loss_fn_no_reduce = nn.CrossEntropyLoss(ignore_index = -1, reduction = 'sum')
    optimizer = torch.optim.Adam([{'params' : self.encoder.parameters(), 'weight_decay' : self.weight_decay}, {'params' : self.decoder.parameters(), 'weight_decay' : self.weight_decay}], lr=self.lr, amsgrad = True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs, eta_min=0, last_epoch=- 1, verbose=False)
    
    dev_loss = []
    test_loss = []
    dev_acc = []
    test_acc = []
    learning_rates = []
    
    epochs = self.n_epochs
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        l, a = self.train_epoch(t, dev_dataloader, loss_fn, optimizer)
        dev_loss.append(l)
        dev_acc.append(a)
        l, a = self.test_epoch(t, test_dataloader, 
                               loss_fn_no_reduce, 
                               name = "Test", 
                               loss_denom = num_examples_test* self.temporal_window_samples * self.context_window_samples)
        test_loss.append(l)
        test_acc.append(a)
        
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
      
    print("Done!")
    
    ## Save training progress
    
    # Loss
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    ax.plot(dev_loss, label= 'dev', marker = '.')
    ax.plot(test_loss, label = 'test', marker = '.')
    ax.legend()
    ax.set_title("Cross Entropy Loss")
    ax.set_xlabel('Epoch')
    
    major_tick_spacing = max(1, len(dev_loss) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Loss')
    loss_fp = os.path.join(self.config['output_dir'], 'loss.png')
    fig.savefig(loss_fp)
    plt.close()

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(dev_acc, label= 'dev', marker = '.')
    ax.plot(test_acc, label = 'test', marker = '.')
    ax.legend()
    ax.set_title("Mean accuracy")
    ax.set_xlabel('Epoch')
    major_tick_spacing = max(1, len(dev_acc) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Accuracy')
    acc_fp = os.path.join(self.config['output_dir'], 'acc.png')
    fig.savefig(acc_fp)
    plt.close()
    
    # Learning Rate
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(learning_rates, marker = '.')
    ax.set_title("Learning Rate")
    ax.set_xlabel('Epoch')
    major_tick_spacing = max(1, len(learning_rates) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    lr_fp = os.path.join(self.config['output_dir'], 'learning_rate.png')
    fig.savefig(lr_fp)
    plt.close()
    
  def train_epoch(self, t, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    self.encoder.train()
    gumbel_tau = self.tau_init * (self.tau_decay_rate ** t)
    # acc_score = torchmetrics.Accuracy(mdmc_average = 'global', ignore_index = -1)
    train_loss = 0
    num_batches_seen = 0
    
    num_batches_todo = 1 + len(dataloader) // self.downsizing_factor
    with tqdm.tqdm(dataloader, unit = "batch", total = num_batches_todo) as tepoch:
      for i, (X, y) in enumerate(tepoch):
        if i == num_batches_todo :
          break
        num_batches_seen += 1
        X, y = X.type('torch.FloatTensor').to(device), y.type('torch.LongTensor').to(device)
        
        # Compute prediction error
        latent_logits = self.encoder(X)
        q = torch.nn.functional.gumbel_softmax(latent_logits, tau=gumbel_tau, hard=True, dim=- 1)
        logits = self.decoder(q)
        loss = loss_fn(logits, y)
        train_loss += loss.item()

        # acc_score.update(logits.cpu(), y.cpu())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        loss_str = "%2.2f" % loss.item()
        tepoch.set_postfix(loss=loss_str)
        
    # acc = acc_score.compute()
    acc = 0.
    train_loss = train_loss / num_batches_seen
    print("Train loss: %f, Train accuracy: %f, Temperature: %f" % (train_loss, acc, gumbel_tau))
    return train_loss, acc
    
  def test_epoch(self, t, dataloader, loss_fn, name = "Test", loss_denom = 0):
    size = len(dataloader.dataset)
    self.encoder.eval()
    test_loss = 0
    gumbel_tau = self.tau_init * (self.tau_decay_rate ** t)
    # acc_score = torchmetrics.Accuracy(mdmc_average = 'global', ignore_index = -1)
    
    with torch.no_grad():
        num_batches_todo = 1 + len(dataloader) // self.downsizing_factor
        for i, (X, y) in enumerate(dataloader):
            X, y = X.type('torch.FloatTensor').to(device), y.type('torch.LongTensor').to(device)
            latent_logits = self.encoder(X)
            q = torch.nn.functional.gumbel_softmax(latent_logits, tau=gumbel_tau, hard=True, dim=- 1)
            logits = self.decoder(q)
            # acc_score.update(logits.cpu(), y.cpu())
            test_loss += loss_fn(logits, y).item()
    test_loss /= loss_denom
    #acc = acc_score.compute()
    acc = 0.
    print("%s loss: %f, %s accuracy: %f, Temperature: %f" % (name, test_loss, name, acc, gumbel_tau))
    return test_loss, acc
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    ###
    self.encoder.eval()
    alldata= data
    
    predslist = []
    pred_len = self.temporal_window_samples
    for i in range(0, np.shape(alldata)[0], pred_len):
      data = alldata[i:i+pred_len, :] # window to acommodate more hidden states without making edits to CUDA kernel
    
      with torch.no_grad():
        data = np.expand_dims(data, axis =0)
        data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        preds = self.encoder(data)
        preds = preds.cpu().detach().numpy()
        preds = np.squeeze(preds, axis = 0)
        preds = np.argmax(preds, axis = -1).astype(np.uint8)
        
        predslist.append(preds)
    preds = np.concatenate(predslist)
    return preds, None  
      
class S4Block(nn.Module):
    def __init__(self, H, N, dropout= 0.):
      super(S4Block, self).__init__()
      self.ln1 = nn.LayerNorm(H)
      self.s4 = S4(H, d_state = N, bidirectional = True, dropout = dropout, transposed = False)
      self.ln2 = nn.LayerNorm(H)
      self.linear2 = nn.Linear(H, 2*H)
      self.linear3 = nn.Linear(2*H, H)
      
    def forward(self, x):
      y = x
      x = self.ln1(x)
      x = self.s4(x)[0]
      x = y+ x
      
      y = x
      x = self.ln2(x)
      x = self.linear2(x)
      x = nn.functional.gelu(x)
      x = self.linear3(x)
      x = y+ x
      return x

class Encoder(nn.Module):
    def __init__(self, n_features, n_clusters, hidden_size, state_size, n_s4_blocks, downsample_rate, dropout, blur_scale = 0, jitter_scale = 0):
        super(Encoder, self).__init__()
        self.blur_scale = blur_scale
        self.jitter_scale = jitter_scale
        
        self.embedding = nn.Linear(n_features, hidden_size)
        
        self.bn = nn.BatchNorm1d(n_features)
        self.downsample_rate = downsample_rate
        feature_expansion_factor = 1 #2
        self.down1 = nn.Conv1d(hidden_size, feature_expansion_factor * hidden_size, self.downsample_rate, stride = self.downsample_rate)
        self.down2 = nn.Conv1d(feature_expansion_factor * hidden_size, (feature_expansion_factor ** 2) * hidden_size, self.downsample_rate, stride = self.downsample_rate)
        
        self.s4_blocks_1 = nn.ModuleList([S4Block(hidden_size, state_size, dropout = dropout) for i in range(n_s4_blocks)])
        self.s4_blocks_2 = nn.ModuleList([S4Block(feature_expansion_factor * hidden_size, state_size, dropout = dropout) for i in range(n_s4_blocks)])
        self.s4_blocks_3 = nn.ModuleList([S4Block((feature_expansion_factor ** 2) * hidden_size, state_size, dropout = dropout) for i in range(n_s4_blocks)])
        self.head = nn.Conv1d((feature_expansion_factor ** 2) * hidden_size, n_clusters, 1, padding = 'same')
        
    def forward(self, x):
        seq_len = x.size()[-2]
        
        x = torch.transpose(x, -1, -2)
        x = self.bn(x)
        x = torch.transpose(x, -1, -2)
        
        if self.training:
          # Perform augmentations to normalized data
          size = x.size()
          if self.blur_scale:
            blur = self.blur_scale * torch.randn(size, device = x.device)
          else:
            blur = 0.
          if self.jitter_scale:
            jitter = self.jitter_scale *torch.randn((size[0], 1, size[2]), device = x.device)
          else:
            jitter = 0.
          x = x + blur + jitter 
        
        x = self.embedding(x)
        
        for block in self.s4_blocks_1:
          x = block(x)
          
        x = torch.transpose(x, -1, -2)
        x = self.down1(x)
        x = torch.transpose(x, -1, -2)
        
        for block in self.s4_blocks_2:
          x = block(x)
          
        x = torch.transpose(x, -1, -2)
        x = self.down2(x)
        x = torch.transpose(x, -1, -2)
        
        for block in self.s4_blocks_3:
          x = block(x)
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        logits = self.head(x) # -> [batch, n_clusters, seq_len]
        
        logits = nn.functional.interpolate(logits, size=seq_len, mode='nearest-exact')
        logits = torch.transpose(logits, 1,2) # -> [batch, seq_len, n_clusters]
        return logits  


class Decoder(nn.Module):
  # Linear [batch, seq_len, n_clusters] (one-hot representation) -> [batch, seq_len, context_window, n_pseudolabels]
  def __init__(self, n_clusters, n_pseudolabels, context_window_samples):
      super(Decoder, self).__init__()
      
      self.prediction_heads = nn.ModuleList([nn.Linear(n_clusters, n_pseudolabels) for n in range(context_window_samples)])
      
  def forward(self, q):
      logits = []
      for head in self.prediction_heads:
        logits.append(head(q))
      logits = torch.stack(logits, axis = -2)
      
      logits = torch.transpose(logits, -1, -2)
      logits = torch.transpose(logits, -2, -3)
      return logits
    