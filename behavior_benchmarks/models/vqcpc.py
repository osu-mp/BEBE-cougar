import yaml
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchmetrics
from behavior_benchmarks.models.vqcpc_dataloader import CPCDataset
from behavior_benchmarks.models.scheduler import WarmupScheduler
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from itertools import chain
import torch.optim as optim


import torch.nn.functional as F
from torch.distributions import Categorical

from tqdm import tqdm
import math

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class vq_cpc():
  def __init__(self, config):
    self.config = config
    self.read_latents = config['read_latents']
    self.model_config = config['vq_cpc_config']
    self.metadata = config['metadata']
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    
    ##
    self.downsizing_factor = self.model_config['downsizing_factor']
    self.lr = self.model_config['lr']
    # self.weight_decay = self.model_config['weight_decay']
    self.n_epochs = self.model_config['n_epochs']
    self.conv_stack_hidden_size = self.model_config['conv_stack_hidden_size']
    self.temporal_window_samples = self.model_config['temporal_window_samples']
    self.predict_proportion = self.model_config['predict_proportion']
    self.encoder_kernel_width = self.model_config['encoder_kernel_width']
    self.batch_size =self.model_config['batch_size']
    # self.dropout = self.model_config['dropout']
    # self.blur_scale = self.model_config['blur_scale']
    # self.jitter_scale = self.model_config['jitter_scale']
    # self.rescale_param = self.model_config['rescale_param']
    self.conv_stack_depth = self.model_config['conv_stack_depth']
    self.z_dim = self.model_config['z_dim']
    self.c_dim = self.model_config['c_dim']
    self.warmup_epochs = self.model_config['warmup_epochs']
    self.initial_lr = self.model_config['initial_lr']
    self.blur_scale = self.model_config['blur_scale']
    self.jitter_scale = self.model_config['jitter_scale']
    self.pooling_factor = self.model_config['pooling_factor']
    # ##
    
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_classes = len(self.metadata['label_names']) 
    self.n_features = len(self.cols_included)
    
    ##
    self.model = Encoder(self.n_features, self.conv_stack_hidden_size, self.config['num_clusters'], self.z_dim, self.c_dim, self.encoder_kernel_width, self.conv_stack_depth, blur_scale = self.blur_scale, jitter_scale = self.jitter_scale)
    self.cpc = CPCLoss(1, self.batch_size, int(self.temporal_window_samples * self.predict_proportion / self.pooling_factor), 8, self.z_dim, self.c_dim)
    self.model.to(device)
    self.cpc.to(device)
    ##
    
    #print(self.model)
    print('Model parameters:')
    print(_count_parameters(self.model))
  
  def load_model_inputs(self, filepath, read_latents = False):
    if read_latents:
      raise NotImplementedError("Supervised model is expected to read from raw data")
      #return np.load(filepath)
    else:
      return np.load(filepath)[:, self.cols_included] #[n_samples, n_features]
    
  def load_labels(self, filepath):
    labels = np.load(filepath)[:, self.label_idx].astype(int)
    return labels 
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
      test_fps = self.config['test_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
      test_fps = self.config['test_data_fp']
    
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in dev_fps]
    test_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in test_fps]
    
    dev_labels = [self.load_labels(fp) for fp in dev_fps]
    test_labels = [self.load_labels(fp) for fp in test_fps]
    
    ###
    dev_dataset = CPCDataset(dev_data, True, self.temporal_window_samples)
    dev_dataloader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    test_dataset = CPCDataset(test_data, False, self.temporal_window_samples)
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), self.downsizing_factor)))
    print("Number windowed test examples after subselecting: %d" % len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index = self.unknown_label)
    loss_fn_no_reduce = nn.CrossEntropyLoss(ignore_index = self.unknown_label, reduction = 'sum')
    #optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad = True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs, eta_min=0, last_epoch=- 1, verbose=False)
    
    optimizer = optim.Adam(
        chain(self.model.parameters(), self.cpc.parameters()),
        lr=self.initial_lr)
    
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=self.warmup_epochs,
        initial_lr=self.initial_lr,
        max_lr=self.lr,
        milestones= [20000],
        gamma=0.25)
    
    train_cpc_loss = []
    train_vq_loss = []
    train_perplexity = []
    train_accuracy = []
    
    test_cpc_loss = []
    test_vq_loss = []
    test_perplexity = []
    test_accuracy = []
    
    learning_rates = []
    
    for epoch in range(self.n_epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        cpc_loss, vq_loss, perplexity, accuracy = self.train_epoch(dev_dataloader, loss_fn, optimizer)
        train_cpc_loss.append(cpc_loss)
        train_vq_loss.append(vq_loss)
        train_perplexity.append(perplexity)
        train_accuracy.append(accuracy)
        
        cpc_loss, vq_loss, perplexity, accuracy = self.test_epoch(test_dataloader, loss_fn, optimizer)
        test_cpc_loss.append(cpc_loss)
        test_vq_loss.append(vq_loss)
        test_perplexity.append(perplexity)
        test_accuracy.append(accuracy)
        
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        

    print("Done!")
    
    ## Save training progress
    
    # Loss
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    ax.plot(train_cpc_loss, label= 'train_cpc', marker = '.')
    ax.plot(test_cpc_loss, label = 'test_cpc', marker = '.')
    ax.plot(train_vq_loss, label= 'train_vq', marker = '.')
    ax.plot(test_vq_loss, label = 'test_vq', marker = '.')
    ax.legend()
    ax.set_title("Loss")
    ax.set_xlabel('Epoch')
    
    major_tick_spacing = max(1, len(train_cpc_loss) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Loss')
    loss_fp = os.path.join(self.config['output_dir'], 'loss.png')
    fig.savefig(loss_fp)
    plt.close()

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(train_accuracy, label= 'train', marker = '.')
    ax.plot(test_accuracy, label = 'test', marker = '.')
    ax.legend()
    ax.set_title("Mean accuracy")
    ax.set_xlabel('Epoch')
    major_tick_spacing = max(1, len(train_accuracy) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Accuracy')
    acc_fp = os.path.join(self.config['output_dir'], 'acc.png')
    fig.savefig(acc_fp)
    plt.close()
    
    # Perplexity
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(train_perplexity, label= 'train', marker = '.')
    ax.plot(test_perplexity, label = 'test', marker = '.')
    ax.legend()
    ax.set_title("Perplexity")
    ax.set_xlabel('Epoch')
    major_tick_spacing = max(1, len(train_perplexity) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Perplexity')
    perp_fp = os.path.join(self.config['output_dir'], 'perplexity.png')
    fig.savefig(perp_fp)
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
    
    ###
    ##
    
  def train_epoch(self, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    self.model.train()
    num_batches_seen = 0
    
    num_batches_todo = 1 + len(dataloader) // self.downsizing_factor
    
    vq_losses = []
    perplexities = []
    cpc_losses = []
    accuracies = []
    losses = []
    
    with tqdm(dataloader, unit = "batch", total = num_batches_todo) as tepoch:
      for i, X in enumerate(tepoch):
        if i == num_batches_todo :
          break
        X = X.type('torch.FloatTensor').to(device)
        #X = X.view(cfg.training.n_speakers_per_batch *cfg.training.n_utterances_per_speaker,cfg.preprocessing.n_mels, -1)

        optimizer.zero_grad()

        z, c, vq_loss, perplexity = self.model(X)
        cpc_loss, accuracy = self.cpc(z, c)
        loss = cpc_loss + vq_loss
        
        vq_losses.append(vq_loss.item())
        perplexities.append(perplexity.item())
        cpc_losses.append(cpc_loss.item())
        accuracies.extend(accuracy)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        num_batches_seen += 1
        
        loss_str = "%2.2f" % loss.item()
        tepoch.set_postfix(loss=loss_str)
        
    
    train_loss = np.mean(losses)
    vq_loss = np.mean(vq_losses)
    perplexity = np.mean(perplexities)
    cpc_loss = np.mean(cpc_losses)
    accuracy = np.mean(accuracies)
    print("Train loss: %f, Train accuracy: %f" % (train_loss, accuracy))
    return cpc_loss, vq_loss, perplexity, accuracy
    
  def test_epoch(self, dataloader, loss_fn, name = "Test", loss_denom = 0):
    self.model.eval()
    
    vq_losses = []
    perplexities = []
    cpc_losses = []
    accuracies = []
    losses = []
    
    with torch.no_grad():
      for i, X in enumerate(dataloader):
        
        X = X.type('torch.FloatTensor').to(device)
        #X = X.view(cfg.training.n_speakers_per_batch *cfg.training.n_utterances_per_speaker,cfg.preprocessing.n_mels, -1)

        z, c, vq_loss, perplexity = self.model(X)
        cpc_loss, accuracy = self.cpc(z, c)
        loss = cpc_loss + vq_loss
        
        vq_losses.append(vq_loss.item())
        perplexities.append(perplexity.item())
        cpc_losses.append(cpc_loss.item())
        accuracies.extend(accuracy)
        losses.append(loss.item())

        
    
    test_loss = np.mean(losses)
    vq_loss = np.mean(vq_losses)
    perplexity = np.mean(perplexities)
    cpc_loss = np.mean(cpc_losses)
    accuracy = np.mean(accuracies)
    print("Test loss: %f, Test accuracy: %f" % (test_loss, accuracy))
    return cpc_loss, vq_loss, perplexity, accuracy
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    ###
    self.model.eval()
    with torch.no_grad():
      data = np.expand_dims(data, axis =0)
      
      
      
      X = torch.from_numpy(data).type('torch.FloatTensor').to(device)
      latents, _, preds = self.model.encode(X)
      
      
      # data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
      # latents, _, preds = self.model.encode(data)
      ####
      preds = preds.cpu().detach().numpy()
      latents = latents.cpu().detach().numpy()
      # print("preds shape:")
      # print(np.shape(preds))
      # print("latents shape:")
      # print(np.shape(latents))
      preds = np.squeeze(preds, axis = 0).astype(np.uint8)
      latents = np.squeeze(latents, axis = 0)
      #preds = np.argmax(preds, axis = 0).astype(np.uint8)
      #print(preds.dtype)
    return preds, latents
    ###
  
  def predict_from_file(self, fp):
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    predictions, latents = self.predict(inputs)
    return predictions, latents

class Encoder(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim, kernel_width, conv_stack_depth, blur_scale = 0, jitter_scale = 0, pooling_factor = 1):
        super(Encoder, self).__init__()
        # self.conv = nn.Conv1d(in_channels, channels, 4, 2, 1, bias=False)
        self.pool_size = pooling_factor
        
        self.blur_scale = blur_scale
        self.jitter_scale = jitter_scale
        
        self.codebook = VQEmbeddingEMA(n_embeddings, z_dim)
        self.rnn = nn.LSTM(z_dim, c_dim, batch_first=True)
        
        self.conv_stack = [_conv_block(in_channels, channels, channels-in_channels, kernel_width)]
        for i in range(conv_stack_depth - 1):
          self.conv_stack.append(_conv_block(channels, channels, channels, kernel_width)) 
        self.conv_stack = nn.ModuleList(self.conv_stack)
        #self.head = nn.Conv1d(channels, z_dim, 1, padding = 'same')
        
        pooling = nn.AvgPool1d(self.pool_size)
        self.head = nn.Sequential(pooling, nn.Conv1d(channels, z_dim, 1, padding = 'same'))
        
        self.bn = torch.nn.BatchNorm1d(in_channels)
        

    def encode(self, x):
        # z = self.conv(mel)
        # z = self.encoder(z.transpose(1, 2))
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        
        
        x_len_samples = x.size()[-1] 
        pad_length = self.pool_size - (x_len_samples % self.pool_size)
        
        x = nn.functional.pad(x, (0, pad_length))
        
        
        norm_inputs = self.bn(x)
        
        x = self.conv_stack[0](norm_inputs)
        x = torch.cat([x, norm_inputs], axis = 1)
      
        for layer in self.conv_stack[1:]:
          x = layer(x) + x
        
        x = self.head(x)
        z = torch.transpose(x, 1,2) #[batch, seq_len, n_features]
        
        z, indices = self.codebook.encode(z)
        
        
        c, _ = self.rnn(z)
        
        # upsample predictions and quantized latents for analysis
        upsampler = nn.Upsample(scale_factor=self.pool_size, mode='nearest')
        
        indices = torch.unsqueeze(indices, 1)
        indices = upsampler(indices.type('torch.FloatTensor'))
        indices = torch.squeeze(indices, 1)
        indices = indices[:, :x_len_samples].type('torch.LongTensor')
        
        z = torch.transpose(z, 1,2) #[batch, n_features, seq_len]
        z = upsampler(z)
        z = z[:, :, :x_len_samples]
        z = torch.transpose(z, 1,2) #[batch, seq_len, n_features]
        
        ##
        # upsampling of c not currently implemented
        ##
        
        return z, c, indices

    def forward(self, x):
        # z = self.conv(mel)
        # z = self.encoder(z.transpose(1, 2))
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        norm_inputs = self.bn(x)
        
        if self.training:
          # Perform augmentations to normalized data
          size = norm_inputs.size()
          blur = self.blur_scale * torch.randn(size, device = norm_inputs.device)
          jitter = self.jitter_scale *torch.randn((size[0], size[1], 1), device = norm_inputs.device)
          norm_inputs = norm_inputs + blur + jitter 
          
        x_len_samples = x.size()[-1] #int(x.size()[-1].item())
        pad_length = x_len_samples % self.pool_size
        x = nn.functional.pad(x, (0, pad_length))
        
        x = self.conv_stack[0](norm_inputs)
        x = torch.cat([x, norm_inputs], axis = 1)
      
        for layer in self.conv_stack[1:]:
          x = layer(x) + x
        
        x = self.head(x)
        z = torch.transpose(x, 1,2)
        z, loss, perplexity = self.codebook(z)       
        
        c, _ = self.rnn(z)
        return z, c, loss, perplexity


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class CPCLoss(nn.Module):
    def __init__(self, n_speakers_per_batch, n_utterances_per_speaker, n_prediction_steps, n_negatives, z_dim, c_dim):
        super(CPCLoss, self).__init__()
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_prediction_steps = n_prediction_steps 
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, z, c):
        length = z.size(1) - self.n_prediction_steps

        z = z.reshape(
            self.n_speakers_per_batch,
            self.n_utterances_per_speaker,
            -1,
            self.z_dim
        )
        c = c[:, :-self.n_prediction_steps, :]

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            z_shift = z[:, :, k:length + k, :]

            Wc = self.predictors[k-1](c)
            Wc = Wc.view(
                self.n_speakers_per_batch,
                self.n_utterances_per_speaker,
                -1,
                self.z_dim
            )

            batch_index = torch.randint(
                0, self.n_utterances_per_speaker,
                size=(
                    self.n_utterances_per_speaker,
                    self.n_negatives
                ),
                device=z.device
            )
            batch_index = batch_index.view(
                1, self.n_utterances_per_speaker, self.n_negatives, 1
            )

            seq_index = torch.randint(
                1, length,
                size=(
                    self.n_speakers_per_batch,
                    self.n_utterances_per_speaker,
                    self.n_negatives,
                    length
                ),
                device=z.device
            )
            seq_index += torch.arange(length, device=z.device)
            seq_index = torch.remainder(seq_index, length)

            speaker_index = torch.arange(self.n_speakers_per_batch, device=z.device)
            speaker_index = speaker_index.view(-1, 1, 1, 1)

            z_negatives = z_shift[speaker_index, batch_index, seq_index, :]

            zs = torch.cat((z_shift.unsqueeze(2), z_negatives), dim=2)

            f = torch.sum(zs * Wc.unsqueeze(2) / math.sqrt(self.z_dim), dim=-1)
            f = f.view(
                self.n_speakers_per_batch * self.n_utterances_per_speaker,
                self.n_negatives + 1,
                -1
            )

            labels = torch.zeros(
                self.n_speakers_per_batch * self.n_utterances_per_speaker, length,
                dtype=torch.long, device=z.device
            )

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies

def _conv_block(in_dims, hidden_dims, out_dims, kernel_width):
  block = nn.Sequential(
    nn.Conv1d(in_dims,hidden_dims, kernel_width, bias= False, padding = 'same'),
    torch.nn.BatchNorm1d(hidden_dims),
    nn.ReLU(),
    nn.Conv1d(hidden_dims,out_dims,kernel_width, bias= False, padding = 'same'),
    torch.nn.BatchNorm1d(out_dims),
    nn.ReLU()
  )
  
  return block