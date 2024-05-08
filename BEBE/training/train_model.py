def train_model(config, train=True):
  """
  Added train param for Cougar project. Default is True (orig behavior),
  override to False to skip model training and use already saved model.
  :param config:
  :param train:
  :return:
  """
  ## Instantiate model

  if config['model'] == 'gmm':
    from BEBE.models.gmm import gmm as m
    
  elif config['model'] == 'kmeans':
    from BEBE.models.kmeans import kmeans as m
    
  elif config['model'] == 'vame':
    from BEBE.models.vame import vame as m
    
  elif config['model'] == 'hmm':
    from BEBE.models.hmm import hmm as m
    
  elif config['model'] == 'CRNN':
    from BEBE.models.CRNN import CRNN as m
    
  elif config['model'] == 'umapper':
    from BEBE.models.umapper import umapper as m
    
  elif config['model'] == 'rf':
    from BEBE.models.classic import RandomForest as m

  elif config['model'] == 'dt':
    from BEBE.models.classic import DecisionTree as m

  elif config['model'] == 'svm':
    from BEBE.models.classic import SupportVectorMachine as m

  elif config['model'] == 'random':
    from BEBE.models.random import random as m
    
  elif config['model'] == 'iic':
    from BEBE.models.iic import iic as m
    
  elif config['model'] == 'harnet':
    from BEBE.models.harnet import harnet as m

  else:
    raise ValueError('model type not recognized')
    
  model = m(config)

  if train:
    # Train model
    print("Training model")
    model.fit()

    # Save model
    model.save()
  else:
    print("Loading model")
    model.load()
  
  return model
