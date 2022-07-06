import yaml
import sys
import argparse
import shutil
import os

import BEBE.evaluation.evaluation as evaluation
import BEBE.utils.experiment_setup as experiment_setup
import BEBE.training.train_model as train_model

def main(config, save_latents):
  expanded_config = experiment_setup.experiment_setup(config, save_latents = save_latents)
  model = train_model.train_model(expanded_config)
  
  evaluation.generate_predictions(model, expanded_config)
  evaluation.generate_evaluations(expanded_config)
  
  # Clean up
  if os.path.exists(expanded_config['temp_dir']):
    shutil.rmtree(expanded_config['temp_dir'])
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True)
  parser.add_argument('-save_latents', action='store_true')
  args = parser.parse_args()
  config_fp = args.config
  save_latents = args.save_latents
  
  with open(config_fp) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
  
  main(config, save_latents)
  
  
