


from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import (
    ActivatingExample,
    LatentDataset,
    LatentRecord,
)
import argparse
import torch

def main():
  # model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
  parser = argparse.ArgumentParser(prog='Co Activations', description='Finds Delphi activations and finds coactivations')
  parser.add_argument("--model_name", type=str, required=False, default="llama-3-8B", help="Name of model for delphi activations")
  args = parser.parse_args()
  model_name = args.model_name
  latent_dict = {
    "layers.5.mlp": torch.arange(0, 100)
  }
  sampler_cfg = SamplerConfig()
  constructor_cfg = ConstructorConfig()
# Delphi_Cluster_Co_Activations/src/data/llama-3-8B change this to raw dir?
  dataset = LatentDataset(
      raw_dir="src/data/llama-3-8B/latents",
      modules=["layers.5.mlp"], # This a list of the different caches to load from
      sampler_cfg=sampler_cfg,
      constructor_cfg=constructor_cfg,
      latents=latent_dict,
      tokenizer=None
  )
  print(dataset)

if __name__ == "__main__":
    main()
