

from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import (
    ActivatingExample,
    LatentDataset,
    LatentRecord,
)
import argparse
import torch
import os
import json
from collections import defaultdict
from safetensors import safe_open
import math
from huggingface_hub import login
from tqdm import tqdm
import json

# math functions to in code and decode features from attribute mlp
def cantor(num1, num2):
    return (num1 + num2) * (num1 + num2 + 1) // 2 + num2


def cantor_decode(num):
    w = math.floor((math.sqrt(8 * num + 1) - 1) / 2)
    t = (w * w + w) // 2
    y = num - t
    x = w - y
    return x, y


def get_attribution_features(GRAPH_PATH: str):
  # path to the attribution graph json
  with open(GRAPH_PATH, "r") as f:
    graph_data = json.load(f)
  # gets all the features in the attribution graph and filters out error nodes features
  attribution_features = set([data["feature"] for data in graph_data["nodes"]])
  attribution_features.remove(None)
  return attribution_features

def collect_attribution_graph_delphi(dataset: LatentDataset, attribution_features: set) -> dict:
  node_data = defaultdict(dict)
  for b in dataset.buffers:
    # Load raw data for all latents in this buffer
    latents, split_locations, split_activations = b.load_data_per_latent()
    current_layer = int(b.module_path.split(".")[1]) # might change depending on model
    for i, latent in enumerate(latents):

      # cantor(layer, feature) in delphi it was
      delphi_feature = int(cantor(current_layer, latent))
      #if feature in attribution graph store it
      if delphi_feature in attribution_features:
          latent_locations = split_locations[i]
          latent_activations = split_activations[i]
          node_data[delphi_feature] = {
            "layer": current_layer,
            "layer_latent/feature": latent,
            "activations": latent_activations,
            "locations": latent_locations,
          }
  return node_data
def sparse_mat(nodedata):
  cantor_ids = list(nodedata.keys())
  cantor_to_idx = {cid: idx for idx, cid in enumerate(cantor_ids)}
  idx_to_cantor = {idx: cid for idx, cid in enumerate(cantor_ids)}
  n_features = len(cantor_ids)
  all_feature_indices = []
  all_position_indices = []
  # makes a dataset of feature, locations
  for cantor_id, data in tqdm(nodedata.items(), desc="Processing features"):
      feature_idx = cantor_to_idx[cantor_id]
      locations = data["locations"]  # Shape: [N, 2] or [N, 3] -> (batch, token_pos, ...)
      
      # Convert locations to tensor if needed
      if not isinstance(locations, torch.Tensor):
          locations = torch.tensor(locations)
      
      # Extract batch and token position
      batch_indices = locations[:, 0].long()
      token_indices = locations[:, 1].long()
      
      # Create unique position ID using Cantor pairing
      position_cantor = ((batch_indices + token_indices) * (batch_indices + token_indices + 1)) // 2 + token_indices
      # Store pairs
      all_feature_indices.extend([feature_idx] * len(position_cantor))
      all_position_indices.extend(position_cantor.tolist())
  # one tensor with all features one with positions * features
  feature_indices = torch.tensor(all_feature_indices, dtype=torch.long)
  position_indices = torch.tensor(all_position_indices, dtype=torch.long)
  unique_positions, position_indices = torch.unique(position_indices, return_inverse=True)
  # matrix latent, position_cantor (batch, position)    
  sparse_matrix = torch.sparse_coo_tensor(
      torch.stack([feature_indices, position_indices]),
      torch.ones(len(feature_indices)),
      (n_features, len(unique_positions))
  )
  return sparse_matrix, idx_to_cantor, cantor_to_idx
# from delphi
def compute_jaccard(cooc_matrix):
        self_occurrence = cooc_matrix.diagonal()
        jaccard_matrix = cooc_matrix / (
            self_occurrence[:, None] + self_occurrence - cooc_matrix
        )
        # remove the diagonal and keep the upper triangle
        return jaccard_matrix


def main():
  # model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
  GRAPH_PATH = "/mnt/ssd-1/soar-automated_interpretability/graphs/pawan/circuit-grouping/data/bees/attribution_graph/graph_data/bees.json"
  CACHE_PATH = "/mnt/ssd-1/soar-automated_interpretability/graphs/pawan/delphi/results/gemma2b_transcoder-sparsify-1m_cache"
  TRANSCODER_PATH = "/mnt/ssd-1/soar-automated_interpretability/graphs/pawan/results/gemmma-scope-2b-pt-transcoders"

  parser = argparse.ArgumentParser(prog='Co Activations', description='Finds Delphi activations and finds coactivations')
  parser.add_argument("--attribution_graph", type=str, required=False, default=GRAPH_PATH, help="Json of the attribution graph from attribute not circuit tracer")

  parser.add_argument("--delphi_activations", type=str, required=False, default=CACHE_PATH, help="Delphi activation cache data to use for latent dataset")
  parser.add_argument("--transcoder", type=str, required=False, default=TRANSCODER_PATH, help="Delphi activation cache data to use for latent dataset")
  parser.add_argument("--huggingface_token", type=str, required=True, help="Huggingface token for latentdataset")
  parser.add_argument("--debug", default=False, required=False, help="Debugging shows prints")
  parser.add_argument("--number_of_neighbours", type=int, default = 10, required=False, help="Number of neighbours")
  parser.add_argument("--save", default=False, required=False, help="Save as JSON")
  args = parser.parse_args()

  #logins into huggingface
  login(token=args.huggingface_token)

  with open(GRAPH_PATH, "r") as f:
    graph_data = json.load(f)
  if (args.debug):
    print(f"Prompt: {graph_data['metadata']['prompt']}")
    print(f"Total nodes: {len(graph_data['nodes'])}")
  attribution_features = get_attribution_features(GRAPH_PATH)

  entries = os.listdir(args.transcoder)


  latent_dict = {
    layer: torch.arange(0, 16384) for layer in entries
  }

  sampler_cfg = SamplerConfig()
  constructor_cfg = ConstructorConfig()

  # Delphi_Cluster_Co_Activations/src/data/llama-3-8B change this to raw dir?
  dataset = LatentDataset(
      raw_dir= args.delphi_activations,
      modules=entries, # This a list of the different caches to load from
      sampler_cfg=sampler_cfg,
      constructor_cfg=constructor_cfg,
      latents=latent_dict,
      tokenizer=None
  )

  nodedata = collect_attribution_graph_delphi(dataset, attribution_features)
  sparse_matrix, idx_to_c, c_to_idx = sparse_mat(nodedata) # makes sparse_matrixs from list of features and the positions dict
  if (args.debug):
    print("sparse_matrix done")
  co_occurrence_matrix = (sparse_matrix @ sparse_matrix.T).to_dense()
  jaccard = compute_jaccard(co_occurrence_matrix)
  if (args.debug):
    print("coocurrence matrix and jaccard done")
  # top k for each feature
  top_k_values, top_k_indices = torch.topk(
          jaccard, args.number_of_neighbours + 1, dim=1
      )

  neighbours_list = {
    idx_to_c[i]: [
        (idx_to_c[neighbor_idx], similarity) 
        for neighbor_idx, similarity in zip(
            top_k_indices[i].tolist()[1:],  # Skip first (self)
            top_k_values[i].tolist()[1:]     # Skip first (self)
        )
    ]
    for i in range(len(top_k_indices))
  }
  if (args.debug):
    print("First 5 neighbours")

    test = list(neighbours_list.keys())[:5]
    for t in test:
      print(neighbours_list[t])
  if(args.save):
    with open("neighbors_dict.json", "w") as f:
      json.dump(neighbours_list, f)

if __name__ == "__main__":
    main()
