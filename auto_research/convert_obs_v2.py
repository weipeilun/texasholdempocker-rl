"""Convert an observation-version-1 checkpoint (R004 and earlier) into a current-layout weights file with the same
network function (rl/splendor_model.convert_checkpoint_to_current_observation): the derived fields of observation
version 2 start embedded as zeros, the card / noble rows are rescaled so every token is unchanged.

    python auto_research/convert_obs_v2.py --config config/splendor_auto_r005.yml IN.pth OUT.pth
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rl.splendor_model import SplendorAlphaGoZero, convert_checkpoint_to_current_observation   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='a config whose model_param_dict has the network dimensions')
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()
    with open(args.config, encoding='UTF-8') as f:
        params = yaml.safe_load(f)
    model_params = dict(params['model_param_dict'])
    model_params.update(device='cpu', inference_cuda_graph_batch_sizes=None)
    out = convert_checkpoint_to_current_observation(SplendorAlphaGoZero(**model_params), args.source, args.target)
    print(out)


if __name__ == '__main__':
    main()
