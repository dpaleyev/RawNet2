import argparse
import json
import torch
from pathlib import Path
import os
import torchaudio

import hw_as.model as module_model
from hw_as.utils.parse_config import ConfigParser
from hw_as.logger import get_visualizer


def main(model, input_dir, output_file, device, writer):
    input_dir = Path(input_dir)
    
    with open(output_file, 'w') as f:    
        for path in input_dir.glob("*.flac"):
            audio = torchaudio.load(path)[0].flatten().unsqueeze(0).to(device)
            out = model(audio)
            prob = torch.softmax(out["preds"], dim=-1)
            f.write(f"{os.path.split(path)[-1]}: bonafide prob: {prob[0][0].item()} spoof prob: {prob[0][1].item()}\n")

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="Path to checkpoint"
    )

    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="Path to config"
    )

    args.add_argument(
        "-i",
        "--input",
        default="./test_data",
        type=str,
        help="Path to input directory"
    )

    args.add_argument(
        "-o",
        "--output",
        default="./synth_result",
        type=str,
        help="Path to output directory"
    )

    args = args.parse_args()

    assert args.resume is not None, "Please specify checkpoint path"

    with open(args.config) as f:
        config = ConfigParser(json.load(f))
    
    logger = config.get_logger("test")
    writer = get_visualizer(
            config, logger, "wandb"
        )
    
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(args.resume, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    main(model, args.input, args.output, device, writer)
