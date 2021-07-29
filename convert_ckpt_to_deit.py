from collections import OrderedDict
import argparse

import torch


def convert_to_deit(input_file, output_file):
    if output_file == "":
        output_file = input_file.replace(".ckpt", "") + "_to_deit.pth"

    old_ckpt = torch.load(input_file, map_location="cpu")["model"]
    new_ckpt = OrderedDict()
    for k in old_ckpt.keys():
        if k.startswith('module.trunk.'):
            new_ckpt[k[len('module.trunk.'):]] = old_ckpt[k]
    torch.save(new_ckpt, output_file)
    print(
        f"converted MoCo v3 checkpoint\n\t{input_file}\nto DeiT model\n\t{output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    opts = parser.parse_args()
    convert_to_deit(opts.input, opts.output)
