# Sanity checks for the WTConv encoder variant (FCN_Encoder_WidtherFeature_WT).
# Run from the repo root BEFORE spending GPU time:
#   python3 tests/test_wtconv_equivalence.py [--checkpoint outputs/.../best-IAM_NER_165.pt]
#
# Checks:
#   1. every parameter of the baseline encoder loads into the WT encoder by
#      name/shape (nothing "unexpected"; missing keys are only wavelet ones);
#   2. with the zero-init gates, both encoders compute the SAME function
#      (i.e., the transfer learning is provably intact at step 0);
#   3. the wavelet gates receive non-zero gradients (the branch can learn).
import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basic.encoders import FCN_Encoder_WidtherFeature, FCN_Encoder_WidtherFeature_WT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="optional path to a real checkpoint (e.g. best-IAM_NER_165.pt) "
                             "to use as source weights instead of a random baseline")
    args = parser.parse_args()

    torch.manual_seed(0)
    params = {"dropout": 0.5, "input_channels": 1, "wt_levels": 2}

    base = FCN_Encoder_WidtherFeature(params).eval()
    wt = FCN_Encoder_WidtherFeature_WT(params).eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        source_state = ckpt["encoder_state_dict"]
        base.load_state_dict(source_state, strict=True)
    else:
        source_state = base.state_dict()

    # 1. loading: mimics what the training manager does with strict=False
    result = wt.load_state_dict(source_state, strict=False)
    assert not result.unexpected_keys, \
        "checkpoint keys not accepted by the WT encoder: {}".format(result.unexpected_keys)
    non_wavelet_missing = [k for k in result.missing_keys
                           if not any(s in k for s in ("wavelet_convs", "wavelet_scale", "wt_filter", "iwt_filter"))]
    assert not non_wavelet_missing, \
        "pretrained params NOT loaded (name/shape mismatch): {}".format(non_wavelet_missing)
    print("[1/3] OK: all {} pretrained params load; only {} wavelet keys are new".format(
        len(source_state), len(result.missing_keys)))

    # 2. equivalence: zero gates -> identical function
    x = torch.randn(1, 1, 160, 192)
    with torch.no_grad():
        out_base = base(x)
        out_wt = wt(x)
    max_diff = (out_base - out_wt).abs().max().item()
    assert torch.allclose(out_base, out_wt, atol=1e-5), \
        "outputs differ (max abs diff {:.2e}) -> transfer NOT intact".format(max_diff)
    print("[2/3] OK: identical outputs at init (max abs diff {:.2e})".format(max_diff))

    # 3. gradients: the gates must be able to move away from zero
    wt.zero_grad()
    wt(x).sum().backward()
    scale_grads = [p.grad.abs().sum().item()
                   for m in wt.modules() if hasattr(m, "wavelet_scale")
                   for p in m.wavelet_scale]
    assert scale_grads and any(g > 0 for g in scale_grads), \
        "wavelet gates receive no gradient -> branch would never learn"
    print("[3/3] OK: {}/{} wavelet gates receive non-zero gradient".format(
        sum(g > 0 for g in scale_grads), len(scale_grads)))

    n_new = sum(p.numel() for n, p in wt.named_parameters()
                if "wavelet" in n)
    print("\nPASS — WT encoder adds {:,} trainable wavelet params; transfer learning intact.".format(n_new))


if __name__ == "__main__":
    main()
