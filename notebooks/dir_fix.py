import os
import re
import shutil

# config
source = "cache"
target = "cache_organized"
dry_run = False  # set to True to just print actions

if not os.path.exists(target):
    os.makedirs(target)

print(f"Organizing files from '{source}' to '{target}'...")
if dry_run:
    print("--- DRY RUN MODE ---")


# walk through the source dir
for root, _, files in os.walk(source):
    for fname in files:
        src_path = os.path.join(root, fname)
        dst_path = None

        # Rule 1: Probes (.pth) - using the corrected regex
        probe_match = re.search(
            r"((?:[a-zA-Z0-9_]+_)?(?:linear|mlp|voxel))_layer_(\d+)_probe\.pth", fname
        )
        if probe_match:
            probe_type, layer = probe_match.groups()
            exp_match = re.search(r"(phase\d_[a-zA-Z0-9_]+)", src_path)
            if exp_match:
                exp_name = exp_match.group(1)
                dst_folder = os.path.join(target, exp_name, "probes")
                dst_path = os.path.join(dst_folder, f"{probe_type}_layer_{layer}.pth")

        # Rule 2: Handle .pkl Feature Files (using your broad logic)
        elif fname.endswith(".pkl"):
            feat_match = re.search(
                r"_(\d+)_(train|val|test)_reconstruction_prepared_data\.pkl", fname
            )
            if feat_match:
                layer, split = feat_match.groups()
                exp_match = re.search(r"(phase\d_[a-zA-Z0-9_]+)", src_path)
                if exp_match:
                    exp_name = exp_match.group(1)
                    dst_folder = os.path.join(target, exp_name, "features")
                    dst_path = os.path.join(dst_folder, f"layer_{layer}_{split}.pkl")

        # Rule 3: Handle loose feature files (using your broad logic)
        elif "features" in src_path.split(os.sep):
            feat_match = re.match(r"(.*?)_layer_(\d+)_(train|val|test)", fname)
            if feat_match:
                exp_name, layer, split = feat_match.groups()
                ext = os.path.splitext(fname)[1] or ".pkl"
                dst_folder = os.path.join(target, exp_name, "features")
                dst_path = os.path.join(dst_folder, f"layer_{layer}_{split}{ext}")

        if dst_path:
            print(f"{src_path}  ->  {dst_path}")
            if not dry_run:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)

print("Done.")
