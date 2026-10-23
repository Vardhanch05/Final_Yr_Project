"""
Converts all 8 TabReD datasets from uvx tabred download format to paper data format.
"""
import json
from pathlib import Path
import numpy as np

def prepare_all():
    src_base = Path("tabm/paper/data/tabred")
    dst_base = Path("tabm/paper/data")
    
    datasets = [
        "cooking-time",
        "delivery-eta",
        "ecom-offers",
        "homecredit-default",
        "homesite-insurance",
        "maps-routing",
        "sberbank-housing",
        "weather"
    ]
    
    for d_name in datasets:
        src = src_base / d_name
        dst = dst_base / d_name
        
        # If dst is a junction/symlink or already exists, check
        if dst.exists() and not (dst / "info.json").exists():
            import subprocess
            subprocess.run(["cmd", "/c", "rmdir", str(dst)])
            
        dst.mkdir(parents=True, exist_ok=True)
        
        # 1. info.json
        with open(src / "info.json") as f:
            info_raw = json.load(f)
            
        if "task" in info_raw:
            task_type = info_raw["task"]["type"]
            score = info_raw["task"]["score"]
        else:
            task_type = info_raw["task_type"]
            score = info_raw["score"]
        
        info_paper = {
            "task_type": task_type,
            "score": score
        }
        with open(dst / "info.json", "w") as f:
            json.dump(info_paper, f, indent=4)
            
        # 2. split indices
        split_dir = src / "splits" / "default"
        indices = {
            part: np.load(split_dir / f"{part}.npy")
            for part in ['train', 'val', 'test']
        }
        
        # 3. features and labels
        for fpath in src.glob("*.*"):
            fname = fpath.name.lower()
            if fname.startswith("x_") and not fname.startswith("x_meta"):
                key = fpath.stem.upper()
                parts = key.split("_")
                canonical_key = parts[0] + "_" + parts[1].lower() # X_num, X_bin, X_cat
                if not (dst / f"{canonical_key}_train.npy").exists():
                    arr = np.load(fpath)
                    for p, idx in indices.items():
                        np.save(dst / f"{canonical_key}_{p}.npy", arr[idx])
            elif fname == "y.npy":
                if not (dst / f"Y_train.npy").exists():
                    arr = np.load(fpath)
                    for p, idx in indices.items():
                        np.save(dst / f"Y_{p}.npy", arr[idx])
                        
        print(f"[+] Prepared {d_name} into {dst}")

if __name__ == '__main__':
    prepare_all()
