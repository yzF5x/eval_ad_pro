QUESTION_WITH_TAG = """Are there any defects or anomalies in this image? Answer with the word 'Yes' or 'No'. Before answering, perform a structured visual assessment in the following order:
Overview: Briefly describe the overall content, context, and general appearance of the image.
Analysis: Examine the image closely for any defects, irregularities, or anomalies. If none are present, explain why the image appears normal and consistent with expected standards.
Conclusion: Provide a one-sentence summary that states the final judgment on whether any defects or anomalies exist.
Output your response strictly in this format—without any additional text or tags outside the specified structure:
<overview>...</overview><analyze>...</analyze><conclusion>...</conclusion><answer>Yes/No</answer> """


DATASET_DEFAULTS = {
    "mvtec": {
        "dataset_path": "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/MVTecAD_seg_0shot.tsv",
        "output_root": "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/refactor-per-head-sinkfirst/MVTecAD_seg_0shot",
        "replace_path": "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/dataset/anomaly/MVTec-AD/",
    },
    "sdd": {
        "dataset_path": "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/SDD_seg_0shot.tsv",
        "output_root": "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/refactor-per-head-sinkfirst/SDD_seg_0shot",
        "replace_path": "/home/yizhou/LVLM/dataset/TEST_DATASET/SDD/SDD/",
    },
    "dtd": {
        "dataset_path": "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/DTD_seg_0shot.tsv",
        "output_root": "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/refactor-per-head-sinkfirst/DTD_seg_0shot",
        "replace_path": "/home/yizhou/LVLM/dataset/TEST_DATASET/dtd/",
    },
    "wfdd": {
        "dataset_path": "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/WFDD_seg_0shot.tsv",
        "output_root": "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/refactor-per-head-sinkfirst/WFDD_seg_0shot",
        "replace_path": "/home/yizhou/LVLM/dataset/TEST_DATASET/wfdd/",
    },
    "Liver_AD": {
        "dataset_path": "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/yz/READL/eval_dataset/Liver_AD.tsv",
        "output_root": "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/refactor-per-head-sinkfirst/Liver_AD_seg_0shot",
        "replace_path": "/share/home/yizhou_lustre/dataset/medical_anomaly/Liver_AD/",
    },
    "BraTS2021": {
        "dataset_path": "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/yz/READL/eval_dataset/BraTS2021.tsv",
        "output_root": "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/refactor-per-head-sinkfirst/BraTS2021_seg_0shot",
        "replace_path": "/share/home/yizhou_lustre/dataset/medical_anomaly/BraTS2021_slice/",
    },
}


def apply_dataset_defaults(args, mode: str):
    if args.dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    cfg = DATASET_DEFAULTS[args.dataset]
    args.dataset_path = cfg["dataset_path"]
    if mode == "generator":
        args.save_dir = cfg["output_root"]
        args.replace_path = cfg["replace_path"]
    elif mode == "evaluator":
        args.generated_dir = cfg["output_root"]
        args.replace_path = cfg["replace_path"]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return args

