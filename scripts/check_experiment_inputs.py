from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


EXPECTED_CLASSES = ["COVID", "Lung_Opacity", "Viral_Pneumonia", "Normal"]
EXPECTED_LABELS = {
    "COVID": "0",
    "Lung_Opacity": "1",
    "Viral_Pneumonia": "2",
    "Normal": "3",
}
REQUIRED_SUPERVISED_COLUMNS = {"image_path", "class_name", "label", "source"}
REQUIRED_SYNTHETIC_COLUMNS = {"image_path", "class_name", "label", "source", "generator"}
EXPECTED_EXPERIMENT_FILES = {
    "resnet18": {
        "none.yaml",
        "imagenet.yaml",
        "covidqu.yaml",
        "imagenet_covidqu.yaml",
        "covidqu_syn.yaml",
        "imagenet_covidqu_syn.yaml",
    },
    "vit_s16": {
        "none.yaml",
        "imagenet.yaml",
        "covidqu.yaml",
        "imagenet_covidqu.yaml",
        "covidqu_syn.yaml",
        "imagenet_covidqu_syn.yaml",
    },
}


@dataclass
class CheckResult:
    name: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARNING"
        return "PASS"


def parse_scalar(value: str):
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def simple_yaml_load(text: str):
    """Parse the small YAML subset used by configs/experiments."""
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        clean = strip_yaml_comment(raw_line).rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        lines.append((indent, clean.strip()))

    def parse_block(index: int, indent: int):
        if index >= len(lines):
            return {}, index

        is_list = lines[index][0] == indent and lines[index][1].startswith("- ")
        if is_list:
            values = []
            while index < len(lines):
                current_indent, content = lines[index]
                if current_indent != indent or not content.startswith("- "):
                    break
                values.append(parse_scalar(content[2:].strip()))
                index += 1
            return values, index

        mapping = {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation near: {content}")
            if content.startswith("- "):
                break
            if ":" not in content:
                raise ValueError(f"Expected key/value line, got: {content}")

            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1

            if value:
                mapping[key] = parse_scalar(value)
            elif index < len(lines) and lines[index][0] > current_indent:
                mapping[key], index = parse_block(index, lines[index][0])
            else:
                mapping[key] = None

        return mapping, index

    data, index = parse_block(0, 0)
    if index != len(lines):
        raise ValueError("Could not parse complete YAML document")
    return data


def read_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = simple_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames


def validate_manifest(path: Path, required_columns: set[str]) -> tuple[list[str], dict[str, int]]:
    failures: list[str] = []
    rows, columns = read_csv_rows(path)

    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        failures.append(f"{path} missing required columns: {missing_columns}")
        return failures, {}

    if not rows:
        failures.append(f"{path} has no rows")
        return failures, {}

    class_names = sorted({row["class_name"] for row in rows})
    if class_names != sorted(EXPECTED_CLASSES):
        failures.append(f"{path} class names mismatch: got {class_names}, expected {EXPECTED_CLASSES}")

    bad_labels = sorted(
        {
            (row["class_name"], row["label"])
            for row in rows
            if row["class_name"] in EXPECTED_LABELS and row["label"] != EXPECTED_LABELS[row["class_name"]]
        }
    )
    if bad_labels:
        failures.append(f"{path} has invalid class/label pairs: {bad_labels[:8]}")

    counts = {class_name: 0 for class_name in EXPECTED_CLASSES}
    for row in rows:
        if row["class_name"] in counts:
            counts[row["class_name"]] += 1

    return failures, counts


def validate_supervised_manifests(manifest_dir: Path) -> CheckResult:
    result = CheckResult("fixed supervised manifests")
    for split in ["train", "val", "test"]:
        path = manifest_dir / f"{split}.csv"
        if not path.exists():
            result.failures.append(f"missing required manifest: {path}")
            continue
        failures, counts = validate_manifest(path, REQUIRED_SUPERVISED_COLUMNS)
        result.failures.extend(failures)
        if counts:
            result.notes.append(f"{split}: {counts} total={sum(counts.values())}")
    return result


def validate_expected_config_files(config_root: Path) -> CheckResult:
    result = CheckResult("experiment config files")
    for backbone, expected_files in EXPECTED_EXPERIMENT_FILES.items():
        backbone_dir = config_root / backbone
        if not backbone_dir.exists():
            result.failures.append(f"missing config directory: {backbone_dir}")
            continue
        present_files = {path.name for path in backbone_dir.glob("*.yaml")}
        missing = sorted(expected_files - present_files)
        extra = sorted(present_files - expected_files)
        if missing:
            result.failures.append(f"{backbone_dir} missing configs: {missing}")
        if extra:
            result.warnings.append(f"{backbone_dir} has extra configs: {extra}")
    return result


def validate_config_structure(config: dict, path: Path) -> list[str]:
    failures: list[str] = []
    required_top_level = [
        "experiment_id",
        "backbone",
        "pretraining_strategy",
        "contrastive_method",
        "pretrain",
        "finetune",
        "output_dir",
    ]
    for key in required_top_level:
        if key not in config:
            failures.append(f"{path} missing top-level key: {key}")

    if not isinstance(config.get("pretrain"), dict):
        failures.append(f"{path} pretrain must be a mapping")
    if not isinstance(config.get("finetune"), dict):
        failures.append(f"{path} finetune must be a mapping")

    return failures


def merged_common_data(common_config: dict) -> dict:
    data = common_config.get("data", {})
    return data if isinstance(data, dict) else {}


def validate_experiment_config(
    path: Path,
    common_data: dict,
    real_unlabeled_dir: Path,
    synthetic_manifest: Path,
) -> CheckResult:
    config = read_yaml(path)
    experiment_id = str(config.get("experiment_id", path.stem))
    result = CheckResult(experiment_id)

    structure_failures = validate_config_structure(config, path)
    result.failures.extend(structure_failures)
    if structure_failures:
        return result

    backbone = config["backbone"]
    strategy = config["pretraining_strategy"]
    method = config["contrastive_method"]
    pretrain = config["pretrain"]
    finetune = config["finetune"]
    output_dir = Path(str(config["output_dir"]))

    num_classes = common_data.get("num_classes")
    image_size = common_data.get("image_size")
    class_names = common_data.get("class_names")

    if num_classes != 4:
        result.failures.append(f"common data.num_classes must be 4, got {num_classes}")
    if image_size != 224:
        result.failures.append(f"common data.image_size must be 224, got {image_size}")
    if class_names != EXPECTED_CLASSES:
        result.failures.append(f"common data.class_names mismatch: got {class_names}, expected {EXPECTED_CLASSES}")

    if backbone == "resnet18" and pretrain.get("enabled") and method != "simclr":
        result.failures.append("ResNet18 contrastive pretraining must use simclr")
    if backbone == "vit_s16" and pretrain.get("enabled") and method != "dino":
        result.failures.append("ViT-S/16 contrastive pretraining must use dino")

    if strategy in {"none", "imagenet"}:
        if pretrain.get("enabled"):
            result.failures.append(f"{strategy} experiment should not enable pretraining")
        if pretrain.get("dataset") not in {None, "none"}:
            result.failures.append(f"{strategy} experiment should not require pretrain data")
        if method != "none":
            result.failures.append(f"{strategy} experiment should use contrastive_method: none")
    else:
        if not pretrain.get("enabled"):
            result.failures.append(f"{strategy} experiment should enable contrastive pretraining")

    if "covidqu_syn" in strategy:
        if pretrain.get("dataset") != "synthetic":
            result.failures.append("COVID-QU-Syn experiment must use pretrain.dataset: synthetic")
        if not synthetic_manifest.exists():
            result.warnings.append(
                f"missing {synthetic_manifest}; required before running COVID-QU-Syn pretraining. "
                "Generate it on Colab from the DCGAN Stage 1 output."
            )
        else:
            failures, counts = validate_manifest(synthetic_manifest, REQUIRED_SYNTHETIC_COLUMNS)
            result.failures.extend(failures)
            if counts:
                result.notes.append(f"synthetic_dcgan: {counts} total={sum(counts.values())}")
    elif "covidqu" in strategy:
        if pretrain.get("dataset") != "real_unlabeled":
            result.failures.append("COVID-QU experiment must use pretrain.dataset: real_unlabeled")
        if not real_unlabeled_dir.exists():
            result.failures.append(f"missing real unlabeled dataset: {real_unlabeled_dir}")
    else:
        result.notes.append("no contrastive pretraining data required")

    expected_output_dir = Path("results/experiments") / experiment_id
    if output_dir != expected_output_dir:
        result.failures.append(f"output_dir mismatch: got {output_dir}, expected {expected_output_dir}")
    else:
        result.notes.append(f"planned output_dir: {output_dir}")

    if finetune.get("freeze_backbone") is not False:
        result.warnings.append("finetune.freeze_backbone is not false; report setup expects full fine-tuning")

    return result


def print_result(result: CheckResult) -> None:
    print(f"[{result.status}] {result.name}")
    for failure in result.failures:
        print(f"  FAIL: {failure}")
    for warning in result.warnings:
        print(f"  WARNING: {warning}")
    for note in result.notes:
        print(f"  - {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify experiment configs and data inputs before training.")
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--real-unlabeled-dir", type=Path, default=Path("data/processed/unlabelled_16934"))
    parser.add_argument("--synthetic-manifest", type=Path, default=Path("data/manifests/synthetic_dcgan.csv"))
    args = parser.parse_args()

    results: list[CheckResult] = []

    common_path = args.config_root / "common.yaml"
    if not common_path.exists():
        results.append(CheckResult("common config", failures=[f"missing required config: {common_path}"]))
        common_data = {}
    else:
        try:
            common_config = read_yaml(common_path)
            common_data = merged_common_data(common_config)
            results.append(CheckResult("common config", notes=[f"loaded {common_path}"]))
        except Exception as exc:
            results.append(CheckResult("common config", failures=[str(exc)]))
            common_data = {}

    results.append(validate_supervised_manifests(args.manifest_dir))
    results.append(validate_expected_config_files(args.config_root))

    config_paths = []
    for backbone in EXPECTED_EXPERIMENT_FILES:
        config_paths.extend(sorted((args.config_root / backbone).glob("*.yaml")))

    if len(config_paths) != 12:
        results.append(
            CheckResult(
                "experiment config count",
                failures=[f"expected 12 experiment configs, found {len(config_paths)}"],
            )
        )

    for path in config_paths:
        try:
            results.append(
                validate_experiment_config(
                    path=path,
                    common_data=common_data,
                    real_unlabeled_dir=args.real_unlabeled_dir,
                    synthetic_manifest=args.synthetic_manifest,
                )
            )
        except Exception as exc:
            results.append(CheckResult(path.name, failures=[str(exc)]))

    print("\nExperiment Input Check Report")
    print("=============================")
    for result in results:
        print_result(result)

    pass_count = sum(1 for result in results if result.status == "PASS")
    warning_count = sum(1 for result in results if result.status == "WARNING")
    fail_count = sum(1 for result in results if result.status == "FAIL")

    print("\nSummary")
    print("=======")
    print(f"PASS: {pass_count}")
    print(f"WARNING: {warning_count}")
    print(f"FAIL: {fail_count}")

    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
