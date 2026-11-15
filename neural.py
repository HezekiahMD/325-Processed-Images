import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class DogBreedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[tuple[Path, int]],
        image_transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.records[index]
        with Image.open(image_path) as image:
            image_tensor = cast(
                torch.Tensor, self.image_transform(image.convert("RGB"))
            )
        return image_tensor, torch.tensor(label, dtype=torch.long)


def load_records(
    labels_path: Path, train_dir: Path
) -> tuple[list[tuple[Path, int]], list[str]]:
    with labels_path.open(newline="", encoding="utf-8-sig") as labels_file:
        reader = csv.DictReader(labels_file)
        if not {"id", "breed"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{labels_path} must have 'id' and 'breed' columns")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No labels found in {labels_path}")

    class_names = sorted({row["breed"] for row in rows if row["breed"]})
    if not class_names:
        raise ValueError(f"No breed names found in {labels_path}")
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    records: list[tuple[Path, int]] = []
    seen_ids: set[str] = set()
    missing_images: list[Path] = []
    for row in rows:
        image_id = row["id"].strip()
        breed = row["breed"].strip()
        if not image_id or not breed:
            raise ValueError(f"Empty image id or breed in {labels_path}")
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image id {image_id!r} in {labels_path}")
        seen_ids.add(image_id)

        image_path = train_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            missing_images.append(image_path)
            continue
        records.append((image_path, class_to_idx[breed]))

    if missing_images:
        examples = ", ".join(str(path) for path in missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images)} labeled images are missing from {train_dir}; "
            f"examples: {examples}"
        )
    return records, class_names


def stratified_split(
    records: list[tuple[Path, int]], validation_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")

    by_class: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for record in records:
        by_class[record[1]].append(record)

    rng = random.Random(seed)
    train_records: list[tuple[Path, int]] = []
    validation_records: list[tuple[Path, int]] = []
    for class_records in by_class.values():
        rng.shuffle(class_records)
        validation_count = min(
            round(len(class_records) * validation_fraction),
            max(0, len(class_records) - 1),
        )
        validation_records.extend(class_records[:validation_count])
        train_records.extend(class_records[validation_count:])

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    if not validation_records:
        raise ValueError("Not enough labeled images to create a validation split")
    return train_records, validation_records


def build_model(class_count: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a ResNet-18 dog-breed classifier from train/ and labels.csv."
    )
    parser.add_argument("--data-dir", type=Path, default=script_dir)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start from ImageNet weights (downloads them if they are not cached).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "dog_breed_resnet18.pth",
        help="Path for the best validation checkpoint.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        parser.error("--epochs, --batch-size, and --image-size must be positive")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_path = args.data_dir / "labels.csv"
    train_dir = args.data_dir / "train"
    records, class_names = load_records(labels_path, train_dir)
    train_records, validation_records = stratified_split(
        records, args.validation_fraction, args.seed
    )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size + 32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        DogBreedDataset(train_records, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        DogBreedDataset(validation_records, validation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    model = build_model(len(class_names), args.pretrained).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_accuracy = -1.0

    print(
        f"Device: {device}; images: {len(records)}; breeds: {len(class_names)}; "
        f"train: {len(train_records)}; validation: {len(validation_records)}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, loss_function, device
        )
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train loss={train_loss:.4f} acc={train_accuracy:.4f} "
            f"val loss={validation_loss:.4f} acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            checkpoint: dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "architecture": "resnet18",
                "validation_accuracy": validation_accuracy,
            }
            torch.save(checkpoint, args.output)

    print(
        f"Best validation accuracy: {best_validation_accuracy:.4f}; "
        f"checkpoint saved to {args.output}"
    )


if __name__ == "__main__":
    main()
