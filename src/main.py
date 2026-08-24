"""
CLI entry point: image(s) in -> class prediction + gradient-weighted
attention rollout overlay(s) out.

Single image:
    python src/main.py --image images/dog.jpg --output outputs/dog_overlay.png

Whole folder of images:
    python src/main.py --image-dir images --output-dir outputs

Run `python src/main.py --help` for all options.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from model import (
    load_vit,
    get_transform,
    load_image,
    preprocess_image,
    get_imagenet_labels,
    predict,
)
from rollout import (
    register_attention_hooks,
    remove_hooks,
    collect_gradients,
    class_relevance_map,
)
from visualize import build_overlay, save_image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(description="ViT gradient-weighted attention rollout")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a single input image")
    source.add_argument("--image-dir", help="Path to a folder of images to process all at once")

    parser.add_argument("--output", default="outputs/overlay.png",
                         help="Output path when using --image")
    parser.add_argument("--output-dir", default="outputs",
                         help="Output folder when using --image-dir")

    parser.add_argument("--model", default="vit_base_patch16_224", help="timm ViT model name")
    parser.add_argument("--target-class", type=int, default=None,
                         help="ImageNet class index to explain (default: the model's own top prediction)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay opacity")
    parser.add_argument("--plain-rollout", action="store_true",
                         help="Use plain (non-gradient-weighted) attention rollout instead")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_pipeline(model_name="vit_base_patch16_224", device="cpu"):
    """
    Load everything that's shared across images once: the model, its
    preprocessing transform, and the ImageNet label names. Reused across
    every image in a batch run instead of reloading per image.
    """
    model = load_vit(model_name=model_name, device=device)
    transform = get_transform(model)
    labels = get_imagenet_labels()
    return model, transform, labels


def run_single(model, transform, labels, image_path, output_path,
                target_class=None, alpha=0.5, use_gradient_weighting=True, device="cpu"):
    """
    Run the full pipeline on one image using an already-loaded model.
    Returns (predicted_class_index, predicted_label, confidence, output_path).
    """
    image = load_image(image_path)
    input_tensor = preprocess_image(image, transform, device=device)

    handles, attention_maps = register_attention_hooks(model)

    logits, top_indices, top_probs = predict(model, input_tensor, top_k=1)
    predicted_index = int(top_indices[0].item())
    predicted_confidence = float(top_probs[0].item())

    explain_index = target_class if target_class is not None else predicted_index

    model.zero_grad()
    logits[0, explain_index].backward()

    gradients = collect_gradients(attention_maps) if use_gradient_weighting else None
    relevance = class_relevance_map(attention_maps, gradients=gradients)

    remove_hooks(handles)

    overlay = build_overlay(image, relevance[0], alpha=alpha)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image(overlay, output_path)

    return predicted_index, labels[predicted_index], predicted_confidence, output_path


def list_image_files(directory):
    """All image files directly inside `directory`, sorted for stable ordering."""
    names = sorted(os.listdir(directory))
    return [
        os.path.join(directory, name)
        for name in names
        if name.lower().endswith(IMAGE_EXTENSIONS)
    ]


def output_path_for(image_path, output_dir):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, f"{base_name}_overlay.png")


def run_batch(model, transform, labels, image_dir, output_dir,
              target_class=None, alpha=0.5, use_gradient_weighting=True, device="cpu"):
    """
    Run run_single() over every image file in image_dir. A failure on one
    image is reported and skipped rather than stopping the whole batch.
    Returns a list of result rows (one per successfully processed image).
    """
    image_paths = list_image_files(image_dir)
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_path_for(image_path, output_dir)
        try:
            predicted_index, label, confidence, saved_path = run_single(
                model, transform, labels, image_path, output_path,
                target_class=target_class, alpha=alpha,
                use_gradient_weighting=use_gradient_weighting, device=device,
            )
            print(f"[{index}/{len(image_paths)}] {os.path.basename(image_path)} "
                  f"-> {label} ({confidence:.4f}) -> {saved_path}")
            results.append((image_path, predicted_index, label, confidence, saved_path))
        except Exception as error:
            print(f"[{index}/{len(image_paths)}] {os.path.basename(image_path)} -> FAILED: {error}")

    return results


def main():
    args = parse_args()

    model, transform, labels = load_pipeline(model_name=args.model, device=args.device)
    use_gradient_weighting = not args.plain_rollout

    if args.image_dir:
        results = run_batch(
            model, transform, labels,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            target_class=args.target_class,
            alpha=args.alpha,
            use_gradient_weighting=use_gradient_weighting,
            device=args.device,
        )
        print(f"\nDone: {len(results)} overlays saved to {args.output_dir}/")
    else:
        predicted_index, label, confidence, output_path = run_single(
            model, transform, labels,
            image_path=args.image,
            output_path=args.output,
            target_class=args.target_class,
            alpha=args.alpha,
            use_gradient_weighting=use_gradient_weighting,
            device=args.device,
        )
        print(f"Predicted class: {label} (index {predicted_index}), confidence {confidence:.4f}")
        print(f"Overlay saved to: {output_path}")


if __name__ == "__main__":
    main()
