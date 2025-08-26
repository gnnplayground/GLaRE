import argparse
import numpy as np
import pickle
from tqdm import trange
from dataset import AffectNetMultiLabelDataset, default_transform
import face_alignment

predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='mps')


def visualize_sample(dataset):
    """Visualize one random sample with labels."""
    import matplotlib.pyplot as plt
    import random

    idx = random.randint(0, len(dataset) - 1)
    image, label = dataset[idx]

    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.title(f"Exp: {label['exp'].item()}, Aro: {label['aro'].item():.2f}, "
              f"Val: {label['val'].item():.2f}")
    plt.axis('off')
    plt.show()


def extract_landmarks(dataset, predictor, output_path="landmarks_labels.pkl"):
    """Extract landmarks using predictor and save to pickle file."""
    landmarks_list, labels_list = [], []

    for i in trange(len(dataset), desc="Processing dataset"):
        try:
            image, label_dict = dataset[i]
            lands = predictor.get_landmarks(np.asarray(image.permute(1, 2, 0)))  # tensor → numpy
            label = label_dict['exp']

            landmarks_list.append(lands)
            labels_list.append(label)
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            continue

    data = {'landmarks': landmarks_list, 'labels': labels_list}

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Saved landmarks and labels to {output_path}")


def main(args):
    # Load dataset
    dataset = AffectNetMultiLabelDataset(
        image_folder=args.image_folder,
        annotation_folder=args.annotation_folder,
        transform=default_transform
    )

    if args.mode == "visualize":
        visualize_sample(dataset)

    elif args.mode == "extract":
        if "predictor" not in globals():
            raise RuntimeError("❌ predictor is not defined. Please define it before running.")
        extract_landmarks(dataset, predictor, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AffectNet pipeline runner")
    parser.add_argument("--image_folder", type=str, default="affectnet/images",
                        help="Path to AffectNet images folder")
    parser.add_argument("--annotation_folder", type=str, default="affectnet/annotations",
                        help="Path to AffectNet annotations folder")
    parser.add_argument("--mode", type=str, choices=["visualize", "extract"], default="visualize",
                        help="Pipeline mode: visualize sample or extract landmarks")
    parser.add_argument("--output", type=str, default="landmarks_labels.pkl",
                        help="Path to save landmarks/labels pickle")

    args = parser.parse_args()
    main(args)
