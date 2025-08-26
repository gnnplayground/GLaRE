import argparse
import random
import numpy as np
import pickle
import torch
from tqdm import trange
import matplotlib.pyplot as plt

from dataset import AffectNetMultiLabelDataset, default_transform
import face_alignment

from pipeline import load_pickle_data, build_graphs, run_training_pipeline, get_available_device

# Initialize face alignment predictor
predictor = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.THREE_D, flip_input=False, device='mps'
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------- Visualization / Landmark Extraction -------------------

def visualize_sample(dataset):
    """Visualize one random sample with labels."""
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


# ------------------- Training Pipeline -------------------

def train_pipeline(args):
    # Load pickle data (must include cnn_features)
    landmarks_list, labels_list, features_list, feat_dim = load_pickle_data(args.data_pkl)

    # Build PyG graphs
    graphs = build_graphs(
        landmarks_list=landmarks_list,
        labels_list=labels_list,
        features_list=features_list,
        k=args.k
    )

    if len(graphs) < 10:
        print("⚠️ Warning: very few graphs after filtering; training quality may be poor.")

    # Run training and evaluation
    model, test_acc = run_training_pipeline(
        graphs=graphs,
        feat_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        num_regions=args.num_regions,
        num_classes=args.num_classes,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f"\n✅ Training done. Test Accuracy: {test_acc:.2f}%")


# ------------------- Main -------------------

def main(args):
    set_seed(args.seed)

    if args.mode == "visualize":
        dataset = AffectNetMultiLabelDataset(
            image_folder=args.image_folder,
            annotation_folder=args.annotation_folder,
            transform=default_transform
        )
        visualize_sample(dataset)

    elif args.mode == "extract":
        dataset = AffectNetMultiLabelDataset(
            image_folder=args.image_folder,
            annotation_folder=args.annotation_folder,
            transform=default_transform
        )
        extract_landmarks(dataset, predictor, args.output)

    elif args.mode == "train":
        train_pipeline(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AffectNet pipeline runner")

    # General paths
    parser.add_argument("--image_folder", type=str, default="affectnet/images",
                        help="Path to AffectNet images folder")
    parser.add_argument("--annotation_folder", type=str, default="affectnet/annotations",
                        help="Path to AffectNet annotations folder")
    parser.add_argument("--data_pkl", type=str, default=None,
                        help="Path to pickle containing landmarks, labels, cnn_features (for training)")

    parser.add_argument("--mode", type=str, choices=["visualize", "extract", "train"], default="visualize",
                        help="Pipeline mode: visualize, extract landmarks, or train GNN")

    parser.add_argument("--output", type=str, default="landmarks_labels.pkl",
                        help="Path to save landmarks/labels pickle (used in extract mode)")

    # Training args
    parser.add_argument("--k", type=int, default=3, help="k for k-NN when building graphs")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_regions", type=int, default=8, help="Number of KMeans regions for quotient graph")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of classes")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
