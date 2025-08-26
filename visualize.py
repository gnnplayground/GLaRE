import matplotlib.pyplot as plt
from dataset import AffectNetMultiLabelDataset, default_transform

# Example usage
dataset = AffectNetMultiLabelDataset(
    image_folder="affectnet/images",
    annotation_folder="affectnet/annotations",
    transform=default_transform
)

# Visualize one sample
image, label = dataset[0]
plt.imshow(image.permute(1, 2, 0), cmap='gray')  # convert tensor to HWC
plt.title(f"Exp: {label['exp'].item()}, Aro: {label['aro'].item():.2f}, "
          f"Val: {label['val'].item():.2f}")
plt.axis('off')
plt.show()
