import matplotlib.pyplot as plt
import torch


def show_image(img, label=None, classes=None, denorm=True):
    """
    Display a single image tensor (C, H, W) with optional label.

    Args:
        img (Tensor): Image tensor (C x H x W)
        label (int or str, optional): Class index or name
        classes (list, optional): List of class names if label is index
        denorm (bool): Whether to unnormalize from [-1,1] to [0,1]
    """
    # Move to CPU and detach from graph if needed
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()

    # Convert from (C, H, W) → (H, W, C)
    img = img.permute(1, 2, 0)

    # Undo normalization
    if denorm:
        img = img * 0.5 + 0.5  # for mean=0.5, std=0.5 normalization
    img = img.clamp(0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")

    if label is not None:
        if classes is not None and isinstance(label, (int, torch.Tensor)):
            title = classes[int(label)]
        else:
            title = str(label)
        plt.title(title)

    plt.show()


if __name__ == "__main__":
    # Example usage
    dummy_img = torch.randn(3, 64, 64)  # Random image tensor
    show_image(dummy_img)
