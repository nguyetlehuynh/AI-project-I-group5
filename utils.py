import matplotlib.pyplot as plt
import matplotlib.patches as patches

def resize_box_xyxy(box, old_w, old_h, new_w, new_h):
    """
    Rescales bounding box coordinates (x1, y1, x2, y2) based on image resizing.
    This ensures that the labels match the new image dimensions.
    """
    x1, y1, x2, y2 = box

    # Calculate scale factors for both axes
    scale_x = new_w / old_w
    scale_y = new_h / old_h

    # Apply scaling to coordinates
    x1 *= scale_x
    y1 *= scale_y
    x2 *= scale_x
    y2 *= scale_y

    return x1, y1, x2, y2


def show_batch(images, targets):
    """
    Visualizes a batch of images with their corresponding bounding boxes and labels.
    Useful for verifying data loading and preprocessing.
    """
    for i in range(len(images)):
        # Convert tensor to numpy and change format from (C, H, W) to (H, W, C) for plotting
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        boxes = targets[i]["boxes"].detach().cpu().numpy()
        labels = targets[i]["labels"].detach().cpu().numpy()

        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Define a rectangle patch for the bounding box
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Display the class label above the box
            ax.text(
                x1,
                y1 - 5,
                f"class {label}",
                fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )

        ax.set_title(f"Sample {i + 1} in batch")
        ax.axis("off")  # Hide axes for a cleaner look
        plt.show()