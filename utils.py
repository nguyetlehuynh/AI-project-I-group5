def resize_box_xyxy(box, old_w, old_h, new_w, new_h):
    """
    Rescales bounding box coordinates from the original image size to the new image size.
    This ensures labels remain correctly aligned after image resizing.
    """
    # Unpack the original coordinates (x1, y1 for top-left; x2, y2 for bottom-right)
    x1, y1, x2, y2 = box

    # Calculate the scaling ratios for width and height
    scale_x = new_w / old_w
    scale_y = new_h / old_h

    # Apply the ratios to update each coordinate
    x1 *= scale_x
    y1 *= scale_y
    x2 *= scale_x
    y2 *= scale_y

    # Return the newly scaled coordinates
    return x1, y1, x2, y2