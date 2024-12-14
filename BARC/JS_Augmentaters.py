
########################################################
### Augmentaters
########################################################

import numpy as np
import random

class Augmenter:
    """Base class for augmenters."""

    def __call__(self, grid):
        """Make the augmenter callable."""
        return self.apply_to_grid(grid)

    def apply_to_grid(self, grid):
        """Method to be overridden by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class Rotate(Augmenter):
    """Rotate the grid by 90, 180, or 270 degrees."""

    def __init__(self, angle):
        assert angle in [0, 90, 180, 270], "Angle must be 0, 90, 180, or 270 degrees."
        self.angle = angle

    def __str__(self):
        return f"Rotate {self.angle}°"

    def apply_to_grid(self, grid):
        # Convert the list to a NumPy array
        array = np.array(grid)
        if self.angle == 90:
            rotated = np.rot90(array, k=1)
        elif self.angle == 180:
            rotated = np.rot90(array, k=2)
        elif self.angle == 270:
            rotated = np.rot90(array, k=3)
        else:
            rotated = array  # 0 degrees means no rotation

        return rotated

class Flip(Augmenter):
    """Flip the grid along the specified axis (0 for vertical, 1 for horizontal)."""

    def __init__(self, axis):
        assert axis in [0, 1], "Axis must be 0 (vertical) or 1 (horizontal)."
        self.axis = axis

    def __str__(self):
        return f"Flip {'Vertical' if self.axis == 0 else 'Horizontal'}"

    def apply_to_grid(self, grid):
        # Convert the list to a NumPy array
        array = np.array(grid)
        if self.axis == 0:
            flipped = np.flipud(array)  # Vertical flip
        else:
            flipped = np.fliplr(array)  # Horizontal flip
        # Convert back to a list

        return flipped

class PermuteColors(Augmenter):
    """Permute colors in the grid by randomly swapping values."""

    def __init__(self):
        pass

    def __str__(self):
        return "Permute Colors"

    def apply_to_grid(self, grid):
        # Convert the list to a NumPy array
        array = np.array(grid)
        unique_colors = np.unique(array)
        color_map = {color: random.choice(unique_colors) for color in unique_colors}
        color_mapper = np.vectorize(lambda x: color_map[x])
        # Apply the color mapping

        return color_mapper(array)

class Dropout(Augmenter):
    """Dropout a random rectangular patch by setting it to 0."""

    def __init__(self, dropout_value=0):
        self.dropout_value = dropout_value

    def __str__(self):
        return "Dropout"

    def apply_to_grid(self, grid):
        # Convert the list to a NumPy array
        array = np.array(grid)
        rows, cols = array.shape
        x_start = random.randint(0, rows - 1)
        y_start = random.randint(0, cols - 1)
        x_end = random.randint(x_start, rows)
        y_end = random.randint(y_start, cols)
        array[x_start:x_end, y_start:y_end] = self.dropout_value
        # Convert back to a list
        return array
