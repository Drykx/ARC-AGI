import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Color:
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    GRAY = "gray"
    RED = "red"
    PINK = "pink"
    TEAL = "teal"
    MAROON = "maroon"
    BLACK = "black"


def plot_arc_input_outputs(input_outputs, column_headings=None, figsize_multiplier=2, show_grid=False, title_fontsize=8):
    """
    A more compact version of the plot function.
    
    Parameters:
    - input_outputs: List of lists where each element is either a 2D numpy array or (2D_array, mask).
    - column_headings: Optional list of column headings.
    - figsize_multiplier: Scale down the figure size. Default 2 means smaller than original 5.
    - show_grid: Whether to show grid lines. Default False for a cleaner, smaller plot.
    - title_fontsize: Font size for titles.
    """
    column_headings = column_headings or ["input", "output"]
    n_pairs = len(input_outputs)
    n_cols = len(input_outputs[0])

    # Create a smaller figure
    figure, axs = plt.subplots(n_pairs, n_cols, figsize=(figsize_multiplier * n_cols, figsize_multiplier * n_pairs), dpi=100)

    # Ensure axs is always 2D
    if n_pairs == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_pairs == 1:
        axs = axs[np.newaxis, :]
    elif n_cols == 1:
        axs = axs[:, np.newaxis]

    # Define colors
    colors_rgb = {
        0: (0x00, 0x00, 0x00),
        1: (0x00, 0x74, 0xD9),
        2: (0xFF, 0x41, 0x36),
        3: (0x2E, 0xCC, 0x40),
        4: (0xFF, 0xDC, 0x00),
        5: (0xA0, 0xA0, 0xA0),
        6: (0xF0, 0x12, 0xBE),
        7: (0xFF, 0x85, 0x1B),
        8: (0x7F, 0xDB, 0xFF),
        9: (0x87, 0x0C, 0x25),
        10: (0xFF, 0xFF, 0xFF)
    }

    _float_colors = [tuple(c / 255 for c in col) for col in colors_rgb.values()]
    arc_cmap = ListedColormap(_float_colors)

    for ex, input_output in enumerate(input_outputs):
        for col, grid in enumerate(input_output):
            ax = axs[ex, col]

            # Handle partial output case
            extra_title = ""
            if isinstance(grid, tuple) and len(grid) == 2 and isinstance(grid[0], np.ndarray):
                grid, mask = grid
                grid = grid.copy()
                if isinstance(mask, np.ndarray):
                    grid[~mask] = 10
                else:
                    grid[grid==mask] = 10
                extra_title = " partial output"
            elif not isinstance(grid, np.ndarray):
                # If it's not a numpy array or tuple as expected, skip
                continue

            grid = grid.T

            # Use pcolormesh to plot
            ax.pcolormesh(
                grid,
                cmap=arc_cmap,
                rasterized=True,
                vmin=0,
                vmax=10,
            )
            
            # Optionally show grid and ticks
            if show_grid:
                ax.set_xticks(np.arange(0, grid.shape[1]+1, 1))
                ax.set_yticks(np.arange(0, grid.shape[0]+1, 1))
                ax.grid(True, which='both')
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)

            ax.set_aspect('equal')
            ax.invert_yaxis()

            if col < len(column_headings):
                ax.set_title(column_headings[col] + extra_title, fontsize=title_fontsize)
            else:
                ax.set_title(extra_title, fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()


