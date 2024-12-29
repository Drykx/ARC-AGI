import os
import json
import sys
import re
import random
import importlib.util
from typing import *
from tqdm import tqdm 
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# To plot Tasks
def plot_arc_train_data(train_data: List[dict], id: str = None, label: str = None) -> None:
    """
    Plots input-output pairs from the 'train' data in horizontal layout.
    """
    cmap = ListedColormap([
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    args = {'cmap': cmap, 'vmin': 0, 'vmax': 10}
    
    # Number of examples
    n_examples = len(train_data)
    fig_width = n_examples * 3  # Scale for horizontal spacing
    fig_height = 3
    
    # Create a figure with 2 rows (input-output) and n_examples columns
    fig, axes = plt.subplots(2, n_examples, figsize=(fig_width, fig_height))
    
    # Ensure axes is always a 2D array
    if n_examples == 1:
        axes = axes[:, np.newaxis]
    
    for col, example in enumerate(train_data):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Plot input
        axes[0, col].imshow(input_grid, **args)
        axes[0, col].axis('off')
        
        # Plot output
        axes[1, col].imshow(output_grid, **args)
        axes[1, col].axis('off')
    
    fig.suptitle(f"Task: {id} and perceptions: {label}")
    plt.tight_layout()
    plt.show()

### Functions to create Dictionnary
## General
def create_arc_training_tasks(directory_path, list_labels):
    """
    Create a dictionary of training tasks by reading JSON files from a directory and associating labels.
    
    Args:
        directory_path (str): Path to the directory containing JSON files.
        list_labels (list): List of labels to associate with each task.
    
    Returns:
        dict: A dictionary of training tasks.
    """
    data = {}
    filenames = sorted([f for f in os.listdir(directory_path) if f.endswith(".json")])
    
    # Iterate through the filenames with their index using enumerate
    for i, filename in enumerate(filenames):
        file_path = os.path.join(directory_path, filename)

        try:
            with open(file_path, "r") as f:
                task_data = json.load(f)

            # Ensure there is a corresponding label for each task
            if i < len(list_labels):
                name = os.path.splitext(filename)[0]  # Use the filename without extension
                add_entry(data, name, list_labels[i], task_data)
            else:
                print(f"Warning: No label found for file '{filename}'")

        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from file '{filename}'")
    
    return data

# Helper
def add_entry(dictionary, name, perceptions, example):
    """
    Add an entry to the dictionary of training tasks.

    Args:
        dictionary (dict): The dictionary to add the entry to.
        name (str): The task name.
        perceptions (any): The associated label or perception.
        example (dict): The task data.
    """
    dictionary[name] = {
        "name": name,
        "perceptions": perceptions,
        "example": example
    }
# Helper

# Flatten the list, including nested lists
def flatten(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten(item))  # Recursively flatten nested lists
        else:
            flattened.append(item)
    return flattened

### ConceptARC

def concept_arc(directory_path):
    problems = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Make sure it's a directory before proceeding
        if os.path.isdir(file_path):
            subproblems = [str(filename)]
            print(f"Processing: {file_path}")

            for examples in os.listdir(file_path):
                if examples.endswith(".json"):
                    full_path = os.path.join(file_path, examples)

                    with open(full_path, "r") as f:
                        problem_data = json.load(f)
                        subproblems.append(problem_data)  # Append problem_data here

            problems.append(subproblems)  # Append subproblems to problems

    return problems






##################################

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


