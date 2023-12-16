import os
import matplotlib.pyplot as plt
import system_util

# Call logger for monitoring (in terminal)
logger = system_util.get_logger(__name__)

def get_image_label_pairs_version_dirfile(root_path):
    """
    Fetches image paths and their corresponding labels from a directory.

    This function reads the root directory specified by the `root_path`. Each sub-directory within the root directory
    represents a label, and the image files within each sub-directory are the corresponding samples for that label.
    It returns two lists: one containing paths to all images, and the other containing integer labels corresponding 
    to each image path.

    The function also creates two dictionaries to map between the label names (sub-directory names) and their 
    respective integer indices.

    Args:
        root_path (str): The path to the root directory containing labeled sub-directories.

    Returns:
        list[str]: A list containing paths to all images found within the sub-directories.
        list[int]: A list containing integer labels for each image path.

    Example:
        Let's say we have the following directory structure:
        root_path/
        ├── cat/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── dog/
            └── image3.jpg

        >>> images, labels = get_image_label_pair_version_dirfile("root_path")
        >>> print(images)
        ['root_path/cat/image1.jpg', 'root_path/cat/image2.jpg', 'root_path/dog/image3.jpg']
        >>> print(labels)
        [0, 0, 1]

    Notes:
        - This function assumes that there are no nested sub-directories within the label directories.
        - The function sorts label names in alphabetical order to assign integer indices.
    """
    images = []
    labels = []
    label_to_idx = {}
    idx_to_label = {}
    
    for idx, label in enumerate(sorted(os.listdir(root_path))):
        label_to_idx[label] = idx
        idx_to_label[idx] = label

        label_dir = os.path.join(root_path, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            images.append(image_path)
            labels.append(idx)
            
    return images, labels

def get_images_version_onlyfile(root_path):
    """
    Retrieves paths to all image files from the specified directory.

    This function reads the directory specified by `root_path` and constructs paths to all files within 
    this directory. It returns a list containing paths to all images, sorted in alphabetical order based on their 
    filenames.

    Args:
        root_path (str): The path to the directory containing image files.

    Returns:
        list[str]: A list containing paths to all images found within the specified directory.

    Example:
        Let's say we have the following directory structure:
        root_path/
        ├── image1.jpg
        ├── image2.jpg
        └── image3.jpg

        >>> images = get_images_version_onlyfile("root_path")
        >>> print(images)
        ['root_path/image1.jpg', 'root_path/image2.jpg', 'root_path/image3.jpg']

    Notes:
        - This function assumes that all files within the `root_path` directory are images.
        - Nested directories or sub-directories within the `root_path` are not explored.
    """
    images = [os.path.join(root_path, fname) for fname in sorted(os.listdir(root_path))]
    return images

def visualize_images_in_onebatch(dataloader):
    r, c    = [5, 5]
    k       = 0
    fig, ax = plt.subplots(r, c, figsize= (15, 15))
   
    for data in dataloader:
        x, y = data
        for i in range(r):
            for j in range(c):
                img = x[k].numpy().transpose(1, 2, 0)
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
                k+=1
        break

    del dataloader