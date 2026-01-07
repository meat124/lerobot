
"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from local storage.
It illustrates how to load datasets from a local directory, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Viewing a dataset's metadata and exploring its properties.
- Loading an existing dataset from local storage or a subset of it.
- Accessing frames by episode number.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pathlib import Path
from pprint import pprint

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def main():
    # Specify the local path to your dataset
    # The dataset should be in a directory structure like:
    #   dataset_root/
    #     repo_id/
    #       data/
    #       meta/
    #       videos/
    dataset_root = Path("datasets") 
    repo_id = "rby1_teleop_demo"
    
    # Full path to the dataset
    dataset_path = dataset_root / repo_id
    
    print(f"Loading dataset from local path: {dataset_path}")
    print(f"Dataset exists: {dataset_path.exists()}")
    
    # Verify that the dataset directory structure is correct
    meta_path = dataset_path / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found at {meta_path}. "
            f"Please ensure the dataset directory structure is correct:\n"
            f"  {dataset_path}/\n"
            f"    meta/\n"
            f"      info.json\n"
            f"    data/\n"
            f"    videos/\n"
        )
    print(f"Metadata file found: {meta_path}\n")
    
    # We can have a look and fetch its metadata to know more about it:
    # When loading from local, specify the root parameter with the full path to the dataset directory
    # The root should point to the directory containing meta/, data/, and videos/ folders
    # Note: The root parameter should be the full path to the dataset directory (dataset_root / repo_id)
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_path)

    # By instantiating just this class, you can quickly access useful information about the content and the
    # structure of the dataset without loading the actual data yet (only metadata files — which are
    # lightweight).
    print(f"Total number of episodes: {ds_meta.total_episodes}")
    print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
    print(f"Frames per second used during data collection: {ds_meta.fps}")
    print(f"Robot type: {ds_meta.robot_type}")
    print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

    print("Tasks:")
    print(ds_meta.tasks)
    print("Features:")
    pprint(ds_meta.features)

    # You can also get a short summary by simply printing the object:
    print(ds_meta)

    # You can then load the actual dataset from local storage.
    # Either load any subset of episodes:
    dataset = LeRobotDataset(repo_id, root=dataset_path, episodes=[0]) # 에피소드 개수 늘리면 수정

    # And see how many frames you have:
    print(f"Selected episodes: {dataset.episodes}")
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    # Or simply load the entire dataset:
    dataset = LeRobotDataset(repo_id, root=dataset_path)
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    # The previous metadata class is contained in the 'meta' attribute of the dataset:
    print(dataset.meta)

    # LeRobotDataset actually wraps an underlying Hugging Face dataset
    # (see https://huggingface.co/docs/datasets for more information).
    # print(dataset.hf_dataset)

    # LeRobot datasets also subclasses PyTorch datasets so you can do everything you know and love from working
    # with the latter, like iterating through the dataset.
    # The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by
    # episodes, you can access the frame indices of any episode using dataset.meta.episodes. Here, we access
    # frame indices associated to the first episode:
    episode_index = 0
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

    # Then we grab all the image frames from the first camera:
    if len(dataset.meta.camera_keys) > 0:
        camera_key = dataset.meta.camera_keys[0]
        frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

        # The objects returned by the dataset are all torch.Tensors
        print(type(frames[0]))
        print(frames[0].shape)

        # Since we're using pytorch, the shape is in pytorch, channel-first convention (c, h, w).
        # We can compare this shape with the information available for that feature
        pprint(dataset.features[camera_key])
        # In particular:
        print(dataset.features[camera_key]["shape"])
        # The shape is in (h, w, c) which is a more universal format.

        # For many machine learning applications we need to load the history of past observations or trajectories of
        # future actions. Our datasets can load previous and future frames for each key/modality, using timestamps
        # differences with the current loaded frame. For instance:
        delta_timestamps = {
            # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
            camera_key: [-1, -0.5, -0.20, 0],
            # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
            "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
            # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
            "action": [t / dataset.fps for t in range(64)],
        }
        # Note that in any case, these delta_timestamps values need to be multiples of (1/fps) so that added to any
        # timestamp, you still get a valid timestamp.

        dataset = LeRobotDataset(repo_id, root=dataset_path, delta_timestamps=delta_timestamps)
        print(f"\n{dataset[0][camera_key].shape=}")  # (4, c, h, w)
        print(f"{dataset[0]['observation.state'].shape=}")  # (6, c)
        print(f"{dataset[0]['action'].shape=}\n")  # (64, c)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=32,
            shuffle=True,
        )
        for batch in dataloader:
            print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
            print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
            print(f"{batch['action'].shape=}")  # (32, 64, c)
            break
    else:
        print("No camera keys found in dataset. Skipping camera-related examples.")


if __name__ == "__main__":
    main()

