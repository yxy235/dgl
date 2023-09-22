import os
import tempfile

import numpy as np
import pydantic
import pytest
import torch

from dgl import graphbolt as gb


def to_on_disk_tensor(test_dir, name, t):
    path = os.path.join(test_dir, name + ".npy")
    t = t.numpy()
    np.save(path, t)
    # The Pytorch tensor is a view of the numpy array on disk, which does not
    # consume memory.
    t = torch.as_tensor(np.load(path, mmap_mode="r+"))
    return t


@pytest.mark.parametrize("in_memory", [True, False])
def test_torch_based_feature(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([[1, 2, 3], [4, 5, 6]])
        if not in_memory:
            a = to_on_disk_tensor(test_dir, "a", a)
            b = to_on_disk_tensor(test_dir, "b", b)

        feature_a = gb.TorchBasedFeature(a)
        feature_b = gb.TorchBasedFeature(b)

        # Test read the entire feature.
        assert torch.equal(feature_a.read(), torch.tensor([1, 2, 3]))
        assert torch.equal(
            feature_b.read(), torch.tensor([[1, 2, 3], [4, 5, 6]])
        )

        # Test read the feature with ids.
        assert torch.equal(
            feature_a.read(torch.tensor([0, 2])),
            torch.tensor([1, 3]),
        )
        assert torch.equal(
            feature_a.read(torch.tensor([1, 1])),
            torch.tensor([2, 2]),
        )
        assert torch.equal(
            feature_b.read(torch.tensor([1])),
            torch.tensor([[4, 5, 6]]),
        )

        # Test get the size of the entire feature.
        assert feature_a.size() == 3
        assert feature_b.size() == 2

        # Test get the size of the feature with ids.
        assert feature_a.size(torch.tensor([0, 1])) == 2
        assert feature_b.size(torch.tensor([0])) == 1

        # Test update the feature.
        feature_a.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
        assert torch.equal(feature_a.read(), torch.tensor([0, 1, 2]))
        feature_a.update(torch.tensor([2, 0]), torch.tensor([0, 2]))
        assert torch.equal(feature_a.read(), torch.tensor([2, 1, 0]))

        with pytest.raises(IndexError):
            feature_a.read(torch.tensor([0, 1, 2, 3]))

        # For windows, the file is locked by the numpy.load. We need to delete
        # it before closing the temporary directory.
        a = b = None
        feature_a = feature_b = None


def write_tensor_to_disk(dir, name, t, fmt="torch"):
    if fmt == "torch":
        torch.save(t, os.path.join(dir, name + ".pt"))
    elif fmt == "numpy":
        t = t.numpy()
        np.save(os.path.join(dir, name + ".npy"), t)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


@pytest.mark.parametrize("in_memory", [True, False])
def test_torch_based_feature_store(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([2, 5, 3])
        write_tensor_to_disk(test_dir, "a", a, fmt="torch")
        write_tensor_to_disk(test_dir, "b", b, fmt="numpy")
        feature_data = [
            gb.OnDiskFeatureData(
                domain="node",
                type="paper",
                name="a",
                format="torch",
                path=os.path.join(test_dir, "a.pt"),
                in_memory=True,
            ),
            gb.OnDiskFeatureData(
                domain="edge",
                type="paper:cites:paper",
                name="b",
                format="numpy",
                path=os.path.join(test_dir, "b.npy"),
                in_memory=in_memory,
            ),
        ]
        feature_store = gb.TorchBasedFeatureStore(feature_data)

        # Test read the entire feature.
        assert torch.equal(
            feature_store.read("node", "paper", "a"), torch.tensor([1, 2, 3])
        )
        assert torch.equal(
            feature_store.read("edge", "paper:cites:paper", "b"),
            torch.tensor([2, 5, 3]),
        )

        # Test get the size of the entire feature.
        assert feature_store.size("node", "paper", "a") == 3
        assert feature_store.size("edge", "paper:cites:paper", "b") == 3

        # Test get the size of the feature with ids.
        assert feature_store.size("node", "paper", "a", torch.tensor([0])) == 1

        # For windows, the file is locked by the numpy.load. We need to delete
        # it before closing the temporary directory.
        a = b = None
        feature_store = None

        # ``domain`` should be enum.
        with pytest.raises(pydantic.ValidationError):
            _ = gb.OnDiskFeatureData(
                domain="invalid",
                type="paper",
                name="a",
                format="torch",
                path=os.path.join(test_dir, "a.pt"),
                in_memory=True,
            )

        # ``type`` could be null.
        feature_data = [
            gb.OnDiskFeatureData(
                domain="node",
                name="a",
                format="torch",
                path=os.path.join(test_dir, "a.pt"),
                in_memory=True,
            ),
        ]
        feature_store = gb.TorchBasedFeatureStore(feature_data)
        assert torch.equal(
            feature_store.read("node", None, "a"), torch.tensor([1, 2, 3])
        )

        # Test get the size of the entire feature.
        assert feature_store.size("node", None, "a") == 3

        # Test get the size of the feature with ids.
        assert feature_store.size("node", None, "a", torch.tensor([0])) == 1

        feature_store = None
