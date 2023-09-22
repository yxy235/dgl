"""Feature store for GraphBolt."""

import torch

__all__ = ["Feature", "FeatureStore"]


class Feature:
    r"""Base class for feature."""

    def __init__(self):
        pass

    def read(self, ids: torch.Tensor = None):
        """Read from the feature.

        Parameters
        ----------
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature are read. If None, the entire feature is returned.
        Returns
        -------
        torch.Tensor
            The read feature.
        """
        raise NotImplementedError

    def size(self, ids: torch.Tensor = None):
        """Get the size of the feature.

        Parameters
        ----------
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the size of
            the specified indices of the feature will calculate.
            If None, the entire size of the feature is returned.
        Returns
        -------
        int
            The size of the feature.
        """
        raise NotImplementedError

    def update(self, value: torch.Tensor, ids: torch.Tensor = None):
        """Update the feature.

        Parameters
        ----------
        value : torch.Tensor
            The updated value of the feature.
        ids : torch.Tensor, optional
            The indices of the feature to update. If specified, only the
            specified indices of the feature will be updated. For the feature,
            the `ids[i]` row is updated to `value[i]`. So the indices and value
            must have the same length. If None, the entire feature will be
            updated.
        """
        raise NotImplementedError


class FeatureStore:
    r"""Base class for feature store."""

    def __init__(self):
        pass

    def read(
        self,
        domain: str,
        type_name: str,
        feature_name: str,
        ids: torch.Tensor = None,
    ):
        """Read from the feature store.

        Parameters
        ----------
        domain : str
            The domain of the feature such as "node", "edge" or "graph".
        type_name : str
            The node or edge type name.
        feature_name : str
            The feature name.
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature are read. If None, the entire feature is returned.

        Returns
        -------
        torch.Tensor
            The read feature.
        """
        raise NotImplementedError

    def size(
        self,
        domain: str,
        type_name: str,
        feature_name: str,
        ids: torch.Tensor = None,
    ):
        """Get the size of the feature.

        Parameters
        ----------
        domain : str
            The domain of the feature such as "node", "edge" or "graph".
        type_name : str
            The node or edge type name.
        feature_name : str
            The feature name.
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the size of
            the specified indices of the feature will calculate.
            If None, the entire size of the feature is returned.
        Returns
        -------
        int
            The size of the feature.
        """
        raise NotImplementedError

    def update(
        self,
        domain: str,
        type_name: str,
        feature_name: str,
        value: torch.Tensor,
        ids: torch.Tensor = None,
    ):
        """Update the feature store.

        Parameters
        ----------
        domain : str
            The domain of the feature such as "node", "edge" or "graph".
        type_name : str
            The node or edge type name.
        feature_name : str
            The feature name.
        value : torch.Tensor
            The updated value of the feature.
        ids : torch.Tensor, optional
            The indices of the feature to update. If specified, only the
            specified indices of the feature will be updated. For the feature,
            the `ids[i]` row is updated to `value[i]`. So the indices and value
            must have the same length. If None, the entire feature will be
            updated.
        """
        raise NotImplementedError
