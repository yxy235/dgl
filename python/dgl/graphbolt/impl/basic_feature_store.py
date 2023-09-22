"""Basic feature store for GraphBolt."""

from typing import Dict, Tuple

import torch

from ..feature_store import Feature, FeatureStore

__all__ = ["BasicFeatureStore"]


class BasicFeatureStore(FeatureStore):
    r"""Basic feature store."""

    def __init__(self, features: Dict[Tuple[str, str, str], Feature]):
        r"""Initiate a basic feature store.


        Parameters
        ----------
        features : Dict[Tuple[str, str, str], Feature]
            The dict of features served by the feature store, in which the key
            is tuple of (domain, type_name, feature_name).

        Returns
        -------
        The feature stores.
        """
        super().__init__()
        self._features = features

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
        return self._features[(domain, type_name, feature_name)].read(ids)

    def size(
        self,
        domain: str = None,
        type_name: str = None,
        feature_name: str = None,
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
        if (
            domain is None and type_name is None and
            feature_name is None and ids is None
        ):
            for value in self._features.values():
                return value.size()
        return self._features[(domain, type_name, feature_name)].size(ids)

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
        self._features[(domain, type_name, feature_name)].update(value, ids)

    def __len__(self):
        """Return the number of features."""
        return len(self._features)
