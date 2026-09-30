"""Input adapters for canonical GRIL datasets."""

from gril.adapters.apollo_record import ApolloRecordAdapter
from gril.adapters.rosbag1 import Rosbag1Adapter

__all__ = ["ApolloRecordAdapter", "Rosbag1Adapter"]
