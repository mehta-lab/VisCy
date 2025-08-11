import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ClusterPoint:
    """Represents a single point in a cluster."""

    track_id: int
    t: int
    fov_name: str
    dataset: str
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
    z_coord: Optional[float] = None

    @property
    def cache_key(self) -> Tuple[str, str, int, int]:
        """Get the cache key for this point."""
        return (self.dataset, self.fov_name, self.track_id, self.t)

    @property
    def unique_track_id(self) -> str:
        """Get a globally unique track identifier."""
        return f"{self.dataset}_{self.track_id}"

    def __eq__(self, other) -> bool:
        """Two points are equal if they have the same cache key."""
        if not isinstance(other, ClusterPoint):
            return False
        return self.cache_key == other.cache_key

    def __hash__(self) -> int:
        """Hash based on cache key for use in sets."""
        return hash(self.cache_key)


@dataclass
class Cluster:
    """Represents a cluster of points with metadata."""

    points: List[ClusterPoint] = field(default_factory=list)
    name: str = ""
    color: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def id(self) -> str:
        """Unique identifier for this cluster."""
        return self._id

    @property
    def size(self) -> int:
        """Number of points in this cluster."""
        return len(self.points)

    @property
    def datasets(self) -> Set[str]:
        """Get all datasets represented in this cluster."""
        return {point.dataset for point in self.points}

    @property
    def cache_keys(self) -> Set[Tuple[str, str, int, int]]:
        """Get all cache keys for points in this cluster."""
        return {point.cache_key for point in self.points}

    def add_point(self, point: ClusterPoint) -> None:
        """Add a point to this cluster."""
        if point not in self.points:
            self.points.append(point)

    def remove_point(self, point: ClusterPoint) -> bool:
        """Remove a point from this cluster. Returns True if point was found and removed."""
        try:
            self.points.remove(point)
            return True
        except ValueError:
            return False

    def contains_cache_key(self, cache_key: Tuple[str, str, int, int]) -> bool:
        """Check if this cluster contains a point with the given cache key."""
        return cache_key in self.cache_keys

    def get_default_name(self, cluster_number: int) -> str:
        """Generate a default name for this cluster."""
        if len(self.datasets) == 1:
            dataset_name = list(self.datasets)[0]
            return f"{dataset_name}: Cluster {cluster_number}"
        else:
            datasets_str = ", ".join(sorted(self.datasets))
            return f"[{datasets_str}]: Cluster {cluster_number}"

    def to_dict_list(self) -> List[Dict]:
        """Convert cluster points to the legacy dictionary format for compatibility."""
        return [
            {
                "track_id": point.track_id,
                "t": point.t,
                "fov_name": point.fov_name,
                "dataset": point.dataset,
            }
            for point in self.points
        ]


class ClusterManager:
    """Manages a collection of clusters."""

    def __init__(self):
        self._clusters: List[Cluster] = []

    @property
    def clusters(self) -> List[Cluster]:
        """Get all clusters."""
        return self._clusters

    @property
    def cluster_count(self) -> int:
        """Get the number of clusters."""
        return len(self._clusters)

    @property
    def all_cluster_points(self) -> Set[Tuple[str, str, int, int]]:
        """Get all cache keys from all clusters."""
        all_keys = set()
        for cluster in self._clusters:
            all_keys.update(cluster.cache_keys)
        return all_keys

    def add_cluster(self, cluster: Cluster) -> str:
        """Add a cluster to the manager and return its ID."""
        self._clusters.append(cluster)
        return cluster.id

    def create_cluster_from_points(
        self, points_data: List[Dict], name: str = ""
    ) -> str:
        """Create a new cluster from point data and add it to the manager."""
        cluster = Cluster()

        for point_data in points_data:
            point = ClusterPoint(
                track_id=point_data["track_id"],
                t=point_data["t"],
                fov_name=point_data["fov_name"],
                dataset=point_data["dataset"],
            )
            cluster.add_point(point)

        if name:
            cluster.name = name
        else:
            # Generate default name
            cluster.name = cluster.get_default_name(len(self._clusters) + 1)

        return self.add_cluster(cluster)

    def remove_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster by ID. Returns True if cluster was found and removed."""
        for i, cluster in enumerate(self._clusters):
            if cluster.id == cluster_id:
                del self._clusters[i]
                return True
        return False

    def remove_last_cluster(self) -> Optional[Cluster]:
        """Remove and return the most recently added cluster."""
        if self._clusters:
            return self._clusters.pop()
        return None

    def get_cluster_by_id(self, cluster_id: str) -> Optional[Cluster]:
        """Get a cluster by its ID."""
        for cluster in self._clusters:
            if cluster.id == cluster_id:
                return cluster
        return None

    def get_cluster_by_index(self, index: int) -> Optional[Cluster]:
        """Get a cluster by its index (for backward compatibility)."""
        if 0 <= index < len(self._clusters):
            return self._clusters[index]
        return None

    def clear_all_clusters(self) -> None:
        """Remove all clusters."""
        self._clusters.clear()

    def get_cluster_colors(self) -> Dict[str, str]:
        """Get a mapping of cluster IDs to their colors."""
        import matplotlib.pyplot as plt

        colors = {}
        for i, cluster in enumerate(self._clusters):
            if not cluster.color:
                # Auto-assign color if not set
                cmap = plt.cm.get_cmap("Set2")
                cluster.color = f"rgb{tuple(int(x * 255) for x in cmap(i % 8)[:3])}"
            colors[cluster.id] = cluster.color
        return colors

    def get_cluster_colors_by_index(self) -> List[str]:
        """Get cluster colors as a list (for backward compatibility)."""
        import matplotlib.pyplot as plt

        colors = []
        for i, cluster in enumerate(self._clusters):
            if not cluster.color:
                cmap = plt.cm.get_cmap("Set2")
                cluster.color = f"rgb{tuple(int(x * 255) for x in cmap(i % 8)[:3])}"
            colors.append(cluster.color)
        return colors

    def is_point_in_any_cluster(self, cache_key: Tuple[str, str, int, int]) -> bool:
        """Check if a point is in any cluster."""
        return any(cluster.contains_cache_key(cache_key) for cluster in self._clusters)

    def get_cluster_containing_point(
        self, cache_key: Tuple[str, str, int, int]
    ) -> Optional[Cluster]:
        """Get the cluster containing a specific point."""
        for cluster in self._clusters:
            if cluster.contains_cache_key(cache_key):
                return cluster
        return None

    def get_point_to_cluster_mapping(self) -> Dict[Tuple[str, str, int, int], int]:
        """Get a mapping from cache keys to cluster indices (for backward compatibility)."""
        point_to_cluster = {}
        for cluster_idx, cluster in enumerate(self._clusters):
            for cache_key in cluster.cache_keys:
                point_to_cluster[cache_key] = cluster_idx
        return point_to_cluster

    def update_cluster_name(self, cluster_id: str, new_name: str) -> bool:
        """Update a cluster's name. Returns True if successful."""
        cluster = self.get_cluster_by_id(cluster_id)
        if cluster:
            cluster.name = new_name
            return True
        return False

    def update_cluster_name_by_index(self, index: int, new_name: str) -> bool:
        """Update a cluster's name by index (for backward compatibility)."""
        cluster = self.get_cluster_by_index(index)
        if cluster:
            cluster.name = new_name
            return True
        return False

    def get_cluster_names_by_index(self) -> Dict[int, str]:
        """Get cluster names mapped by index (for backward compatibility)."""
        return {i: cluster.name for i, cluster in enumerate(self._clusters)}

    def to_legacy_format(self) -> Tuple[List[List[Dict]], Dict[int, str]]:
        """Convert to the legacy format for backward compatibility."""
        clusters_data = [cluster.to_dict_list() for cluster in self._clusters]
        cluster_names = self.get_cluster_names_by_index()
        return clusters_data, cluster_names
