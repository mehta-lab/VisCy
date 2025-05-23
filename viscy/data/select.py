from iohub.ngff import Plate, Position

from viscy.preprocessing.precompute import _filter_fovs, _filter_wells


class SelectWell:
    _include_wells: list[str] | None
    _exclude_fovs: list[str] | None

    def _filter_fit_fovs(self, plate: Plate) -> list[Position]:
        positions = []
        for well in _filter_wells(plate, include_wells=self._include_wells):
            for fov in _filter_fovs(well, exclude_fovs=self._exclude_fovs):
                positions.append(fov)
        if len(positions) < 2:
            raise ValueError(
                "At least 2 FOVs are required for training and validation."
            )
        return positions
