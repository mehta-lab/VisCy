from typing import Generator

from iohub.ngff.nodes import Plate, Position, Well


def _filter_wells(
    plate: Plate, include_wells: list[str] | None
) -> Generator[Well, None, None]:
    for well_name, well in plate.wells():
        if include_wells is None or well_name in include_wells:
            yield well


def _filter_fovs(
    well: Well, exclude_fovs: list[str] | None
) -> Generator[Position, None, None]:
    for _, fov in well.positions():
        fov_name = fov.zgroup.name.strip("/")
        if exclude_fovs is None or fov_name not in exclude_fovs:
            yield fov


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
