import pytest
from iohub.ngff import open_ome_zarr

from viscy.data.select import SelectWell


@pytest.mark.parametrize("include_wells", [None, ["A/1", "A/2", "B/2"]])
@pytest.mark.parametrize("exclude_fovs", [None, ["A/1/0", "A/1/1", "A/2/2"]])
def test_select_well(include_wells, exclude_fovs, preprocessed_hcs_dataset):
    dummy = SelectWell()
    dummy._include_wells = include_wells
    dummy._exclude_fovs = exclude_fovs
    plate = open_ome_zarr(preprocessed_hcs_dataset)
    filtered_positions = dummy._filter_fit_fovs(plate)
    fovs_per_well = len(plate["A/1"])
    if include_wells is None:
        total_wells = len(list(plate.wells()))
    else:
        total_wells = len(include_wells)
    total_fovs = total_wells * fovs_per_well
    if exclude_fovs is not None:
        total_fovs -= len(exclude_fovs)
    assert len(filtered_positions) == total_fovs
    for position in filtered_positions:
        fov_name = position.zgroup.name.strip("/")
        well_name, _ = fov_name.rsplit("/", 1)
        if include_wells is not None:
            assert well_name in include_wells
        if exclude_fovs is not None:
            assert fov_name not in exclude_fovs
