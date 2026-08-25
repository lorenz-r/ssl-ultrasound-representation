import torch

from scripts.train_nv_projected_complex_unet import make_mask


def test_rx_tube_masks_complete_receiver_columns():
    mask = make_mask(
        "rx_tube",
        3,
        generator=torch.Generator().manual_seed(7),
        width=8,
    )

    assert mask.shape == (8, 8)
    assert torch.equal(mask.sum(dim=0).sort().values, torch.tensor([0] * 5 + [8] * 3))
    assert torch.all(mask.sum(dim=1) == 3)


def test_scanline_masks_complete_transmit_rows():
    mask = make_mask(
        "scanline",
        3,
        generator=torch.Generator().manual_seed(7),
        width=8,
    )

    assert torch.equal(mask.sum(dim=1).sort().values, torch.tensor([0] * 5 + [8] * 3))
    assert torch.all(mask.sum(dim=0) == 3)
