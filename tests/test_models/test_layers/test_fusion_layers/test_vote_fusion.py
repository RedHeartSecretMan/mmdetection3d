# Copyright (c) OpenMMLab. All rights reserved.
"""Tests the core function of vote fusion.

CommandLine:
    pytest tests/test_models/test_fusion/test_vote_fusion.py
"""

import torch
from mmdet3d.models.layers.fusion_layers import VoteFusion


def test_vote_fusion():
    img_meta = {
        "ori_shape": (530, 730),
        "img_shape": (600, 826),
        "pad_shape": (608, 832),
        "scale_factor": torch.tensor([1.1315, 1.1321, 1.1315, 1.1321]),
        "flip": False,
        "pcd_horizontal_flip": False,
        "pcd_vertical_flip": False,
        "pcd_trans": torch.tensor([0.0, 0.0, 0.0]),
        "pcd_scale_factor": 1.0308290128214932,
        "pcd_rotation": torch.tensor(
            [
                [0.9747, 0.2234, 0.0000],
                [-0.2234, 0.9747, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ]
        ),
        "transformation_3d_flow": ["HF", "R", "S", "T"],
    }

    rt_mat = torch.tensor(
        [
            [0.979570, 0.047954, -0.195330],
            [0.047954, 0.887470, 0.458370],
            [0.195330, -0.458370, 0.867030],
        ]
    )
    k_mat = torch.tensor(
        [
            [529.5000, 0.0000, 365.0000],
            [0.0000, 529.5000, 265.0000],
            [0.0000, 0.0000, 1.0000],
        ]
    )
    rt_mat = rt_mat.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ rt_mat.transpose(
        1, 0
    )
    depth2img = k_mat @ rt_mat
    img_meta["depth2img"] = depth2img

    bboxes = torch.tensor(
        [
            [
                [5.4286e02, 9.8283e01, 6.1700e02, 1.6742e02, 9.7922e-01, 3.0000e00],
                [4.2613e02, 8.4646e01, 4.9091e02, 1.6237e02, 9.7848e-01, 3.0000e00],
                [2.5606e02, 7.3244e01, 3.7883e02, 1.8471e02, 9.7317e-01, 3.0000e00],
                [6.0104e02, 1.0648e02, 6.6757e02, 1.9216e02, 8.4607e-01, 3.0000e00],
                [2.2923e02, 1.4984e02, 7.0163e02, 4.6537e02, 3.5719e-01, 0.0000e00],
                [2.5614e02, 7.4965e01, 3.3275e02, 1.5908e02, 2.8688e-01, 3.0000e00],
                [9.8718e00, 1.4142e02, 2.0213e02, 3.3878e02, 1.0935e-01, 3.0000e00],
                [6.1930e02, 1.1768e02, 6.8505e02, 2.0318e02, 1.0720e-01, 3.0000e00],
            ]
        ]
    )

    seeds_3d = torch.tensor(
        [
            [
                [0.044544, 1.675476, -1.531831],
                [2.500625, 7.238662, -0.737675],
                [-0.600003, 4.827733, -0.084022],
                [1.396212, 3.994484, -1.551180],
                [-2.054746, 2.012759, -0.357472],
                [-0.582477, 6.580470, -1.466052],
                [1.313331, 5.722039, 0.123904],
                [-1.107057, 3.450359, -1.043422],
                [1.759746, 5.655951, -1.519564],
                [-0.203003, 6.453243, 0.137703],
                [-0.910429, 0.904407, -0.512307],
                [0.434049, 3.032374, -0.763842],
                [1.438146, 2.289263, -1.546332],
                [0.575622, 5.041906, -0.891143],
                [-1.675931, 1.417597, -1.588347],
            ]
        ]
    )

    imgs = (
        torch.linspace(-1, 1, steps=608 * 832)
        .reshape(1, 608, 832)
        .repeat(3, 1, 1)[None]
    )

    expected_tensor1 = torch.tensor(
        [
            [
                [
                    0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    1.193706e-01,
                    -0.000000e00,
                    -2.879214e-01,
                    -0.000000e00,
                    0.000000e00,
                    1.422463e-01,
                    -6.474612e-01,
                    -0.000000e00,
                    1.490057e-02,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    -1.873745e00,
                    -0.000000e00,
                    1.576240e-01,
                    0.000000e00,
                    -0.000000e00,
                    -3.646177e-02,
                    -7.751858e-01,
                    0.000000e00,
                    9.593642e-02,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    -6.263277e-02,
                    0.000000e00,
                    -3.646387e-01,
                    0.000000e00,
                    0.000000e00,
                    -5.875812e-01,
                    -6.263450e-02,
                    0.000000e00,
                    1.149264e-01,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    8.899736e-01,
                    0.000000e00,
                    9.019017e-01,
                    0.000000e00,
                    0.000000e00,
                    6.917775e-01,
                    8.899733e-01,
                    0.000000e00,
                    9.812444e-01,
                    0.000000e00,
                ],
                [
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -4.516903e-01,
                    -0.000000e00,
                    -2.315422e-01,
                    -0.000000e00,
                    -0.000000e00,
                    -4.197519e-01,
                    -4.516906e-01,
                    -0.000000e00,
                    -1.547615e-01,
                    -0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    3.571937e-01,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    3.571937e-01,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    9.731653e-01,
                    0.000000e00,
                    0.000000e00,
                    1.093455e-01,
                    0.000000e00,
                    0.000000e00,
                    8.460656e-01,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    2.316288e-03,
                    -1.948284e-03,
                    -3.694394e-03,
                    2.176163e-04,
                    -3.882605e-03,
                    -1.901490e-03,
                    -3.355042e-03,
                    -1.774631e-03,
                    -6.981542e-04,
                    -3.886823e-03,
                    -1.302233e-03,
                    -1.189933e-03,
                    2.540967e-03,
                    -1.834944e-03,
                    1.032048e-03,
                ],
                [
                    2.316288e-03,
                    -1.948284e-03,
                    -3.694394e-03,
                    2.176163e-04,
                    -3.882605e-03,
                    -1.901490e-03,
                    -3.355042e-03,
                    -1.774631e-03,
                    -6.981542e-04,
                    -3.886823e-03,
                    -1.302233e-03,
                    -1.189933e-03,
                    2.540967e-03,
                    -1.834944e-03,
                    1.032048e-03,
                ],
                [
                    2.316288e-03,
                    -1.948284e-03,
                    -3.694394e-03,
                    2.176163e-04,
                    -3.882605e-03,
                    -1.901490e-03,
                    -3.355042e-03,
                    -1.774631e-03,
                    -6.981542e-04,
                    -3.886823e-03,
                    -1.302233e-03,
                    -1.189933e-03,
                    2.540967e-03,
                    -1.834944e-03,
                    1.032048e-03,
                ],
            ]
        ]
    )

    expected_tensor2 = torch.tensor(
        [
            [
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
            ]
        ]
    )

    expected_tensor3 = torch.tensor(
        [
            [
                [
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    1.720988e-01,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    4.824460e-02,
                    0.000000e00,
                ],
                [
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    1.447314e-01,
                    -0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    9.759269e-01,
                    0.000000e00,
                ],
                [
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -0.000000e00,
                    -1.631542e-01,
                    -0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    1.072001e-01,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                ],
                [
                    2.316288e-03,
                    -1.948284e-03,
                    -3.694394e-03,
                    2.176163e-04,
                    -3.882605e-03,
                    -1.901490e-03,
                    -3.355042e-03,
                    -1.774631e-03,
                    -6.981542e-04,
                    -3.886823e-03,
                    -1.302233e-03,
                    -1.189933e-03,
                    2.540967e-03,
                    -1.834944e-03,
                    1.032048e-03,
                ],
                [
                    2.316288e-03,
                    -1.948284e-03,
                    -3.694394e-03,
                    2.176163e-04,
                    -3.882605e-03,
                    -1.901490e-03,
                    -3.355042e-03,
                    -1.774631e-03,
                    -6.981542e-04,
                    -3.886823e-03,
                    -1.302233e-03,
                    -1.189933e-03,
                    2.540967e-03,
                    -1.834944e-03,
                    1.032048e-03,
                ],
                [
                    2.316288e-03,
                    -1.948284e-03,
                    -3.694394e-03,
                    2.176163e-04,
                    -3.882605e-03,
                    -1.901490e-03,
                    -3.355042e-03,
                    -1.774631e-03,
                    -6.981542e-04,
                    -3.886823e-03,
                    -1.302233e-03,
                    -1.189933e-03,
                    2.540967e-03,
                    -1.834944e-03,
                    1.032048e-03,
                ],
            ]
        ]
    )

    fusion = VoteFusion()
    out1, out2 = fusion(imgs, bboxes, seeds_3d, [img_meta])
    assert torch.allclose(expected_tensor1, out1[:, :, :15], 1e-3)
    assert torch.allclose(expected_tensor2.float(), out2.float(), 1e-3)
    assert torch.allclose(expected_tensor3, out1[:, :, 30:45], 1e-3)

    out1, out2 = fusion(imgs, bboxes[:, :2], seeds_3d, [img_meta])
    out1 = out1[:, :15, 30:45]
    out2 = out2[:, 30:45].float()
    assert torch.allclose(torch.zeros_like(out1), out1, 1e-3)
    assert torch.allclose(torch.zeros_like(out2), out2, 1e-3)
