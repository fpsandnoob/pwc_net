from functools import partial
import mindspore
import mindspore.nn as nn
import mindspore.ops as P

from src.pwc_part import *


class PWCNet(nn.Cell):
    def __init__(self, div_flow=0.05):
        super(PWCNet, self).__init__()
        self.div_flow = div_flow
        self.search_range = 4
        self.num_channels = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyReLU = nn.LeakyReLU(0.1)
        self.pad = P.Pad(
            (
                (0, 0),
                (self.search_range, self.search_range),
                (self.search_range, self.search_range),
                (0, 0),
            )
        )

        self.feature_pyramid_extractor = FeatureExtractor(self.num_channels)
        self.warping_layer = WarpingLayer(warp_type="bilinear")
        self.dense_flow_estimators = nn.CellList()
        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for i, channel in enumerate(self.num_channels[::-1]):
            if i > self.output_level:
                break

            if i == 0:
                num_in_channel = self.dim_corr
            else:
                num_in_channel = self.dim_corr + 2 + channel

            print(num_in_channel)
            self.dense_flow_estimators.append(DenseFlowEstimator(num_in_channel))

        self.context_networks = ContextNetwork(self.dim_corr + 32 + 2 + 448 + 2)

    def compute_cost_volume(self, x1, x2_warp):
        padded = self.pad(x2_warp)
        _, h1, w1, _ = P.Shape()(x1)
        max_offset = self.search_range * 2 + 1

        cost_vol = []
        for i in range(max_offset):
            for j in range(max_offset):
                cost_vol.append(
                    P.ReduceMean(keep_dims=True)(
                        padded[:, i : h1 + i, j : w1 + j, :] * x1, 3
                    )
                )
        cost_vol = P.Concat(3)(cost_vol)
        return cost_vol

    @staticmethod
    def upsample2d_as(input, target_shape_tensor):
        _, _, h1, w1 = P.Shape()(target_shape_tensor)
        _, _, h2, _ = P.Shape()(input)
        resize = h1 / h2
        return P.ResizeBilinear((h1, w1))(input) * resize

    def construct(self, x1_raw, x2_raw, training=True):

        x1_pyramid_feat = self.feature_pyramid_extractor(x1_raw)
        x1_pyramid_feat = x1_pyramid_feat + [x1_raw]
        x2_pyramid_feat = self.feature_pyramid_extractor(x2_raw)
        x2_pyramid_feat = x2_pyramid_feat + [x2_raw]

        b, _, h, w = P.Shape()(x1_pyramid_feat[0])
        flow = P.Zeros()((b, 2, h, w), mindspore.float32)

        flow_pyramid = []

        for i, (x1_feat, x2_feat) in enumerate(zip(x1_pyramid_feat, x2_pyramid_feat)):
            if i == 0:
                warpped = x2_feat
            else:
                # print(flow.shape, x1_feat.shape)
                flow = self.upsample2d_as(flow, x1_feat)
                warpped = self.warping_layer(x2_feat, flow)

            x1_feat_transpose = P.Transpose()(x1_feat, (0, 2, 3, 1))
            warpped = P.Transpose()(warpped, (0, 2, 3, 1))
            cost_vol = self.compute_cost_volume(x1_feat_transpose, warpped)
            cost_vol = P.Transpose()(cost_vol, (0, 3, 1, 2))
            cost_vol_act = self.leakyReLU(cost_vol)

            if i == 0:
                intermediate_feat, flow = self.dense_flow_estimators[i](cost_vol_act)
            else:
                # print(cost_vol_act.shape, x1_feat.shape, flow.shape)
                intermediate_feat, flow = self.dense_flow_estimators[i](
                    P.Concat(1)((cost_vol_act, x1_feat, flow))
                )

            if i == self.output_level:
                flow_res = self.context_networks(P.Concat(1)((intermediate_feat, flow)))
                flow = flow + flow_res
                flow_pyramid.append(flow)
                break
            else:
                flow_pyramid.append(flow)

        if self.training:
            return flow_pyramid
        else:
            out_flow = self.upsample2d_as(flow, x1_raw) * (1.0 / self.div_flow)
            return out_flow
