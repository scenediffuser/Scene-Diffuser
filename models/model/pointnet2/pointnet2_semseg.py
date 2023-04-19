import torch
import torch.nn as nn
from models.model.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
import models.model.pointnet2.pytorch_utils as pt_utils


def get_model(num_classes, is_msg=True, input_channels=6, use_xyz=True, bn=True):
    if is_msg:
        model = Pointnet2MSG(
            num_classes=num_classes, 
            input_channels=input_channels, 
            use_xyz=use_xyz, 
            bn=bn
        )
    else:
        model = Pointnet2SSG(
        num_classes=num_classes, 
        input_channels=input_channels, 
        use_xyz=use_xyz, 
        bn=bn
    )

    return model

class Pointnet2MSG(nn.Module):
    def __init__(self, num_classes, input_channels=3, use_xyz=True, bn=True):
        super().__init__()

        NPOINTS = [1024, 256, 64, 16]
        RADIUS = [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
        NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
        MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]],
                [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
        FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
        CLS_FC = [128]
        DP_RATIO = 0.5

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=bn
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k],
                    bn=bn
                )
            )

        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=bn))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, num_classes, activation=None, bn=bn))
        cls_layers.insert(1, nn.Dropout(DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, 1)
        return pred_cls

class Pointnet2SSG(nn.Module):
    def __init__(self, num_classes, input_channels=3, use_xyz=True, bn=True):
        super().__init__()

        NPOINTS = [1024, 256, 64, 16]
        RADIUS = [0.1, 0.2, 0.4, 0.8]
        NSAMPLE = [32, 32, 32, 32]
        MLPS = [[32, 32, 64], [64, 64, 128],
                [128, 128, 256], [256, 256, 512]]
        FP_MLPS = [[128, 128], [256, 128], [256, 256], [256, 256]]
        CLS_FC = [128]
        DP_RATIO = 0.5

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            mlps = [channel_in] + mlps
            channel_out += mlps[-1]

            self.SA_modules.append(
                PointnetSAModule(
                    npoint=NPOINTS[k],
                    radius=RADIUS[k],
                    nsample=NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=bn
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k],
                    bn=bn
                )
            )

        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=bn))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, num_classes, activation=None, bn=bn))
        cls_layers.insert(1, nn.Dropout(DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        # last layer, l_xyz <4, 16, 3>, l_feature <4, 512, 16>
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, categories)
        return pred_cls


def get_feature_extractor(is_msg=True, input_channels=6, use_xyz=True, bn=True):
    if is_msg:
        model = Pointnet2MSG_Feature(
            input_channels=input_channels, 
            use_xyz=use_xyz, 
            bn=bn
        )
    else:
        model = Pointnet2SSG_Feature( 
        input_channels=input_channels, 
        use_xyz=use_xyz, 
        bn=bn
    )

    return model

class Pointnet2MSG_Feature(nn.Module):
    def __init__(self, input_channels=3, use_xyz=True, bn=True):
        super().__init__()

        NPOINTS = [1024, 256, 64, 16]
        RADIUS = [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
        NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
        MLPS = [[[16, 16, 32], [32, 32, 64]],
                [[64, 64, 128], [64, 96, 128]],
                [[128, 196, 256], [128, 196, 256]],
                [[256, 256, 512], [256, 384, 512]]]

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=bn
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        return l_xyz, l_features

class Pointnet2SSG_Feature(nn.Module):
    def __init__(self, input_channels=3, use_xyz=True, bn=True):
        super().__init__()

        # NPOINTS = [1024, 256, 64, 16]
        NPOINTS = [2048, 512, 128, 16]

        RADIUS = [0.02, 0.04, 0.06, 0.08]
        NSAMPLE = [32, 32, 16, 16]
        MLPS = [[32, 32, 64], [64, 64, 128],
                [128, 128, 256], [256, 256, 512]]

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            mlps = [channel_in] + mlps
            channel_out += mlps[-1]

            self.SA_modules.append(
                PointnetSAModule(
                    npoint=NPOINTS[k],
                    radius=RADIUS[k],
                    nsample=NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=bn
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        return l_xyz, l_features


def pointnet2_enc_repro(c=3, num_points=2048):
    assert (num_points == 2048)  ## cannot adapt num of points here
    model = get_feature_extractor(is_msg=False, input_channels=c-3, use_xyz=True, bn=True)
    return model


if __name__ == '__main__':
    print(f'get pointnet2 semseg')
    msg_model = get_feature_extractor(is_msg=False, input_channels=0, use_xyz=True, bn=True)
    msg_model.cuda()

    dummy_pc_input = torch.randn(8, 1024, 3, device='cuda')
    dummy_feat = msg_model(dummy_pc_input)
    print(dummy_feat[-1][-1].shape)