import os
import torch
import torch.nn as nn

from loguru import logger


class PointNetEnc(nn.Module):
    def __init__(self,
                 layers_size=[64, 128, 512], c=3, num_points=2048, num_tokens=8):
        super(PointNetEnc, self).__init__()
        self.num_groups = num_tokens
        self.c = c
        self.num_points = num_points
        self.layers_size = [c] + layers_size
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activate_func = nn.ReLU()
        for i in range(len(self.layers_size) - 1):
            self.conv_layers.append(nn.Conv1d(self.layers_size[i], self.layers_size[i + 1], 1))
            self.bn_layers.append(nn.BatchNorm1d(self.layers_size[i+1]))
            nn.init.xavier_normal_(self.conv_layers[-1].weight)

        self.feat_size = layers_size[-1]
        self.togen_layers_size = [self.feat_size, 4 * self.feat_size, num_tokens * self.feat_size]
        self.togen_conv_layers = nn.ModuleList()
        self.togen_bn_layers = nn.ModuleList()
        self.togen_activate_func = nn.ReLU()
        for i in range(len(self.togen_layers_size) - 1):
            self.togen_conv_layers.append(nn.Conv1d(self.togen_layers_size[i], self.togen_layers_size[i+1], 1))
            self.togen_bn_layers.append(nn.BatchNorm1d(self.togen_layers_size[i+1]))
            nn.init.xavier_normal_(self.togen_conv_layers[-1].weight)

    def forward(self, x):
        # input: B * N * c
        # output: B * latent_size
        x = x.transpose(1, 2)
        for i in range(len(self.conv_layers) - 1):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.activate_func(x)
        x = self.bn_layers[-1](self.conv_layers[-1](x))
        x = torch.max(x, 2, keepdim=True)[0]  # B x self.feat_size x 1

        # x = x.view(-1, 1, self.layers_size[-1])
        # forward token padding(togen) layer
        for i in range(len(self.togen_conv_layers) - 1):
            x = self.togen_conv_layers[i](x)
            x = self.togen_bn_layers[i](x)
            x = self.togen_activate_func(x)
        x = self.togen_bn_layers[-1](self.togen_conv_layers[-1](x)).squeeze(2)

        return x

    def load_pretrained_weight(self, weigth_path: str) -> None:
        if weigth_path is None:
            logger.info(f'Train Scene model(PointNet) from scratch...')
            return

        if not os.path.exists(weigth_path):
            raise Exception('Can\'t find pretrained point-transformer weights.')

        model_dict = torch.load(weigth_path)
        static_dict = {}
        for key in model_dict.keys():
            if 'enc' in key:
                static_dict[key] = model_dict[key]

        self.load_state_dict(static_dict)
        logger.info(f'Load pretrained scene model(PointNet): {weigth_path}')


def pointnet_enc_repro(**kwargs):
    model = PointNetEnc([64, 128, 512], **kwargs)
    return model


if __name__ == '__main__':
    random_model = pointnet_enc_repro(c=3, num_points=2048)
    dummy_inputs = torch.randn(1, 2048, 3)
    o = random_model(dummy_inputs)
    print()

