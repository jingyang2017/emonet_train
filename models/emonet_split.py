import torch
import torch.nn as nn
import torch.nn.functional as F
nn.InstanceNorm2d = nn.BatchNorm2d
def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)
class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.InstanceNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.InstanceNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.InstanceNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

class FAN(nn.Module):
    def __init__(self,num_modules):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.InstanceNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0))
            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0))

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg_features = []

        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            hg_features.append(ll)
        return x, hg_features, tmp_out

class FAN_(nn.Module):
    def __init__(self):
        super(FAN_,self).__init__()
        self.num_modules = 2
        self.fan = FAN(self.num_modules)
    def forward(self, x):
        x,hg_features,tmp_out = self.fan(x)
        hg_features_cat = torch.cat(tuple(hg_features), dim=1)
        mask = torch.sum(tmp_out, dim=1, keepdim=True)
        hg_features_cat *= mask
        emo_feat = torch.cat((x, hg_features_cat), dim=1)
        return tmp_out,emo_feat

class EmoNet_(nn.Module):
    def __init__(self,n_expression=5):
        super(EmoNet_, self).__init__()
        from easydict import EasyDict as edict
        self.config = edict()
        self.config.num_input_channels = 256*3
        self.config.n_blocks = 4
        self.config.n_reg = 2
        self.config.emotion_labels = n_expression
        self.conv1x1_input_emo_2 = nn.Conv2d(self.config.num_input_channels, 256, kernel_size=1, stride=1, padding=0)
        self.emo_convs = []
        for in_f, out_f in [(256, 256)] * self.config.n_blocks:
            self.emo_convs.append(ConvBlock(in_f, out_f))
            self.emo_convs.append(nn.MaxPool2d(2, 2))
        self.emo_net_2 = nn.Sequential(*self.emo_convs)
        self.avg_pool_2 = nn.AvgPool2d(4)
        self.emo_fc_2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                      nn.Linear(128, self.config.emotion_labels + self.config.n_reg))

    def forward(self, x):
        emo_feat = self.conv1x1_input_emo_2(x)
        final_features = self.emo_net_2(emo_feat)
        final_features = self.avg_pool_2(final_features)
        batch_size = final_features.shape[0]
        final_features = final_features.view(batch_size, final_features.shape[1])
        final_features = self.emo_fc_2(final_features)
        return final_features

class EmoNet(nn.Module):
    def __init__(self,n_expression=5):
        super(EmoNet, self).__init__()
        self.feature = FAN_()
        self.feature.fan.load_state_dict(torch.load('./models/2dfan2.pth',map_location='cpu'))
        self.feature.eval()
        for p in self.feature.parameters():
           p.requires_grad = False
        self.predictor = EmoNet_(n_expression=n_expression)
    def forward(self,x):
        with torch.no_grad():
            tmp_out, emo_feat = self.feature(x)
        final_features = self.predictor(emo_feat)
        return {'heatmap': tmp_out, 'expression': final_features[:,:-2], 'valence': final_features[:,-2], 'arousal':final_features[:,-1]}

if __name__ == '__main__':
    net = EmoNet(5)
    out = net(torch.randn(2,3,256,256))
    print(out['heatmap'].size())
    print(out['expression'])
    print(out['valence'].size())
    print(out['arousal'].size())