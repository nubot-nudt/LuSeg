import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class LuSeg_T(nn.Module):
    def __init__(self, n_class):
        super(LuSeg_T, self).__init__()
        resnet_raw_model1 = models.resnet152(pretrained=True)
        resnet_raw_model2 = models.resnet152(pretrained=True)
        self.inplanes = 2048

        ########  Thermal ENCODER  ########
        self.encoder_depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_depth_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                               dim=1)
        self.encoder_depth_bn1 = resnet_raw_model1.bn1
        self.encoder_depth_relu = resnet_raw_model1.relu
        self.encoder_depth_maxpool = resnet_raw_model1.maxpool
        self.encoder_depth_layer1 = resnet_raw_model1.layer1
        self.encoder_depth_layer2 = resnet_raw_model1.layer2
        self.encoder_depth_layer3 = resnet_raw_model1.layer3
        self.encoder_depth_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ########  DECODER  ########
        self.skip_tranform_rgb = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv5_rgb = upbolckV1(cin=2048, cout=1024)
        self.deconv4_rgb = upbolckV1(cin=1024, cout=512)
        self.deconv3_rgb = upbolckV1(cin=512, cout=256)
        self.deconv2_rgb = upbolckV1(cin=256, cout=128)
        self.deconv1_rgb = upbolckV1(cin=128, cout=n_class)

        self.skip_tranform_depth = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv5_depth = upbolckV1(cin=2048, cout=1024)
        self.deconv4_depth = upbolckV1(cin=1024, cout=512)
        self.deconv3_depth = upbolckV1(cin=512, cout=256)
        self.deconv2_depth = upbolckV1(cin=256, cout=128)
        self.deconv1_depth = upbolckV1(cin=128, cout=n_class)

        self.fusion5 = FusionV1()
        self.fusion4 = FusionV1()
        self.fusion3 = FusionV10(in_channel=256, n_class=n_class)
        self.fusion2 = FusionV10(in_channel=128, n_class=n_class)
        self.fusion1 = FusionV10(in_channel=n_class, n_class=n_class)

    def forward(self, input):

        rgb = input[:, :3]
        depth = input[:, 3:]

        verbose = False

        # encoder

        ######################################################################

        if verbose: print("rgb.size() original: ", rgb.size())  # (288, 512)
        if verbose: print("depth.size() original: ", depth.size())  # (288, 512)
        ######################################################################

        depth = self.encoder_depth_conv1(depth)
        if verbose: print("depth.size() after conv1: ", depth.size())  # (144, 256)
        depth = self.encoder_depth_bn1(depth)
        if verbose: print("depth.size() after bn1: ", depth.size())  # (144, 256)
        depth = self.encoder_depth_relu(depth)
        if verbose: print("depth.size() after relu: ", depth.size())  # (144, 256)
        skip_d_1 = depth

        ######################################################################

        depth = self.encoder_depth_maxpool(depth)
        if verbose: print("depth.size() after maxpool: ", depth.size())  # (72, 128)
        depth = self.encoder_depth_layer1(depth)
        if verbose: print("depth.size() after layer1: ", depth.size())  # (72, 128)
        skip_d_2 = depth

        ######################################################################

        depth = self.encoder_depth_layer2(depth)
        if verbose: print("depth.size() after layer2: ", depth.size())  # (36, 64)
        skip_d_3 = depth
        ######################################################################

        depth = self.encoder_depth_layer3(depth)
        if verbose: print("depth.size() after layer3: ", depth.size())  # (18, 32)
        skip_d_4 = depth
        ######################################################################

        depth = self.encoder_depth_layer4(depth)
        if verbose: print("depth.size() after layer4: ", depth.size())  # (9, 16)

        depth_en_out = depth.view(depth.size(0), -1)

        ######################################################################

        # decoder

        depth_decoder_out_5 = self.deconv5_depth(depth)
        depth_decoder_out_5 = depth_decoder_out_5 + skip_d_4
        if verbose: print("fuse after deconv1: ", depth_decoder_out_5.size())  # (18, 32)

        depth_decoder_out_4 = self.deconv4_depth(depth_decoder_out_5)
        depth_decoder_out_4 = depth_decoder_out_4 + skip_d_3
        if verbose: print("fuse after deconv2: ", depth_decoder_out_4.size())  # (36, 64)

        depth_decoder_out_3 = self.deconv3_depth(depth_decoder_out_4)
        depth_decoder_out_3 = depth_decoder_out_3 + skip_d_2
        if verbose: print("fuse after deconv3: ", depth_decoder_out_3.size())  # (72, 128)

        depth_decoder_out_2 = self.deconv2_depth(depth_decoder_out_3)
        skip_d_1 = self.skip_tranform_depth(skip_d_1)
        depth_decoder_out_2 = depth_decoder_out_2 + skip_d_1
        if verbose: print("fuse after deconv4: ", depth_decoder_out_2.size())  # (144, 256)

        depth_decoder_out_1 = self.deconv1_depth(depth_decoder_out_2)
        if verbose: print("fuse after deconv5: ", depth_decoder_out_2.size())  # (288, 512)

        # RGB Encoder

        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())  # (144, 256)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (144, 256)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (144, 256)
        skip_r_1 = rgb

        ######################################################################

        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size())  # (72, 128)
        rgb = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size())  # (72, 128)
        skip_r_2 = rgb

        ######################################################################

        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())  # (36, 64)
        skip_r_3 = rgb
        ######################################################################

        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())  # (18, 32)
        skip_r_4 = rgb
        ######################################################################

        rgb = self.encoder_rgb_layer4(rgb)
        rgb_en_out = rgb.view(rgb.size(0), -1)
        if verbose: print("rgb.size() after layer4: ", rgb.size())  # (9, 16)

        ######################################################################

        # RGB decoder
        rgb_decoder_out_5 = self.deconv5_rgb(rgb)
        rgb_decoder_out_5 = rgb_decoder_out_5 + skip_r_4
        if verbose: print("fuse after deconv1: ", rgb_decoder_out_5.size())  # (18, 32)
        rgb_decoder_out_5 = self.fusion5(rgb_decoder_out_5, depth_decoder_out_5)

        rgb_decoder_out_4 = self.deconv4_rgb(rgb_decoder_out_5)
        rgb_decoder_out_4 = rgb_decoder_out_4 + skip_r_3
        if verbose: print("fuse after deconv2: ", rgb_decoder_out_4.size())  # (36, 64)
        rgb_decoder_out_4 = self.fusion4(rgb_decoder_out_4, depth_decoder_out_4)

        rgb_decoder_out_3 = self.deconv3_rgb(rgb_decoder_out_4)
        rgb_decoder_out_3 = rgb_decoder_out_3 + skip_r_2
        if verbose: print("fuse after deconv3: ", rgb_decoder_out_3.size())  # (72, 128)
        rgb_seg_f3, depth_add_f3, rgb_decoder_out_3 = self.fusion3(rgb_decoder_out_3,
                                                                   depth_decoder_out_3)  ##  rgb image pre-segmentaion,  complement feature, fusion result

        rgb_decoder_out_2 = self.deconv2_rgb(rgb_decoder_out_3)
        skip_r_1 = self.skip_tranform_rgb(skip_r_1)
        rgb_decoder_out_2 = rgb_decoder_out_2 + skip_r_1
        if verbose: print("fuse after deconv4: ", rgb_decoder_out_2.size())  # (144, 256)
        rgb_seg_f2, depth_add_f2, rgb_decoder_out_2 = self.fusion2(rgb_decoder_out_2,
                                                                   depth_decoder_out_2)  ##  rgb image pre-segmentaion,  complement feature, fusion result

        rgb_decoder_out_1 = self.deconv1_rgb(rgb_decoder_out_2)
        if verbose: print("fuse after deconv5: ", rgb_decoder_out_1.size())  # (288, 512)
        rgb_seg_f1, depth_add_f1, rgb_decoder_out_1 = self.fusion1(rgb_decoder_out_1,
                                                                   depth_decoder_out_1)  ##  rgb image pre-segmentaion,  complement feature, fusion result

        return depth_en_out, rgb_en_out, depth_decoder_out_1, rgb_decoder_out_1, rgb_seg_f1, depth_add_f1, rgb_seg_f2, depth_add_f2, rgb_seg_f3, depth_add_f3  ## depth image result, rgb image result, rgb image pre-segmentaion,  complement feature


class LuSeg_S(nn.Module):
    def __init__(self, n_class):
        super(LuSeg_S, self).__init__()
        resnet_raw_model1 = models.resnet152(pretrained=True)
        resnet_raw_model2 = models.resnet152(pretrained=True)
        self.inplanes = 2048

        ########  Thermal ENCODER  ########
        self.encoder_depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_depth_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                               dim=1)
        self.encoder_depth_bn1 = resnet_raw_model1.bn1
        self.encoder_depth_relu = resnet_raw_model1.relu
        self.encoder_depth_maxpool = resnet_raw_model1.maxpool
        self.encoder_depth_layer1 = resnet_raw_model1.layer1
        self.encoder_depth_layer2 = resnet_raw_model1.layer2
        self.encoder_depth_layer3 = resnet_raw_model1.layer3
        self.encoder_depth_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ########  DECODER  ########
        self.skip_tranform_rgb = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv5_rgb = upbolckV1(cin=2048, cout=1024)
        self.deconv4_rgb = upbolckV1(cin=1024, cout=512)
        self.deconv3_rgb = upbolckV1(cin=512, cout=256)
        self.deconv2_rgb = upbolckV1(cin=256, cout=128)
        self.deconv1_rgb = upbolckV1(cin=128, cout=n_class)

        self.skip_tranform_depth = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv5_depth = upbolckV1(cin=2048, cout=1024)
        self.deconv4_depth = upbolckV1(cin=1024, cout=512)
        self.deconv3_depth = upbolckV1(cin=512, cout=256)
        self.deconv2_depth = upbolckV1(cin=256, cout=128)
        self.deconv1_depth = upbolckV1(cin=128, cout=n_class)

        self.fusion5 = FusionV1()
        self.fusion4 = FusionV1()
        self.fusion3 = FusionV10(in_channel=256, n_class=n_class)
        self.fusion2 = FusionV10(in_channel=128, n_class=n_class)
        self.fusion1 = FusionV10(in_channel=n_class, n_class=n_class)

        self.decoder_cat1 = nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(1024)

        self.decoder_cat2 = nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(512)

        self.decoder_cat3 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.decoder_bn3 = nn.BatchNorm2d(256)

        self.decoder_cat4 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(128)

    def forward(self, input):

        rgb = input[:, :3]
        depth = input[:, 3:]

        verbose = False

        # encoder

        ######################################################################

        if verbose: print("rgb.size() original: ", rgb.size())  # (288, 512)
        if verbose: print("depth.size() original: ", depth.size())  # (288, 512)
        ######################################################################

        depth = self.encoder_depth_conv1(depth)
        if verbose: print("depth.size() after conv1: ", depth.size())  # (144, 256)
        depth = self.encoder_depth_bn1(depth)
        if verbose: print("depth.size() after bn1: ", depth.size())  # (144, 256)
        depth = self.encoder_depth_relu(depth)
        if verbose: print("depth.size() after relu: ", depth.size())  # (144, 256)
        skip_d_1 = depth

        ######################################################################

        depth = self.encoder_depth_maxpool(depth)
        if verbose: print("depth.size() after maxpool: ", depth.size())  # (72, 128)
        depth = self.encoder_depth_layer1(depth)
        if verbose: print("depth.size() after layer1: ", depth.size())  # (72, 128)
        skip_d_2 = depth

        ######################################################################

        depth = self.encoder_depth_layer2(depth)
        if verbose: print("depth.size() after layer2: ", depth.size())  # (36, 64)
        skip_d_3 = depth
        ######################################################################

        depth = self.encoder_depth_layer3(depth)
        if verbose: print("depth.size() after layer3: ", depth.size())  # (18, 32)
        skip_d_4 = depth
        ######################################################################

        depth = self.encoder_depth_layer4(depth)
        if verbose: print("depth.size() after layer4: ", depth.size())  # (9, 16)

        depth_en_out = depth.view(depth.size(0), -1)

        ######################################################################


        # RGB Encoder

        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())  # (144, 256)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (144, 256)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (144, 256)
        skip_r_1 = rgb

        ######################################################################

        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size())  # (72, 128)
        rgb = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size())  # (72, 128)
        skip_r_2 = rgb

        ######################################################################

        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())  # (36, 64)
        skip_r_3 = rgb
        ######################################################################

        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())  # (18, 32)
        skip_r_4 = rgb
        ######################################################################

        rgb = self.encoder_rgb_layer4(rgb)
        rgb_en_out = rgb.view(rgb.size(0), -1)
        if verbose: print("rgb.size() after layer4: ", rgb.size())  # (9, 16)

        ######################################################################

        # RGB decoder
        out = rgb + depth
        rgb_decoder_out_5 = self.deconv5_rgb(out)
        rgb_decoder_out_5 = torch.cat((rgb_decoder_out_5, skip_r_4, skip_d_4), dim=1)
        rgb_decoder_out_5 = self.decoder_cat1(rgb_decoder_out_5)
        rgb_decoder_out_5 = self.decoder_bn1(rgb_decoder_out_5)

        if verbose: print("fuse after deconv1: ", rgb_decoder_out_5.size())  # (18, 32)


        rgb_decoder_out_4 = self.deconv4_rgb(rgb_decoder_out_5)
        rgb_decoder_out_4 = torch.cat((rgb_decoder_out_4, skip_r_3, skip_d_3), dim=1)
        rgb_decoder_out_4 = self.decoder_cat2(rgb_decoder_out_4)
        rgb_decoder_out_4 = self.decoder_bn2(rgb_decoder_out_4)

        if verbose: print("fuse after deconv2: ", rgb_decoder_out_4.size())  # (36, 64)

        rgb_decoder_out_3 = self.deconv3_rgb(rgb_decoder_out_4)

        rgb_decoder_out_3 = torch.cat((rgb_decoder_out_3, skip_r_2, skip_d_2), dim=1)
        rgb_decoder_out_3 = self.decoder_cat3(rgb_decoder_out_3)
        rgb_decoder_out_3 = self.decoder_bn3(rgb_decoder_out_3)
        if verbose: print("fuse after deconv3: ", rgb_decoder_out_3.size())  # (72, 128)
        
        rgb_decoder_out_2 = self.deconv2_rgb(rgb_decoder_out_3)
        skip_r_1 = self.skip_tranform_rgb(skip_r_1)
        skip_d_1 = self.skip_tranform_depth(skip_d_1)
        
        rgb_decoder_out_2 = torch.cat((rgb_decoder_out_2, skip_r_1, skip_d_1), dim=1)
        rgb_decoder_out_2 = self.decoder_cat4(rgb_decoder_out_2)
        rgb_decoder_out_2 = self.decoder_bn4(rgb_decoder_out_2)

        if verbose: print("fuse after deconv4: ", rgb_decoder_out_2.size())  # (144, 256)
        
        rgb_decoder_out_1 = self.deconv1_rgb(rgb_decoder_out_2)
        if verbose: print("fuse after deconv5: ", rgb_decoder_out_1.size())  # (288, 512)
        
        return depth_en_out, rgb_en_out, rgb_decoder_out_1  ## depth image result, rgb image result, rgb image pre-segmentaion,  complement feature

class LuSeg_RGB(nn.Module):
    def __init__(self, n_class):
        super(LuSeg_RGB, self).__init__()
        resnet_raw_model1 = models.resnet152(pretrained=True)
        resnet_raw_model2 = models.resnet152(pretrained=True)
        self.inplanes = 2048

        ########  Thermal ENCODER  ########
        self.encoder_depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_depth_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                               dim=1)
        self.encoder_depth_bn1 = resnet_raw_model1.bn1
        self.encoder_depth_relu = resnet_raw_model1.relu
        self.encoder_depth_maxpool = resnet_raw_model1.maxpool
        self.encoder_depth_layer1 = resnet_raw_model1.layer1
        self.encoder_depth_layer2 = resnet_raw_model1.layer2
        self.encoder_depth_layer3 = resnet_raw_model1.layer3
        self.encoder_depth_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ########  DECODER  ########
        self.skip_tranform_rgb = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv5_rgb = upbolckV1(cin=2048, cout=1024)
        self.deconv4_rgb = upbolckV1(cin=1024, cout=512)
        self.deconv3_rgb = upbolckV1(cin=512, cout=256)
        self.deconv2_rgb = upbolckV1(cin=256, cout=128)
        self.deconv1_rgb = upbolckV1(cin=128, cout=n_class)

        self.skip_tranform_depth = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv5_depth = upbolckV1(cin=2048, cout=1024)
        self.deconv4_depth = upbolckV1(cin=1024, cout=512)
        self.deconv3_depth = upbolckV1(cin=512, cout=256)
        self.deconv2_depth = upbolckV1(cin=256, cout=128)
        self.deconv1_depth = upbolckV1(cin=128, cout=n_class)

        self.fusion5 = FusionV1()
        self.fusion4 = FusionV1()
        self.fusion3 = FusionV10(in_channel=256, n_class=n_class)
        self.fusion2 = FusionV10(in_channel=128, n_class=n_class)
        self.fusion1 = FusionV10(in_channel=n_class, n_class=n_class)

        self.decoder_cat1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(1024)

        self.decoder_cat2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(512)

        self.decoder_cat3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.decoder_bn3 = nn.BatchNorm2d(256)

        self.decoder_cat4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(128)

    def forward(self, input):

        rgb = input[:, :3]
        depth = input[:, 3:]

        verbose = False


        # RGB Encoder

        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())  # (144, 256)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (144, 256)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (144, 256)
        skip_r_1 = rgb

        ######################################################################

        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size())  # (72, 128)
        rgb = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size())  # (72, 128)
        skip_r_2 = rgb

        ######################################################################

        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())  # (36, 64)
        skip_r_3 = rgb
        ######################################################################

        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())  # (18, 32)
        skip_r_4 = rgb
        ######################################################################

        rgb = self.encoder_rgb_layer4(rgb)
        rgb_en_out = rgb.view(rgb.size(0), -1)
        if verbose: print("rgb.size() after layer4: ", rgb.size())  # (9, 16)

        ######################################################################

        # RGB decoder
        out = rgb
        rgb_decoder_out_5 = self.deconv5_rgb(out)
        # rgb_decoder_out_5 = rgb_decoder_out_5 + skip_r_4
        rgb_decoder_out_5 = torch.cat((rgb_decoder_out_5, skip_r_4), dim=1)
        rgb_decoder_out_5 = self.decoder_cat1(rgb_decoder_out_5)
        rgb_decoder_out_5 = self.decoder_bn1(rgb_decoder_out_5)

        if verbose: print("fuse after deconv1: ", rgb_decoder_out_5.size())  # (18, 32)


        rgb_decoder_out_4 = self.deconv4_rgb(rgb_decoder_out_5)

        rgb_decoder_out_4 = torch.cat((rgb_decoder_out_4, skip_r_3), dim=1)
        rgb_decoder_out_4 = self.decoder_cat2(rgb_decoder_out_4)
        rgb_decoder_out_4 = self.decoder_bn2(rgb_decoder_out_4)

        if verbose: print("fuse after deconv2: ", rgb_decoder_out_4.size())  # (36, 64)

        rgb_decoder_out_3 = self.deconv3_rgb(rgb_decoder_out_4)

        rgb_decoder_out_3 = torch.cat((rgb_decoder_out_3, skip_r_2), dim=1)
        rgb_decoder_out_3 = self.decoder_cat3(rgb_decoder_out_3)
        rgb_decoder_out_3 = self.decoder_bn3(rgb_decoder_out_3)
        if verbose: print("fuse after deconv3: ", rgb_decoder_out_3.size())  # (72, 128)
      
        rgb_decoder_out_2 = self.deconv2_rgb(rgb_decoder_out_3)
        skip_r_1 = self.skip_tranform_rgb(skip_r_1)

        rgb_decoder_out_2 = torch.cat((rgb_decoder_out_2, skip_r_1), dim=1)
        rgb_decoder_out_2 = self.decoder_cat4(rgb_decoder_out_2)
        rgb_decoder_out_2 = self.decoder_bn4(rgb_decoder_out_2)

        if verbose: print("fuse after deconv4: ", rgb_decoder_out_2.size())  # (144, 256)
        
        rgb_decoder_out_1 = self.deconv1_rgb(rgb_decoder_out_2)
        if verbose: print("fuse after deconv5: ", rgb_decoder_out_1.size())  # (288, 512)
        
        return rgb_en_out, rgb_decoder_out_1  ## depth image result, rgb image result, rgb image pre-segmentaion,  complement feature



class LuSeg_Depth(nn.Module):
    def __init__(self, n_class):
        super(LuSeg_Depth, self).__init__()
        resnet_raw_model1 = models.resnet152(pretrained=True)
        resnet_raw_model2 = models.resnet152(pretrained=True)
        self.inplanes = 2048

        ########  Thermal ENCODER  ########
        self.encoder_depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_depth_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                               dim=1)
        self.encoder_depth_bn1 = resnet_raw_model1.bn1
        self.encoder_depth_relu = resnet_raw_model1.relu
        self.encoder_depth_maxpool = resnet_raw_model1.maxpool
        self.encoder_depth_layer1 = resnet_raw_model1.layer1
        self.encoder_depth_layer2 = resnet_raw_model1.layer2
        self.encoder_depth_layer3 = resnet_raw_model1.layer3
        self.encoder_depth_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ########  DECODER  ########
        self.skip_tranform_rgb = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv5_rgb = upbolckV1(cin=2048, cout=1024)
        self.deconv4_rgb = upbolckV1(cin=1024, cout=512)
        self.deconv3_rgb = upbolckV1(cin=512, cout=256)
        self.deconv2_rgb = upbolckV1(cin=256, cout=128)
        self.deconv1_rgb = upbolckV1(cin=128, cout=n_class)

        self.skip_tranform_depth = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv5_depth = upbolckV1(cin=2048, cout=1024)
        self.deconv4_depth = upbolckV1(cin=1024, cout=512)
        self.deconv3_depth = upbolckV1(cin=512, cout=256)
        self.deconv2_depth = upbolckV1(cin=256, cout=128)
        self.deconv1_depth = upbolckV1(cin=128, cout=n_class)

        self.fusion5 = FusionV1()
        self.fusion4 = FusionV1()
        self.fusion3 = FusionV10(in_channel=256, n_class=n_class)
        self.fusion2 = FusionV10(in_channel=128, n_class=n_class)
        self.fusion1 = FusionV10(in_channel=n_class, n_class=n_class)

        self.decoder_cat1 = nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(1024)

        self.decoder_cat2 = nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(512)

        self.decoder_cat3 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.decoder_bn3 = nn.BatchNorm2d(256)

        self.decoder_cat4 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(128)

    def forward(self, input):

        rgb = input[:, :3]
        depth = input[:, 3:]

        verbose = False

        # encoder

        ######################################################################

        if verbose: print("rgb.size() original: ", rgb.size())  # (288, 512)
        if verbose: print("depth.size() original: ", depth.size())  # (288, 512)
        ######################################################################

        depth = self.encoder_depth_conv1(depth)
        if verbose: print("depth.size() after conv1: ", depth.size())  # (144, 256)
        depth = self.encoder_depth_bn1(depth)
        if verbose: print("depth.size() after bn1: ", depth.size())  # (144, 256)
        depth = self.encoder_depth_relu(depth)
        if verbose: print("depth.size() after relu: ", depth.size())  # (144, 256)
        skip_d_1 = depth

        ######################################################################

        depth = self.encoder_depth_maxpool(depth)
        if verbose: print("depth.size() after maxpool: ", depth.size())  # (72, 128)
        depth = self.encoder_depth_layer1(depth)
        if verbose: print("depth.size() after layer1: ", depth.size())  # (72, 128)
        skip_d_2 = depth

        ######################################################################

        depth = self.encoder_depth_layer2(depth)
        if verbose: print("depth.size() after layer2: ", depth.size())  # (36, 64)
        skip_d_3 = depth
        ######################################################################

        depth = self.encoder_depth_layer3(depth)
        if verbose: print("depth.size() after layer3: ", depth.size())  # (18, 32)
        skip_d_4 = depth
        ######################################################################

        depth = self.encoder_depth_layer4(depth)
        if verbose: print("depth.size() after layer4: ", depth.size())  # (9, 16)

        depth_en_out = depth.view(depth.size(0), -1)

        ######################################################################

        # decoder

        depth_decoder_out_5 = self.deconv5_depth(depth)
        depth_decoder_out_5 = depth_decoder_out_5 + skip_d_4
        if verbose: print("fuse after deconv1: ", depth_decoder_out_5.size())  # (18, 32)

        depth_decoder_out_4 = self.deconv4_depth(depth_decoder_out_5)
        depth_decoder_out_4 = depth_decoder_out_4 + skip_d_3
        if verbose: print("fuse after deconv2: ", depth_decoder_out_4.size())  # (36, 64)

        depth_decoder_out_3 = self.deconv3_depth(depth_decoder_out_4)
        depth_decoder_out_3 = depth_decoder_out_3 + skip_d_2
        if verbose: print("fuse after deconv3: ", depth_decoder_out_3.size())  # (72, 128)

        depth_decoder_out_2 = self.deconv2_depth(depth_decoder_out_3)
        skip_d_1 = self.skip_tranform_depth(skip_d_1)
        depth_decoder_out_2 = depth_decoder_out_2 + skip_d_1
        if verbose: print("fuse after deconv4: ", depth_decoder_out_2.size())  # (144, 256)

        depth_decoder_out_1 = self.deconv1_depth(depth_decoder_out_2)
        if verbose: print("fuse after deconv5: ", depth_decoder_out_2.size())  # (288, 512)

        return depth_en_out, depth_decoder_out_1## depth image result, rgb image result, rgb image pre-segmentaion,  complement feature


###########################################
# constrast loss
###########################################
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-6

    def forward(self, x1, x2):
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)

        sim_matrix = torch.matmul(x1, x2.T) / self.temperature
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=-1, keepdim=True)[0]


        labels = torch.arange(x1.size(0)).to(x1.device)
        loss = nn.CrossEntropyLoss()(sim_matrix, labels)

        return loss


class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class upbolckV1(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        self.conv1 = nn.Conv2d(cin, cin // 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cin // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(cin // 2, cin // 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(cin // 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(cin // 2, cin // 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(cin // 2)
        self.relu3 = nn.ReLU(inplace=True)

        self.shortcutconv = nn.Conv2d(cin, cin // 2, kernel_size=1, stride=1)
        self.shortcutbn = nn.BatchNorm2d(cin // 2)
        self.shortcutrelu = nn.ReLU(inplace=True)

        self.se = SE_fz(in_channels=cin // 2, med_channels=cin // 4)

        self.transconv = nn.ConvTranspose2d(cin // 2, cout, kernel_size=2, stride=2, padding=0, bias=False)
        self.transbn = nn.BatchNorm2d(cout)
        self.transrelu = nn.ReLU(inplace=True)

    def forward(self, x):
        fusion = self.conv1(x)
        fusion = self.bn1(fusion)
        fusion = self.relu1(fusion)

        fusion = self.conv2(fusion)
        fusion = self.bn2(fusion)
        fusion = self.relu2(fusion)

        fusion = self.conv3(fusion)
        fusion = self.bn3(fusion)
        fusion = self.relu3(fusion)

        sc = self.shortcutconv(x)
        sc = self.shortcutbn(sc)
        sc = self.shortcutrelu(sc)

        fusion = fusion + sc

        fusion = self.se(fusion)

        fusion = self.transconv(fusion)
        fusion = self.transbn(fusion)
        fusion = self.transrelu(fusion)

        return fusion


class SE_fz(nn.Module):
    def __init__(self, in_channels, med_channels):
        super(SE_fz, self).__init__()

        self.average = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels, med_channels)
        self.bn1 = nn.BatchNorm1d(med_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(med_channels, in_channels)
        self.sg = nn.Sigmoid()

    def forward(self, input):
        x = input
        x = self.average(input)
        x = x.squeeze(2)
        x = x.squeeze(2)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sg(x)
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        out = torch.mul(input, x)
        return out


class FusionV10(nn.Module):
    def __init__(self, in_channel, n_class):
        super(FusionV10, self).__init__()
        self.seg = in_channel != n_class
        if self.seg:
            self.rgbsegconv = nn.Conv2d(in_channels=in_channel, out_channels=n_class, kernel_size=1, stride=1,
                                        padding=0)
            self.depthsegconv = nn.Conv2d(in_channels=in_channel, out_channels=n_class, kernel_size=1, stride=1,
                                          padding=0)
            self.feedback = nn.Conv2d(in_channels=n_class, out_channels=in_channel, kernel_size=1)

        self.feature_conv1 = nn.Conv2d(in_channels=n_class, out_channels=n_class, kernel_size=3, stride=1, padding=1)

        self.fusion_conv = nn.Conv2d(in_channels=3 * in_channel, out_channels=in_channel, kernel_size=1)

    def forward(self, rgb, depth):

        sub_fusion_B = depth - rgb
        sub_fusion = sub_fusion_B

        if self.seg:
            rgb_seg = self.rgbsegconv(rgb)
            sub_fusion = self.depthsegconv(sub_fusion)
        else:
            rgb_seg = rgb

        depth_add = self.feature_conv1(sub_fusion)
        depth_add = depth_add + sub_fusion

        if self.seg:
            depth_add_feedback = self.feedback(depth_add)
        else:
            depth_add_feedback = depth_add

        fusion_result = torch.cat((rgb, rgb * depth_add_feedback, depth_add_feedback), dim=1)
        fusion_result = self.fusion_conv(fusion_result)

        return rgb_seg, depth_add, fusion_result  ##  rgb image pre-segmentaion,  complement feature, fusion result


class FusionV1(nn.Module):
    def __init__(self):
        super(FusionV1, self).__init__()

    def forward(self, depth, rgb):
        fusion = depth + rgb

        return fusion


def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 288, 512).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 288, 512).cuda(0)
    rtf_net = InconSeg_S(9).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    rgb_en_out, depth_decoder_out_1 = rtf_net(input)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()
