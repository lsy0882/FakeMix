import torch
import torchvision.models as models
import torch.nn.functional as F

class Visual_encoder(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(Visual_encoder, self).__init__()
        self.num_classes = num_classes
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = torch.nn.Linear(2048, 512)
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.5)
        self.fc = torch.nn.Linear(512, 128)

    def forward(self, x):
        pre_x = self.backbone(x[:, 0, :, :, :])
        current_x = self.backbone(x[:, 1, :, :, :])
        pre_x, current_x = pre_x.unsqueeze(1), current_x.unsqueeze(1)
        current_x, _ = self.cross_attention(current_x, pre_x, pre_x)
        x_rep = current_x.squeeze(1)
        return self.fc(x_rep), x_rep

class Audial_encoder(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(Audial_encoder, self).__init__()
        self.num_classes = num_classes
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = torch.nn.Linear(2048, 512)
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.5)
        self.fc = torch.nn.Linear(512, 128)

    def forward(self, x):
        pre_x = self.backbone(x[:, 0, :, :, :])
        current_x = self.backbone(x[:, 1, :, :, :])
        pre_x, current_x = pre_x.unsqueeze(1), current_x.unsqueeze(1)
        current_x, _ = self.cross_attention(current_x, pre_x, pre_x)
        x_rep = current_x.squeeze(1)
        return self.fc(x_rep), x_rep

class SmdaNet(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(SmdaNet, self).__init__()
        self.visual_encoder = Visual_encoder(num_classes, dropout)
        self.audial_encoder = Audial_encoder(num_classes, dropout)
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.5)
        self.fc_va = torch.nn.Linear(1536, 384)
        self.fc_av = torch.nn.Linear(1536, 384)
        self.fc_vva = torch.nn.Linear(512, 128)
        self.fc_aav = torch.nn.Linear(512, 128)
        self.fc_vva_aav = torch.nn.Linear(256, num_classes)
        self.unimodal_weight = 0.5

    def forward(self, v_input, a_input):
        v_proj, v_rep = self.visual_encoder(v_input)
        a_proj, a_rep = self.audial_encoder(a_input)
        v_rep, a_rep = v_rep.unsqueeze(1), a_rep.unsqueeze(1)
        va_rep, _ = self.cross_attention(v_rep, a_rep, a_rep)
        av_rep, _ = self.cross_attention(a_rep, v_rep, v_rep)
        va_concat = torch.cat((v_rep, a_rep, va_rep), dim=2).squeeze(1)
        av_concat = torch.cat((v_rep, a_rep, av_rep), dim=2).squeeze(1)
        va_proj = self.fc_va(va_concat)
        av_proj = self.fc_av(av_concat)
        vva_concat = torch.cat((v_proj, va_proj), dim=1).squeeze(1)
        aav_concat = torch.cat((a_proj, av_proj), dim=1).squeeze(1)
        vva_proj = self.fc_vva(vva_concat)
        aav_proj = self.fc_aav(aav_concat)
        vva_aav_concat = torch.cat((vva_proj, aav_proj), dim=1).squeeze(1)
        vva_aav_proj = self.fc_vva_aav(vva_aav_concat)
        return vva_aav_proj