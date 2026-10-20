import torch.nn as nn
from models.block.Base import Conv3Relu
import torch
import torch.nn.functional as F
import math

class CotSR(nn.Module):

    def __init__(self, in_dim=96, num_classes=96, max_attention_size=64):
        super(CotSR, self).__init__()
        self.in_dim = in_dim
        self. num_classes= num_classes
        self.max_attention_size = max_attention_size
     
        self.query_conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim //8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim//8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=1)
        
        self.query_conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim //8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim //8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=1)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.attention_scale = 1.0 / math.sqrt(max(self.in_dim // 8, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.classifier1 = nn.Conv2d(self.in_dim, self. num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(self.in_dim, self. num_classes, kernel_size=1)
    
    def forward(self, x1, x2):
        residual_x1, residual_x2 = x1, x2
        m_batchsize, C, height, width = x1.size()
        if max(height, width) > self.max_attention_size:
            scale = self.max_attention_size / float(max(height, width))
            attention_size = (max(1, round(height * scale)), max(1, round(width * scale)))
            x1 = F.interpolate(x1, size=attention_size, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x2, size=attention_size, mode='bilinear', align_corners=True)
            m_batchsize, C, height_att, width_att = x1.size()
        else:
            height_att, width_att = height, width
        
        q1 = self.query_conv1(x1).view(m_batchsize, -1, width_att*height_att).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width_att*height_att)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width_att*height_att)
        
        q2 = self.query_conv2(x2).view(m_batchsize, -1, width_att*height_att).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width_att*height_att)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width_att*height_att)
        
        # Keep scaled attention in FP32 to prevent AMP overflow.
        attention1 = self.softmax(torch.bmm(q1.float(), k2.float()) * self.attention_scale)
        out1 = torch.bmm(v2.float(), attention1.permute(0, 2, 1)).to(x1.dtype)
        out1 = out1.view(m_batchsize, C, height_att, width_att)
                
        attention2 = self.softmax(torch.bmm(q2.float(), k1.float()) * self.attention_scale)
        out2 = torch.bmm(v1.float(), attention2.permute(0, 2, 1)).to(x2.dtype)
        out2 = out2.view(m_batchsize, C, height_att, width_att)

        if (height_att, width_att) != (height, width):
            out1 = F.interpolate(out1, size=(height, width), mode='bilinear', align_corners=True)
            out2 = F.interpolate(out2, size=(height, width), mode='bilinear', align_corners=True)
        
        out1 = residual_x1 + self.gamma1*out1
        out2 = residual_x2 + self.gamma2*out2  

        out1 = self.classifier1(out1)
        out2 = self.classifier2(out2)  
        
        return out1, out2

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=5):
        super().__init__()
        inter_channels = in_channels
        self.head1 = nn.Sequential(Conv3Relu(in_channels, inter_channels),
                                  nn.Dropout(0.2), 
                                  nn.Conv2d(inter_channels, num_classes, (1, 1)))
        self.head2 = nn.Sequential(Conv3Relu(in_channels, inter_channels),
                                  nn.Dropout(0.2), 
                                  nn.Conv2d(inter_channels, num_classes, (1, 1)))
        self.head3 = nn.Sequential(Conv3Relu(in_channels, inter_channels),
                                  nn.Dropout(0.2),  
                                  nn.Conv2d(inter_channels, out_channels, (1, 1)))

    def forward(self, change, change_s1, change_s2):
        return self.head3(change), self.head2(change_s1), self.head1(change_s2)
