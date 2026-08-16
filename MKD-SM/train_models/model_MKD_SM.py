import torch
from torch import nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import os
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
    
class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

	    
class Downsample_flatten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample_flatten, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        # import pdb;pdb.set_trace()
        out = self.conv(x).contiguous()  # B H*W C
        return out
    

class Upsample_flatten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample_flatten, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.deconv(x).contiguous() # B H*W C
        return out


######################################################################################
################################## multi SM fusion Block ##############################
class MultiSM_Fusion(nn.Module):
    def __init__(self, num_feat=32, num_frame=3, center_frame_idx=0):
        super(MultiSM_Fusion, self).__init__()

        '''
        # Compuate the attention map, highlight distinctions while keep similarities
        Input: Aligned frames, [B, T, C, H, W]
        Output: Fused frame, [B, C, H, W]
        '''
        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention
        self.downsample1 = Downsample_flatten(num_feat, num_feat*2)
        self.upsample1 = Upsample_flatten(num_feat*2, num_feat)

        n_resblocks = 2
        conv = default_conv
        f_res_block1 = [ResBlock(conv, num_feat, kernel_size=3) for _ in range(n_resblocks)]
        f_res_block2 = [ResBlock(conv, num_feat*2, kernel_size=3) for _ in range(n_resblocks)]
        f_res_block3 = [ResBlock(conv, num_feat*2, kernel_size=3) for _ in range(n_resblocks)]
        
        f_fusion_tail = [conv(num_feat*2, num_feat, kernel_size=3)]

        self.res_block1 = nn.Sequential(*f_res_block1)
        self.res_block2 = nn.Sequential(*f_res_block2)
        self.res_block3 = nn.Sequential(*f_res_block3)
        
        self.fusion_tail = nn.Sequential(*f_fusion_tail)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_feat):
        
        b, t, c, h, w = aligned_feat.size()
        
        # attention map, highlight distinctions while keep similarities
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # [b,t,c,h,w]

        corr_diff = []
        corr_l = []
        
        # (Fi - F1) * F1
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]    # [b,c,h,w]
            corr = torch.sum(emb_neighbor * embedding_ref, 1).unsqueeze(1)  # [b,1,h,w]
            corr_l.append(corr)
            if i == 0:
                continue
            else:
                # compute the difference among each frame and the base frame
                corr_difference = torch.abs(corr_l[i] - corr_l[0])
                corr_diff.append(corr_difference)

        # compute the attention map
        corr_prob = torch.sigmoid(torch.cat(corr_diff, dim=1))  # [b,t-1,h,w]
        corr_prob = corr_prob.unsqueeze(2).expand(b, t-1, c, h, w)  # [b,t-1,c,h,w]
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # [b,(t-1)*c,h,w]
        
        aligned_oth_feat = aligned_feat[:, 1 : t, :, :, :]
        aligned_oth_feat = aligned_oth_feat.view(b, -1, h, w) * corr_prob
        aligned_feat_guided = torch.zeros(b, t*c, h, w).to(device)
        aligned_feat_guided[:, 0 : c, :, :] = aligned_feat[:, 0 : 1, :, :, :].view(b, -1, h, w)
        aligned_feat_guided[:, c : t*c, :, :] = aligned_oth_feat

        feat = self.lrelu(self.feat_fusion(aligned_feat_guided))  # [b,c,h,w]

        feat_res1 = self.res_block1(feat)
        feat_out = feat_res1 + feat

        return feat_out


############################################################################
################################## multi SM fusion Block ###################
class Extrac_deep_feature(nn.Module):
    def __init__(self, num_feat=32):
        super(Extrac_deep_feature, self).__init__()

        '''
        # Compuate the attention map, highlight distinctions while keep similarities
        Input: Aligned frames, [B, T, C, H, W]
        Output: Fused frame, [B, C, H, W]
        '''
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # spatial attention
        self.downsample1 = Downsample_flatten(num_feat, num_feat*2)
        self.upsample1 = Upsample_flatten(num_feat*2, num_feat)

        n_resblocks = 2
        conv = default_conv
        f_res_block1 = [ResBlock(conv, num_feat, kernel_size=3) for _ in range(n_resblocks)]
        f_res_block2 = [ResBlock(conv, num_feat*2, kernel_size=3) for _ in range(n_resblocks)]
        f_res_block3 = [ResBlock(conv, num_feat*2, kernel_size=3) for _ in range(n_resblocks)]

        f_fusion_tail = [conv(num_feat*2, num_feat, kernel_size=3)]

        self.res_block1 = nn.Sequential(*f_res_block1)
        self.res_block2 = nn.Sequential(*f_res_block2)
        self.res_block3 = nn.Sequential(*f_res_block3)

        self.fusion_tail = nn.Sequential(*f_fusion_tail)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_feat):

        b, c, h, w = aligned_feat.size()

        # attention map, highlight distinctions while keep similarities
        embedding_ref = self.temporal_attn1(aligned_feat.clone())
        feat = self.lrelu(embedding_ref)  # [b,c,h,w]

        feat_res1 = self.res_block1(feat)
        feat_out = feat_res1 + feat
        return feat_out


######################################################################################
################################## transformer encoder ##############################
class Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=20,
                 patch_size=2,
                 feat_embed=32,
                 embed_dim=96,
                 num_layer=6,
                 num_head=4,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patchs_num = (image_size // patch_size) ** 2
        self.pos_embedding = torch.nn.Parameter(torch.zeros(self.patchs_num, 1, embed_dim))

        self.patchify = torch.nn.Conv2d(feat_embed, embed_dim, patch_size, patch_size)
        self.transformer = torch.nn.Sequential(*[Block(embed_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')

        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = features[:, 1:, :]

        return features

#######################################################################################
################################## upsampling for Reconstruction ########################


class PixelShuffle2d(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        # (b, c, H, W) -> (b, c/r^2, r*H, r*W)
        # r is scale factor, c % r^2 is required
        # print('input shape', input.shape)
        batch_size, channels, in_height, in_width = input.size()
        assert channels % (self.scale ** 2) == 0
        nOut = channels // self.scale ** 2
        out_height = in_height * self.scale
        out_width = in_width * self.scale
        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, in_height, in_width)
        output = input_view.permute(0, 1, 4, 2, 5, 3,).contiguous()
        return output.view(batch_size, nOut, out_height, out_width)


class UpSampler(nn.Module):
    def __init__(self, scale, in_channel, out_channel, kernel_size, n_conv):
        super().__init__()
        self.upsampling = nn.Sequential(
            PixelShuffle2d(scale=scale)
        )
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2),
            nn.PReLU(),
            nn.BatchNorm2d(out_channel),
        ])
        for _ in range(n_conv-1):
            self.convs.append(
                nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size,
                                        stride=1, padding=(kernel_size - 1) // 2),
                              nn.PReLU(),
                              nn.BatchNorm2d(out_channel),
                              ))

    def forward(self, img, additional_emb=None):
        # img shape (b, dim, H, W)
        # dim is considered as channels
        # up-sampling (b, dim,  H, W) -> (b, dim, r*H, r*W)
        x = self.upsampling(img)
        if additional_emb is not None:
            for emb in additional_emb:
                x = torch.cat([x, emb], 1)

        for step, conv in enumerate(self.convs):
            if step == 0:
                x = conv(x)
            else:
                x = conv(x) + x
        return x


class Reso_decoder(torch.nn.Module):
    def __init__(self,
                 image_size=20,
                 patch_size=2,
                 scale=2,
                 embed_dim=96,
                 kernel_size=3,
                 n_conv=4,
                 out_channel=64,
                 HR_channel=2,
                 ) -> None:
        super().__init__()
        self.scale = scale
        self.token_size = (image_size // patch_size, image_size // patch_size)
        self.token2image = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=self.token_size[0], w=self.token_size[1]),
            PixelShuffle2d(scale=2)
            if patch_size in [2, 4] else nn.Identity()
        )
        if scale == 2:
            in_channel = embed_dim // (scale * patch_size) ** 2
            self.SRdecoder = UpSampler(scale, in_channel, out_channel, kernel_size, n_conv)
        elif scale == 4:
            assert scale == 4
            half_in_channel = embed_dim // (2 * patch_size) ** 2
            in_channel = out_channel // (2 * patch_size) ** 2
            self.half_SRdecoder = UpSampler(2, half_in_channel, out_channel, kernel_size, n_conv)
            self.SRdecoder = UpSampler(2, in_channel, out_channel, kernel_size, n_conv)
            self.half_residual_conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(HR_channel, out_channel, kernel_size=kernel_size,
                          stride=1, padding=(kernel_size - 1) // 2),
                nn.PReLU(),
                nn.BatchNorm2d(out_channel),
            )
        else:
            half_in_channel_1 = embed_dim // (2 * patch_size) ** 2   # 48
            in_channel = out_channel // (2 * patch_size) ** 2      # 16

            self.token_SRdecoder_1 = UpSampler(2, half_in_channel_1, out_channel, kernel_size, n_conv) # 64

            self.token_SRdecoder_2 = UpSampler(2, in_channel, out_channel, kernel_size, n_conv)

            self.token_SRdecoder_3 = UpSampler(2, in_channel, out_channel, kernel_size, n_conv)

            self.input_SR_1 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(HR_channel, out_channel, kernel_size=kernel_size,
                          stride=1, padding=(kernel_size - 1) // 2),
                nn.PReLU(),
                nn.BatchNorm2d(out_channel),
            )

            self.input_SR_2 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size,
                          stride=1, padding=(kernel_size - 1) // 2),
                nn.PReLU(),
                nn.BatchNorm2d(out_channel),
            )

        self.pred = nn.Conv2d(out_channel, HR_channel, kernel_size=1, stride=1)
        self.residual_conv = nn.Sequential(
            nn.Upsample(scale_factor=scale),
            nn.Conv2d(HR_channel, out_channel, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size-1) // 2),
            nn.PReLU(),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, features, LR_img):

        # print('features shape', features.shape)
        #  (b, h*w, dim) -->   (b, h, w, dim)
        encoded_tokens = self.token2image(features)

        if self.scale == 2:
            decoded_tokens = self.SRdecoder(encoded_tokens)
            # print('decoded_tokens shape', decoded_tokens.shape)
        elif self.scale == 4:
            half_decoded_tokens = self.half_SRdecoder(encoded_tokens) + self.half_residual_conv(LR_img)
            decoded_tokens = self.SRdecoder(half_decoded_tokens)
        else:
            decoded_tokens_SR_x2 = self.token_SRdecoder_1(encoded_tokens)
            input_SR_x2 = self.input_SR_1(LR_img)

            x1 = decoded_tokens_SR_x2 + input_SR_x2
            decoded_tokens_SR_x4 = self.token_SRdecoder_2(x1)
            input_SR_x4 = self.input_SR_2(input_SR_x2)

            x2 = decoded_tokens_SR_x4 + input_SR_x4
            decoded_tokens = self.token_SRdecoder_3(x2)

        pred_HR_SM = self.pred(decoded_tokens + self.residual_conv(LR_img))
        # caculate loss
        return pred_HR_SM

##############################################################################################
################################## model (MultiSMNet) #########################################
class MultiSMNet(nn.Module):
    def __init__(self, in_chans=2, num_frames=3, scale=2, feat_embed=32,
                 img_size=20, patch_size=2, embed_dim=96, num_head=4, num_layer=3,
                 out_channel=64, kernel_size=3, n_conv_layer=3, HR_channel=2,
                 ):
        super(MultiSMNet, self).__init__()
        '''
        # Compuate the attention map, highlight distinctions while keep similarities
        Input: Aligned frames, [B, T, C, H, W]
        Output: Fused frame, [B, C, H, W]
        '''
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.img_size = img_size

        self.patch_size = patch_size
        self.num_layer = num_layer
        self.num_head = num_head
        
        n_resblocks = 2
        conv = default_conv
        
        ##  feature extraction before multi frames feature fusion
        ex_head = [conv(in_chans, feat_embed, kernel_size=3)]
        ex_body = [ResBlock(conv, feat_embed, kernel_size=3) for _ in range(n_resblocks)]
        
        self.head = nn.Sequential(*ex_head)
        self.body = nn.Sequential(*ex_body)

        self.fusion = MultiSM_Fusion(num_feat=feat_embed, num_frame=num_frames, center_frame_idx=0)
        
        self.encoder = Encoder(image_size=img_size, patch_size=patch_size, feat_embed=feat_embed,
                               embed_dim=embed_dim, num_head=num_head, num_layer=num_layer)
        
        self.reso_decoder = Reso_decoder(image_size=img_size, patch_size=patch_size, scale=scale, 
                                         embed_dim=embed_dim, kernel_size=kernel_size, n_conv=n_conv_layer,
                                         out_channel=out_channel, HR_channel=HR_channel,
                                         )
        
    def forward(self, x, mask=None):
        
        b, t, c, h, w = x.size()
        x_base = x[:, 0, :, :, :].contiguous()
        x_feat_head = self.head(x.view(-1, c, h, w))  # [b*t, embed_dim, h, w]
        x_feat_body = self.body(x_feat_head)          # [b*t, embed_dim, h, w]
        
        feature = x_feat_body.view(b, t, -1, h, w)    # [b, t, embed_dim, h, w]
        
        fusion_feature = self.fusion(feature)         # [b, embed_dim, h, w]
        
        trans_feature = self.encoder(fusion_feature)  # [b, embed_dim, h, w]

        pre_HR_SM = self.reso_decoder(trans_feature, x_base)  # [b, 2, h*scal, w h*scal]

        return pre_HR_SM

