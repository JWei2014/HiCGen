import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math
from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def get_positional_features_exponential(
    positions, features, seq_len, min_half_life=3.0
):
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(
        min_half_life, max_range, features, device=positions.device
    )
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = (
        2 ** torch.arange(1, features + 1, device=positions.device).float()
    )
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(
        concentration
    ) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(
    positions, features, seq_len, stddev=None, start_mean=None, eps=1e-8
):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(
        start_mean, seq_len, features, device=positions.device
    )
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = gamma_pdf(
        positions.float().abs()[..., None], concentration, rate
    )
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)
    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma,
    ]
    if feature_size % 6 == 0:
        feature_functions = feature_functions
    elif feature_size % 4 == 0:
        feature_functions = feature_functions[:2]
    else:
        feature_functions = feature_functions[:1]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size {feature_size} is not divisible"
            f"by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))
    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings


def relative_shift_swin(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    h, _, window, t1, t2 = x.shape         
    x = x.reshape(h, -1, window, t2, t1)
    x = x[:, :, :, 1:, :]
    x = x.reshape(h, -1, window, t1, t2 - 1)
    # up to this point: (i,j) in x represets the emb of dot product (i,i-j)
    return x[..., : ((t2 + 1) // 2)]


def create_padding_mask(b, remainder, padding_size, device):
    mask = torch.zeros(
        b, remainder + padding_size, dtype=torch.bool, device=device
    )
    # set the padding part to True
    mask[:, remainder:] = True
    return mask


class CirculateSwinBlock(nn.Module):
    def __init__(
        self,
        stage,
        dim,
        heads=8,
        attn_dim_key=64,
        dropout_rate=0.02,
        dw_conv=True,
        **kwargs,
    ):
        super().__init__()
        transformers = []

        for lay_num, reduce, squeeze, win_size in stage:
            stage_transformers = []  
            for i in range(lay_num):
                reducing = False
                if i == 0:
                    reducing = reduce
                dim = dim * 2 if reducing is True and squeeze is False else dim
                stage_transformers.append(
                    nn.Sequential(
                        CirculateSwinAttention(
                            reducing=reducing,
                            squeeze=squeeze,
                            dim=dim,
                            window_size=win_size,
                            heads=heads,
                            dim_key=attn_dim_key,
                            dropout=dropout_rate,
                            dw_conv=dw_conv,
                            **kwargs,
                        ),
                    )
                )
            transformers.append(nn.Sequential(*stage_transformers))  
        self.transformers = nn.ModuleList(transformers) 

    def forward(self, x):
        outputs = []  
        for transformers_stage in self.transformers:  
            x = transformers_stage(x)  
            outputs.append(x) 
        return outputs  


class CirculateSwinAttention(nn.Module):
    def __init__(self, reducing=False, squeeze=False, **kwargs):
        super().__init__()
        self.reducing = reducing
        self.half_window_size = kwargs["window_size"] // 2
        self.regular_swin = CirculateSwinAttentionHelp(**kwargs)
        self.shift_swin = CirculateSwinAttentionHelp(**kwargs)
        self.squeeze = squeeze
        if squeeze:
            self.linear = nn.Linear(kwargs["dim"] * 2, kwargs["dim"])
        # else:
        #     self.linear =  nn.Linear(kwargs["dim"] * 2, kwargs["dim"] * 2)

    def forward(self, x):
        if self.reducing:
            x0 = x[:, 0::2, :]
            x1 = x[:, 1::2, :]
            if x1.shape[1] < x0.shape[1]:
                to_pad = x0[:, -1, :].unsqueeze(1)
                x1 = torch.cat([x1, to_pad], 1)
            x = torch.cat([x0, x1], -1)
            if self.squeeze:
                x = self.linear(x)

        x = self.regular_swin(x)
        # shift the start of the sequence to the end
        x = torch.roll(x, -self.half_window_size, dims=1)
        x = self.shift_swin(x)
        x = torch.roll(x, self.half_window_size, dims=1)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    

class DWConv(nn.Module):

    def __init__(self, dim, kernel_size=3):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        return x


class CirculateSwinAttentionHelp(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        heads=8,
        dim_key=32,
        dropout=0.02,
        num_rel_pos_features=None,
        dw_conv=False,
        debug=False,
        padding_mode=True,
        window_specific_bias=False,
        seq_len=336,
    ):
        super().__init__()
        swin = SwinAttentionHelp(
            dim,
            window_size,
            heads,
            dim_key,
            dropout,
            num_rel_pos_features,
            debug,
            padding_mode,
            window_specific_bias,
            seq_len,
        )
        self.att = Residual(
            nn.Sequential(nn.LayerNorm(dim), swin, nn.Dropout(dropout))
        )
        fc_layers = [
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        ]
        if dw_conv:
            fc_layers.insert(2, DWConv(dim * 2))
        self.fc = Residual(nn.Sequential(*fc_layers))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Sequential):
            for submodule in m.children():
                self._init_weights(submodule)
        elif isinstance(m, nn.ModuleList):
            for submodule in m:
                self._init_weights(submodule)

    def forward(self, x):
        return self.fc(self.att(x))


class SwinAttentionHelp(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        heads=8,
        dim_key=32,
        dropout=0.0,
        num_rel_pos_features=None,
        debug=False,
        padding_mode=True,
        window_specific_bias=False,
        seq_len=336,
    ):
        super().__init__()
        self.dim_key = dim_key
        self.heads = heads
        self.window_size = window_size

        self.to_qkv = nn.Linear(dim, dim_key * 3 * heads, bias=False)
        self.to_out = nn.Linear(dim_key * heads, dim)

        self.num_rel_pos_features = default(                                ## def default(num_rel_pos_features, d): return num_rel_pos_features if exists(num_rel_pos_features) else d
            num_rel_pos_features, dim_key * heads
        )
        self.rel_pos_embedding = nn.Linear(
            self.num_rel_pos_features, dim_key * heads, bias=False
        )
        self.rel_content_bias = nn.Parameter(
            torch.randn(1, heads, 1, 1, dim_key)
        )
        rel_pos_bias_shape = (
            seq_len * 2 // window_size - 1 if window_specific_bias else 1
        )
        self.rel_pos_bias = nn.Parameter(
            torch.randn(1, heads, rel_pos_bias_shape, 1, dim_key)
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.pos_dropout = nn.Dropout(dropout)

        self.debug = debug
        self.padding_mode = padding_mode
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Sequential):
            for submodule in m.children():
                self._init_weights(submodule)
        elif isinstance(m, nn.ModuleList):
            for submodule in m:
                self._init_weights(submodule)

    def forward(self, x):
        b, n, c = x.shape
        original_n = n
        device = x.device
        remainder = n % self.window_size
        needs_padding = remainder > 0
        assert (
            n >= self.window_size
        ), f"the sequence {n} is too short for the window {self.window_size}"
        if self.padding_mode is False:
            assert needs_padding, (
                f"Sequence length ({n}) should be"
                f"divisibleby the window size ({self.window_size})."
            )
        else:
            if needs_padding:
                padding_size = self.window_size - remainder
                x = F.pad(x, (0, 0, 0, padding_size, 0, 0), value=0)            ## add padding_size at the bottom of the second dim
                mask = create_padding_mask(b, remainder, padding_size, device)  ## mask the last window for output
                n += padding_size
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (h d k) -> k b h n d", h=self.heads, k=3)
        q, k, v = qkv[0], qkv[1], qkv[2]                                        ## shape of q: (b h n d)

        # Create sliding window indices                                         
        window_indices = torch.arange(
            0, n - self.window_size + 1, self.window_size, device=device
        )
        q_windows = q[                                                          ## shape of q_windows: (b h n//window_size window_size dim_key)
            ...,
            window_indices.unsqueeze(-1)
            + torch.arange(self.window_size, device=device).unsqueeze(0),
            :,
        ]

        k_windows = k[
            ...,
            window_indices.unsqueeze(-1)
            + torch.arange(self.window_size, device=device).unsqueeze(0),
            :,
        ]

        v_windows = v[
            ...,
            window_indices.unsqueeze(-1)
            + torch.arange(self.window_size, device=device).unsqueeze(0),
            :,
        ]

        # position
        positions = get_positional_embed(           
            self.window_size, self.num_rel_pos_features, device
        )
        positions = self.pos_dropout(positions)
        rel_k = self.rel_pos_embedding(positions)   
        rel_k = rearrange(
            rel_k, "n (h d) -> h n d", h=self.heads, d=self.dim_key
        )
        # original rel_k is (h,windowSize, dimKey)
        # duplicate the rel_K for each window it should have shape
        # (h,numWindows,windowSize, dimKey)
        rel_k = rel_k.unsqueeze(1).repeat(1, q_windows.shape[2], 1, 1)

        k_windows = k_windows.transpose(-2, -1)      ## new shape of k_windows: (b h numWindows dim_key window_size)
        content_attn = torch.matmul(                 ## shape of content_attn: (b h numWindows window_size window_size)  
            q_windows + self.rel_content_bias,       ## self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, 1, dim_key))
            k_windows
            # q_windows,
            # k_windows,
        ) * (self.dim_key**-0.5)

        # calculate position attention         
        rel_k = rel_k.transpose(-2, -1)             ## new shape of rel_k: (h, numWindows, dim_key, window_size)

        rel_logits = torch.matmul(                  ## shape of rel_logits: (b h numWindows window_size window_size) 
            q_windows + self.rel_pos_bias,          ## self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, 1, dim_key))
            rel_k
            # q_windows,
            # rel_k,
        )
        # reshape position_attn to (b, h, n, w, w)
        position_attn = relative_shift_swin(rel_logits)     # shape of position_attn: (b h numWindows window_size (window_size+1)//2)  

        attn = content_attn + position_attn
        if needs_padding:
            mask_value = -torch.finfo(attn.dtype).max
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn[:, :, -1, :, :] = attn[:, :, -1, :, :].masked_fill(
                mask, mask_value
            )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v_windows)           #   (b h numWindows window_size window_size) * (b h numWindows window_size dim_key)  --> (b h n w d)
        out = rearrange(out, "b h w n d -> b w n (h d)")   
        out = self.to_out(out)           #  b n w (h dim_key) -> b n w dim
        out = self.proj_dropout(out)

        out = rearrange(out, "b w n d -> b (w n) d")
        return out[:, :original_n]
