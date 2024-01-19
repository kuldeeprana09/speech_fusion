# wujian@2018
import librosa
from pyparsing import Optional
import torch as th
from torch._C import Size
import torch.nn as nn
from scipy.signal import get_window
import numpy as np
import torch.nn.functional as F
# from convstft import ConvSTFT
import torchaudio
def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        print(x)
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x
# *********************ConvSTFT*********************************
# def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
#     if win_type == 'None' or win_type is None:
#         window = np.ones(win_len)
#     else:
#         window = get_window(win_type, win_len, fftbins=True)#**0.5
    
#     N = fft_len
#     fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
#     real_kernel = np.real(fourier_basis)
#     imag_kernel = np.imag(fourier_basis)
#     kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    
#     if invers :
#         kernel = np.linalg.pinv(kernel).T 

#     kernel = kernel*window
#     kernel = kernel[:, None, :]
#     return th.from_numpy(kernel.astype(np.float32)), th.from_numpy(window[None,:,None].astype(np.float32))
# class ConvSTFT(nn.Module):

#     def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
#         super(ConvSTFT, self).__init__() 
        
#         if fft_len == None:
#             self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
#         else:
#             self.fft_len = fft_len
        
#         kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
#         #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
#         self.register_buffer('weight', kernel)
#         self.feature_type = feature_type
#         self.stride = win_inc
#         self.win_len = win_len
#         self.dim = self.fft_len

#     def forward(self, inputs):
#         if inputs.dim() == 2:
#             inputs = th.unsqueeze(inputs, 1)
#         inputs = F.pad(inputs,[self.win_len-self.stride, self.win_len-self.stride])
#         outputs = F.conv1d(inputs, self.weight, stride=self.stride)
         
#         if self.feature_type == 'complex':
#             return outputs
#         else:
#             dim = self.dim//2+1
#             real = outputs[:, :dim, :]
#             imag = outputs[:, dim:, :]
#             mags = th.sqrt(real**2+imag**2)
#             phase = th.atan2(imag, real)
#             return mags, phase


# *********************************end of STFT******************
def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
        With the reference tensor.
        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
        With the reference tensor and dimension indicator.
        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = th.arange(0, maxlen, dtype=th.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


class Stft(th.nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        # assert check_argument_types()
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(th, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(self, input: th.Tensor, ilens: th.Tensor = None):
        """STFT forward function.
        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)
        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # NOTE(kamo):
        #   The default behaviour of torch.stft is compatible with librosa.stft
        #   about padding and scaling.
        #   Note that it's different from scipy.signal.stft

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(th, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None

        # For the compatibility of ARM devices, which do not support
        # torch.stft() due to the lake of MKL.
        if input.is_cuda or th.backends.mkl.is_available():
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                normalized=self.normalized,
                onesided=self.onesided,
            )
            # if is_torch_1_7_plus:
            #     stft_kwargs["return_complex"] = False
            # output = torch.stft(input, **stft_kwargs)
        else:
            if self.training:
                raise NotImplementedError(
                    "stft is implemented with librosa on this device, which does not "
                    "support the training mode."
                )

            # use stft_kwargs to flexibly control different PyTorch versions' kwargs
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                pad_mode="reflect",
            )

            if window is not None:
                # pad the given window to n_fft
                n_pad_left = (self.n_fft - window.shape[0]) // 2
                n_pad_right = self.n_fft - window.shape[0] - n_pad_left
                stft_kwargs["window"] = th.cat(
                    [th.zeros(n_pad_left), window, th.zeros(n_pad_right)], 0
                ).numpy()
            else:
                win_length = (
                    self.win_length if self.win_length is not None else self.n_fft
                )
                stft_kwargs["window"] = th.ones(win_length)

            output = []
            # iterate over istances in a batch
            for i, instance in enumerate(input):
                stft = librosa.stft(input[i].numpy(), **stft_kwargs)
                output.append(th.tensor(np.stack([stft.real, stft.imag], -1)))
            output = th.stack(output, 0)
            if not self.onesided:
                len_conj = self.n_fft - output.shape[1]
                conj = output[:, 1 : 1 + len_conj].flip(1)
                conj[:, :, :, -1].data *= -1
                output = th.cat([output, conj], 1)
            if self.normalized:
                output = output * (stft_kwargs["window"].shape[0] ** (-0.5))

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (ilens - self.n_fft) // self.hop_length + 1
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    # def inverse(
    #     self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor = None
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     """Inverse STFT.
    #     Args:
    #         input: Tensor(batch, T, F, 2) or ComplexTensor(batch, T, F)
    #         ilens: (batch,)
    #     Returns:
    #         wavs: (batch, samples)
    #         ilens: (batch,)
    #     """
    #     if V(torch.__version__) >= V("1.6.0"):
    #         istft = torch.functional.istft
    #     else:
    #         try:
    #             import torchaudio
    #         except ImportError:
    #             raise ImportError(
    #                 "Please install torchaudio>=0.3.0 or use torch>=1.6.0"
    #             )

    #         if not hasattr(torchaudio.functional, "istft"):
    #             raise ImportError(
    #                 "Please install torchaudio>=0.3.0 or use torch>=1.6.0"
    #             )
    #         istft = torchaudio.functional.istft

    #     if self.window is not None:
    #         window_func = getattr(torch, f"{self.window}_window")
    #         if is_complex(input):
    #             datatype = input.real.dtype
    #         else:
    #             datatype = input.dtype
    #         window = window_func(self.win_length, dtype=datatype, device=input.device)
    #     else:
    #         window = None

    #     if is_complex(input):
    #         input = torch.stack([input.real, input.imag], dim=-1)
    #     elif input.shape[-1] != 2:
    #         raise TypeError("Invalid input type")
    #     input = input.transpose(1, 2)

    #     wavs = istft(
    #         input,
    #         n_fft=self.n_fft,
    #         hop_length=self.hop_length,
    #         win_length=self.win_length,
    #         window=window,
    #         center=self.center,
    #         normalized=self.normalized,
    #         onesided=self.onesided,
    #         length=ilens.max() if ilens is not None else ilens,
    #     )

    #     return wavs, ilens
# **************************************************************
class ConvSTFT(nn.Module):
    """STFT encoder for speech enhancement and separation"""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        use_builtin_complex: bool = True,
    ):
        super().__init__()
        self.stft = th.stft(n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self._output_dim = n_fft // 2 + 1 if onesided else n_fft
        self.use_builtin_complex = use_builtin_complex

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: th.Tensor, ilens: th.Tensor):
        """Forward.
        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        """
        spectrum, flens = self.stft(input, ilens)
        if self.feature_type == 'complex':
            return input
        else:
            dim = self.dim//2+1
            real = input[:, :dim, :]
            imag = input[:, dim:, :]
            mags = th.sqrt(real**2+imag**2)
            phase = th.atan2(imag, real)
            return mags, phase
        
        # if is_th_1_9_plus and self.use_builtin_complex:
        #     spectrum = th.complex(spectrum[..., 0], spectrum[..., 1])
        # else:
        #     spectrum = ComplexTensor(spectrum[..., 0], spectrum[..., 1])

        # return spectrum, flens


# *********************************end of STFT******************

class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                in_channels=256,
                conv_channels=512,
                Sc = 512,
                kernel_size=3,
                dilation=1,
                dropout=0.2,
                norm="cLN",
                causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv nxBxT -> nXHXT
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv 
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        # self.dropout = nn.Dropout(dropout, inplace=True)
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        #1x1 conv skip-connection
        self.skip_out = nn.Conv1d(conv_channels, Sc, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        
        if __name__ == "__main__":
            print('1D blick after fist 1x1Conv size', y.size())
            
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        
        if __name__ == "__main__":
            print('1D Conv block after dconv size', y.size())
        
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        out = self.sconv(y)
        skip = self.skip_out(y)
        x = x + out

        return skip, x


#  *******************************************************   
class MS_SL2_split_model(nn.Module):
    def __init__(self,
                L=16, #length of filters(in sample) 
                N=512, #Number of filters in Autoencoder
                X=8, #Number of convolution block in each repeat
                R=1,#Number of repeats
                B=256,#Number of channels in bottleneck and residual paths' 1 by 1-conv blocks
                Sc=256,# number of channels in skip-connection paths' 1 by 1-conv blocks
                Slice=1,
                H=512,#Number of channel in convolution blocks
                P=3, #kernal size in convolution blocks
                norm="cLN",
                num_spks=2,
                non_linear="sigmoid",
                win_len = 256,
                win_inc = 10 ,
                fft_len = 1023,
                dropout=0.2,
                causal=False):
        super(MS_SL2_split_model, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                            format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder_1d = Conv1D(1, N, L, stride=L // 2, padding=0)
        
        # keep T not change
        # T = int((xlen - L) / (L // 2)) + 1
        # before repeat blocks, always cLN
        self.ln = ChannelWiseLayerNorm(N)
        # n x N x T => n x B x T
        self.proj = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.slices = self._build_slices(
            Slice,
            R,
            X,
            Sc=Sc,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        
        #weight for each branch
        self.wList = nn.Parameter(th.tensor([0.5+0.001, 0.5-0.001,0.4+0.001, 0.6-0.001, 0.3+0.001, 0.7-0.001]),requires_grad=True)
        
        self.PRelu = nn.PReLU()
        # output 1x1 conv
        # n x B x T => n x N x T
        # NOTE: using ModuleList not python list
        # self.conv1x1_2 = th.nn.ModuleList(
        #     [Conv1D(B, N, 1) for _ in range(num_spks)])
        # n x Sc x T => n x 2N x T
        self.mask = Conv1D(Sc, num_spks * N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d = ConvTrans1D(
            N, 1, kernel_size=L, stride=L // 2, bias=True)
        self.num_spks = num_spks
        self.R = R #numbers of repeat
        self.X = X #numbers of Conv1Dblock in each repeat
        self.slice = Slice #numbers of slices
    
        
    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)
    
    def _build_slices(self, num_slice, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        slices = [
            self._build_repeats(num_repeats, num_blocks, **block_kwargs)
            for r in range(num_slice)
        ]
        return nn.Sequential(*slices)
    def forward(self, x):
        
        if __name__ == "__main__":
            print('input size', x.size())
        
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
        #encoder
        # n x 1 x S => n x N x T
        print(x.dim())
        w = F.relu(self.encoder_1d(x))
        en_stft = th.stft(x, n_fft=512, hop_length=100, win_length=200)
        atime = th.fft.hfft(abs(en_stft[0:3]), n=124)
        print("value of atime encoder",atime.size())
        print("value of atime encoder",atime)
        
        en_stft = F.relu(atime)
        print("value of stft encoder", en_stft.size())
        print("value of stft encoder", en_stft)
        
        # w = en_1d.permute(1, 0, 2).reshape(en_stft.size())
        # en_stft = en_stft.np()[0]
        # en_stft = th.from_numpy(en_stft)
        # np_inputs = x.tensor.detach().numpy().reshape([-1])
        # w =th.cat([en_1d, en_stft], dim=0)
        # st_1d = (en_1d,en_stft)
        # w = th.cat(st_1d)
        print("value of w encoder", w)
        print("value of w encoder", w.size())
        # w =  en_stft
        if __name__ == "__main__":
            # print('after 1D-encoder size', en_1d.size())
            print('after stft_encoder size', en_stft.size())
            print('after 1D-encoder size', w.size())
        #Seperation
        #   LayerNorm & 1X1 Conv
        # n x B x T
        y = self.proj(self.ln(w))
        
        if __name__ == "__main__":
            print('after LayerNorm and 1x1 Conv', y.size())
        
        #Slices of TCN
        # n x B x T
        
        th.total_connection = 0
        skip_connection = 0
        Slice_input = y
        Tcn_into_weight=[]
        
        Tcn_output_result=0
        for Slice in range(self.slice):
            #print("Value of Slice Number: ",Slice)
            if __name__ == "__main__":
                print('slice input size', y.size())
            
            for i in range(self.R):
                for j in range(self.X):
                    if __name__ == "__main__":
                        print('1D Conv block input size', y.size())
                    
                    skip, y = self.slices[Slice][i][j](y)
                    skip_connection = skip_connection + skip
                    
                    if __name__ == "__main__":
                        print('finished 1D Conv block skip_connection size', skip.size())
                        print('finished 1D Conv block ouput size', y.size())
                    #     print("Weight lenght ",self.wList)
            
            for i in range (len(self.wList)): 
                #if a==0:  
                print("Weight value1", self.wList[i])
                #f = open(filename,'w')
                #print('whatever', file=f)
                total_connection = skip_connection *self.wList[i]
                Tcn_into_weight.append(total_connection)
                #print("lenght of skip connection TCN :", len(skip_connection))
                #print("Weight value1 after append ", W_firstResult)
                total_connection=0
                
        #     print("Weight lenght ",len(self.wList))
        #     if __name__ == "__main__":
        #         print('slice weight last', self.wList[Slice-2])
        
            skip_connection = 0
            y = Slice_input
        for i in range(len(Tcn_into_weight)):
            Tcn_output_result+=Tcn_into_weight[i]
        #print("TcnResult values", Tcn_output_result)
        #print(" Out put of TcnResult shape", Tcn_output_result.size())
        
        y = self.PRelu(Tcn_output_result)
        #print("Output of Tcn size after pRelu",y.size())
        # n x 2N x T
        e = th.chunk(self.mask(y), self.num_spks, 1)
        
        if __name__ == "__main__":
            print('after 1x1 Conv mask)', e[0].size())
        # n x N x T
        if self.non_linear_type == "softmax":
            m = self.non_linear(th.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(th.stack(e, dim=0))
        # spks x [n x N x T]
        s = [w * m[n] for n in range(self.num_spks)]
        # spks x n x S
        return [self.decoder_1d(x, squeeze=True) for x in s]
#***********************End*********************************************************
def foo_conv1d_block():
    nnet = Conv1DBlock(256, 512, 3, 20)
    print(param(nnet))
# ***********************************************
def foo_layernorm():
    C, T = 256, 20
    nnet1 = nn.LayerNorm([C, T], elementwise_affine=True)
    print(param(nnet1, Mb=False))
    nnet2 = nn.LayerNorm([C, T], elementwise_affine=False)
    print(param(nnet2, Mb=False))
# ***********************************************
def SL2_split():
    x = th.rand(2, 1000)
    nnet = MS_SL2_split_model(norm="cLN", causal=False)
    print(nnet)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)

if __name__ == "__main__":
    SL2_split()
    
