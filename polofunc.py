import vapoursynth as vs
from vapoursynth import core
from vsutil import Dither, fallback, get_depth, insert_clip, get_w, get_y, depth, iterate, plane, join, disallow_variable_format, disallow_variable_resolution
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from awsmfunc import bbmod

Range = Union[int, Tuple[Optional[int], Optional[int]]]

def _get_bits(clip: vs.VideoNode, expected_depth: int = 16) -> Tuple[int, vs.VideoNode]:
    """Checks bitdepth, set bitdepth if necessary, and sends original clip's bitdepth back with the clip"""
    bits = get_depth(clip)
    return bits, depth(clip, expected_depth) if bits != expected_depth else clip

def detail_mask(clip: vs.VideoNode,
                sigma: float = 1.0, rxsigma: List[int] = [50, 200, 350],
                pf_sigma: Optional[float] = 1.0,
                rad: int = 3, brz: Tuple[int, int] = (2500, 4500),
                rg_mode: int = 17,
                ) -> vs.VideoNode:
    """
    A detail mask aimed at preserving as much detail as possible within darker areas,
    even if it contains mostly noise.
    """
    bits, clip = _get_bits(clip)

    clip_y = get_y(clip)
    pf = core.bilateral.Gaussian(clip_y, sigma=pf_sigma) if pf_sigma else clip_y
    ret = core.retinex.MSRCP(pf, sigma=rxsigma, upper_thr=0.005)

    blur_ret = core.bilateral.Gaussian(ret, sigma=sigma)
    blur_ret_diff = core.std.Expr([blur_ret, ret], "x y -")
    blur_ret_dfl = core.std.Deflate(blur_ret_diff)
    blur_ret_ifl = iterate(blur_ret_dfl, core.std.Inflate, 4)
    blur_ret_brz = core.std.Binarize(blur_ret_ifl, brz[0])
    # blur_ret_brz = core.morpho.Close(blur_ret_brz, size=8)

    prewitt_mask = core.std.Prewitt(clip_y).std.Binarize(brz[1])
    prewitt_ifl = prewitt_mask.std.Deflate().std.Inflate()
    prewitt_brz = core.std.Binarize(prewitt_ifl, brz[1])
    # prewitt_brz = core.morpho.Close(prewitt_brz, size=4)

    merged = core.std.Expr([blur_ret_brz, prewitt_brz], "x y +")
    rm_grain = core.rgvs.RemoveGrain(merged, rg_mode)
    return rm_grain if bits == 16 else depth(rm_grain, bits)

def masked_f3kdb(clip: vs.VideoNode,
                 rad: int = 16,
                 thr: Union[int, List[int]] = 24,
                 grain: Union[int, List[int]] = [12, 0],
                 mask_args: Dict[str, Any] = {}
                 ) -> vs.VideoNode:
    """Basic f3kdb debanding with detail mask"""
    from debandshit import dumb3kdb

    deb_mask_args: Dict[str, Any] = dict(brz=(1000, 2750))
    deb_mask_args |= mask_args

    bits, clip = _get_bits(clip)

    deband_mask = detail_mask(clip, **deb_mask_args)

    deband = dumb3kdb(clip, radius=rad, threshold=thr, grain=grain, seed=69420)
    deband_masked = core.std.MaskedMerge(deband, clip, deband_mask)
    deband_masked = deband_masked if bits == 16 else depth(deband_masked, bits)
    return deband_masked

def taa(clip: vs.VideoNode, aafun: Callable[[vs.VideoNode], vs.VideoNode]) -> vs.VideoNode:
    """
    Perform transpose AA.
    :param clip:   Input clip.
    :param aafun:  Antialiasing function
    :return:       Antialiased clip
    """
    if clip.format is None:
        raise ValueError("taa: 'Variable-format clips not supported'")

    y = get_y(clip)

    aa = aafun(y.std.Transpose())
    aa = aa.resize.Spline36(height=clip.width, src_top=0.5).std.Transpose()
    aa = aafun(aa)
    aa = aa.resize.Spline36(height=clip.height, src_top=0.5)

    return aa if clip.format.color_family == vs.GRAY \
        else core.std.ShufflePlanes([aa, clip], planes=[0, 1, 2], colorfamily=vs.YUV)

def nnedi3(clip: vs.VideoNode, opencl: bool = False, **override: Any) -> vs.VideoNode:
    """
    Standard nnedi3 antialiasing.
    :param clip:     Input clip
    :param opencl:   Use OpenCL (Default: False)
    :param override: nnedi3 parameter overrides
    :return:         Antialiased clip
    """
    nnedi3_args: Dict[str, Any] = dict(field=0, dh=True, nsize=3, nns=3, qual=1)
    nnedi3_args.update(override)

    def _nnedi3(clip: vs.VideoNode) -> vs.VideoNode:
        return clip.nnedi3cl.NNEDI3CL(**nnedi3_args) if opencl \
            else clip.nnedi3.nnedi3(**nnedi3_args)

    return taa(clip, _nnedi3)

@disallow_variable_format
@disallow_variable_resolution
def dumb3kdb(clip: vs.VideoNode, radius: int = 16,
             threshold: Union[int, List[int]] = 30, grain: Union[int, List[int]] = 0,
             sample_mode: int = 2, use_neo: bool = False, **kwargs) -> vs.VideoNode:
    """
        "f3kdb but better".
        Both f3kdb and neo_f3kdb actually change strength at 1 + 16 * n for sample_mode=2
        and 1 + 32 * n for sample_mode=1, 3 or 4. This function is aiming to average n and n + 1 strength
        for a better accuracy.
        Original function written by Z4ST1N, modified by Vardë.
        https://f3kdb.readthedocs.io/en/latest/index.html
        https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb

    Args:
        clip (vs.VideoNode): Source clip.

        radius (int, optional):
            Banding detection range. Defaults to 16.

        threshold (Union[int, List[int]], optional):
            Banding detection threshold(s) for planes.
            If difference between current pixel and reference pixel is less than threshold,
            it will be considered as banded. Defaults to 30.

        grain (Union[int, List[int]], optional):
            Specifies amount of grains added in the last debanding stage. Defaults to 0.

        sample_mode (int, optional):
            Valid modes are:
                – 1: Take 2 pixels as reference pixel. Reference pixels are in the same column of current pixel.
                – 2: Take 4 pixels as reference pixel. Reference pixels are in the square around current pixel.
                – 3: Take 2 pixels as reference pixel. Reference pixels are in the same row of current pixel.
                – 4: Arithmetic mean of 1 and 3.
            Reference points are randomly picked within the range. Defaults to 2.

        use_neo (bool, optional): Use neo_f3kdb.Deband. Defaults to False.

    Returns:
        vs.VideoNode: Debanded clip.
    """

    # neo_f3kdb nukes frame props
    def _trf(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:
        (fout := f[0].copy()).props.update(f[1].props)
        return fout

    if sample_mode > 2 and not use_neo:
        raise ValueError('dumb3kdb: "sample_mode" argument should be less or equal to 2 when "use_neo" is false.')


    thy, thcb, thcr = [threshold] * 3 if isinstance(threshold, int) else threshold + [threshold[-1]] * (3 - len(threshold))
    gry, grc = [grain] * 2 if isinstance(grain, int) else grain + [grain[-1]] * (2 - len(grain))


    step = 16 if sample_mode == 2 else 32
    f3kdb = core.neo_f3kdb.Deband if use_neo else core.f3kdb.Deband


    f3kdb_args: Dict[str, Any] = dict(keep_tv_range=True, output_depth=16)
    f3kdb_args.update(kwargs)

    if thy % step == 1 and thcb % step == 1 and thcr % step == 1:
        deband = f3kdb(clip, radius, thy, thcb, thcr, gry, grc, sample_mode, **f3kdb_args)
    else:
        loy, locb, locr = [(th - 1) // step * step + 1 for th in [thy, thcb, thcr]]
        hiy, hicb, hicr = [lo + step for lo in [loy, locb, locr]]

        lo_clip = f3kdb(clip, radius, loy, locb, locr, gry, grc, sample_mode, **f3kdb_args)
        hi_clip = f3kdb(clip, radius, hiy, hicb, hicr, gry, grc, sample_mode, **f3kdb_args)

        if clip.format.color_family == vs.GRAY:
            weight = (thy - loy) / step
        else:
            weight = [(thy - loy) / step, (thcb - locb) / step, (thcr - locr) / step]

        deband = core.std.Merge(lo_clip, hi_clip, weight)

def nnedi3_rpow2CL(clip, rfactor=2, width=None, height=None, correct_shift=True,
                 kernel="spline36", nsize=0, nns=3, qual=None, etype=None, pscrn=None,
                 device=-1):
    """nnedi3_rpow2 is for enlarging images by powers of 2.

    Args:
        rfactor (int): Image enlargement factor.
            Must be a power of 2 in the range [2 to 1024].
        correct_shift (bool): If False, the shift is not corrected.
            The correction is accomplished by using the subpixel
            cropping capability of fmtc's resizers.
        width (int): If correcting the image center shift by using the
            "correct_shift" parameter, width/height allow you to set a
            new output resolution.
        kernel (string): Sets the resizer used for correcting the image
            center shift that nnedi3_rpow2 introduces. This can be any of
            fmtc kernels, such as "cubic", "spline36", etc.
            spline36 is the default one.
        nnedi3_args (mixed): For help with nnedi3 args
            refert to nnedi3 documentation.
    """

    # Setting up variables

    if width is None:
        width = clip.width*rfactor
    if height is None:
        height = clip.height*rfactor
    hshift = 0.0
    vshift = -0.5
    pkdnnedi = dict(dh=True, nsize=nsize, nns=nns, qual=qual, etype=etype,
                    pscrn=pscrn, device=device)
    pkdchroma = dict(kernel=kernel, sy=-0.5, planes=[2, 3, 3])

    tmp = 1
    times = 0
    while tmp < rfactor:
        tmp *= 2
        times += 1

    # Checks

    if rfactor < 2 or rfactor > 1024:
        raise ValueError("nnedi3_rpow2: rfactor must be between 2 and 1024")

    if tmp != rfactor:
        raise ValueError("nnedi3_rpow2: rfactor must be a power of 2")

    if hasattr(core, 'nnedi3') is not True:
        raise RuntimeError("nnedi3_rpow2: nnedi3 plugin is required")

    if correct_shift or clip.format.subsampling_h:
        if hasattr(core, 'fmtc') is not True:
            raise RuntimeError("nnedi3_rpow2: fmtconv plugin is required")

    # Processing

    last = clip

    for i in range(times):
        field = 1 if i == 0 else 0
        last = core.nnedi3cl.NNEDI3CL(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)
        if last.format.subsampling_w:
            # Apparently always using field=1 for the horizontal pass somehow
            # keeps luma/chroma alignment.
            field = 1
            hshift = hshift*2 - 0.5
        else:
            hshift = -0.5
        last = core.nnedi3cl.NNEDI3CL(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)

    # Correct vertical shift of the chroma.

    if clip.format.subsampling_h:
        last = core.fmtc.resample(last, w=last.width, h=last.height, **pkdchroma)

    if correct_shift is True:
        last = core.fmtc.resample(last, w=width, h=height, kernel=kernel,
                                  sx=hshift, sy=vshift)

    if last.format.id != clip.format.id:
        last = core.fmtc.bitdepth(last, csp=clip.format.id)

    return last

def nnedi3_rpow2(clip, rfactor=2, width=None, height=None, correct_shift=True,
                 kernel="spline36", nsize=0, nns=3, qual=None, etype=None, pscrn=None,
                 opt=True, int16_prescreener=None, int16_predictor=None, exp=None):
    """nnedi3_rpow2 is for enlarging images by powers of 2.

    Args:
        rfactor (int): Image enlargement factor.
            Must be a power of 2 in the range [2 to 1024].
        correct_shift (bool): If False, the shift is not corrected.
            The correction is accomplished by using the subpixel
            cropping capability of fmtc's resizers.
        width (int): If correcting the image center shift by using the
            "correct_shift" parameter, width/height allow you to set a
            new output resolution.
        kernel (string): Sets the resizer used for correcting the image
            center shift that nnedi3_rpow2 introduces. This can be any of
            fmtc kernels, such as "cubic", "spline36", etc.
            spline36 is the default one.
        nnedi3_args (mixed): For help with nnedi3 args
            refert to nnedi3 documentation.
    """
    core = vs.core

    # Setting up variables

    if width is None:
        width = clip.width*rfactor
    if height is None:
        height = clip.height*rfactor
    hshift = 0.0
    vshift = -0.5
    pkdnnedi = dict(dh=True, nsize=nsize, nns=nns, qual=qual, etype=etype,
                    pscrn=pscrn, opt=opt, int16_prescreener=int16_prescreener,
                    int16_predictor=int16_predictor, exp=exp)
    pkdchroma = dict(kernel=kernel, sy=-0.5, planes=[2, 3, 3])

    tmp = 1
    times = 0
    while tmp < rfactor:
        tmp *= 2
        times += 1

    # Checks

    if rfactor < 2 or rfactor > 1024:
        raise ValueError("nnedi3_rpow2: rfactor must be between 2 and 1024")

    if tmp != rfactor:
        raise ValueError("nnedi3_rpow2: rfactor must be a power of 2")

    if hasattr(core, 'nnedi3') is not True:
        raise RuntimeError("nnedi3_rpow2: nnedi3 plugin is required")

    if correct_shift or clip.format.subsampling_h:
        if hasattr(core, 'fmtc') is not True:
            raise RuntimeError("nnedi3_rpow2: fmtconv plugin is required")

    # Processing

    last = clip

    for i in range(times):
        field = 1 if i == 0 else 0
        last = core.nnedi3.nnedi3(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)
        if last.format.subsampling_w:
            # Apparently always using field=1 for the horizontal pass somehow
            # keeps luma/chroma alignment.
            field = 1
            hshift = hshift*2 - 0.5
        else:
            hshift = -0.5
        last = core.nnedi3.nnedi3(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)

    # Correct vertical shift of the chroma.

    if clip.format.subsampling_h:
        last = core.fmtc.resample(last, w=last.width, h=last.height, **pkdchroma)

    if correct_shift is True:
        last = core.fmtc.resample(last, w=width, h=height, kernel=kernel,
                                  sx=hshift, sy=vshift)

    if last.format.id != clip.format.id:
        last = core.fmtc.bitdepth(last, csp=clip.format.id)

    return last
