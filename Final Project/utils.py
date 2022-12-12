from fastai.vision.all import *
from thop import profile, clever_format
import importlib
import segmentation_models_pytorch as smp


@patch
@delegates(Learner.fit_one_cycle)
def fine_tune(
    self: Learner,
    epochs,
    base_lr=2e-3,
    freeze_to=-2,
    freeze_epochs=1,
    lr_mult=100,
    pct_start=0.3,
    div=5.0,
    **kwargs
):
    "Fine tune with `Learner.freeze` for `freeze_epochs`, then with `Learner.unfreeze` for `epochs`, using discriminative LR."
    self.freeze_to(freeze_to)
    self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    base_lr /= 2
    self.unfreeze()
    self.fit_one_cycle(
        epochs,
        slice(base_lr / lr_mult, base_lr),
        pct_start=pct_start,
        div=div,
        **kwargs
    )


def segment_split(m):
    return L(m.encoder, m.decoder, m.segmentation_head).map(params)


def _smp_norm(dls, cfg):
    if not dls.after_batch.fs.filter(risinstance(Normalize)):
        tfm = Normalize.from_stats(cfg["mean"], cfg["std"])
        dls.add_tfms([tfm], "after_batch")


def seg_learner(
    dls,
    arch,
    encoder,
    weights,
    normalize=True,
    n_out=None,
    freeze=-2,
    config=None,
    pretrained=True,
    loss_func=None,
    opt_func=Adam,
    lr=defaults.lr,
    splitter=segment_split,
    cbs=None,
    metrics=None,
    path=None,
    model_dir="models",
    wd=None,
    wd_bn_bias=False,
    train_bn=True,
    moms=(0.95, 0.85, 0.95),
    **kwargs
):

    cfg = smp.encoders.get_preprocessing_params(encoder, weights)
    n_in = kwargs["n_in"] if "n_in" in kwargs else 3
    if normalize:
        _smp_norm(dls, cfg)

    n_out = ifnone(n_out, get_c(dls))
    assert (
        n_out
    ), "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    model = arch(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=n_in,
        classes=n_out,
    )

    learn = Learner(
        dls=dls,
        model=model,
        loss_func=loss_func,
        opt_func=opt_func,
        lr=lr,
        splitter=splitter,
        cbs=cbs,
        metrics=metrics,
        path=path,
        model_dir=model_dir,
        wd=wd,
        wd_bn_bias=wd_bn_bias,
        train_bn=train_bn,
        moms=moms,
    )
    if pretrained:
        learn.freeze_to(freeze)
    # keep track of args for loggers
    store_attr("encoder,normalize,n_out,pretrained,cfg", self=learn, **kwargs)
    return learn


# https://discuss.pytorch.org/t/finding-model-size/130275/2
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))
    return size_all_mb


def get_thop(model):
    dummy_input = torch.randn(1, 3, 128, 256).cuda()
    macs, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([macs * 2, params], "%.3f")
    return flops, params


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)
