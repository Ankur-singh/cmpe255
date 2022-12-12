# Semantic Segmentation - 255 final project

## Setup

1. Git clone this repo

```bash
git clone
```

2. Create new conda environment and activate it

```bash
conda create -n cmpe255
conda activate cmpe255
```

3. Install python dependencies

```bash
pip install -r requirements.txt
```

4. Setup up Wandb

To enable wandb tracking, its important that you login to wandb using the following command

```bash
wandb login
```

This should open a new tab in your browser. Just copy if the token key and paste it. 

## Usage

There are only two important files:

- `utils.py` : it has all the utility functions for creating custom learner, dynamically loading SMP architectures, getting model size, params, and flops.
- `train.py` : this is the main script responsible for loading the cityscapes dataset, and training the segmentation models.

Here is a list of arguments that it takes:
- `encoder` : The CNN backbone that is to be used. You can use any encoder from SMP library. Default : `mit_b2`
- `arch` : The Segmentation architecture that is to be used. Any architecture from SMP library. Default : `smp.MAnet`
- `epochs` : Number of epochs to train for in Stage-2 (i.e. complete model). Default : `5`
- `project` : wandb project for tracking. Default : `255_final_project`
- `bs` : Batch size. Default : `8`

Here an examples to highlight the usage of `train.py` script:

```bash
python train.py arch=smp.Unet encoder=timm-efficientnet-b5 bs=16 epochs=10
```

### Notes:

- `epochs` argument : The model training is divided into two parts: Stage-1 and Stage-2. In Stage-1, only the decoder and head block will be trained for 2 epochs. Then in Stage-2, all the three blocks (i.e. encoder, decoder, and head) will be trained at half the initial learning rate of 0.001. The `epochs` argument in training script is used to set number of epochs in Stage-2.

- `arch` argument : One can use any segmentation model from SMP library. There are around 9 possible options to select from. Instead of writing if-else block for each, I am loading the segmentation model class dynamically using `load_obj` function from `utils.py` file. Hence, it is important that you append `smp.` before you architecture name because I am importing `segmentation_models_pytorch` as `smp` inside `utils.py`

- I am using OneCycleLR scheduler because it tends to converge much quickly. Hence, I am training the models for 5 epochs only. 

### Makefile

You can also use the `make` command to train all the models. 

```bash
make -f Makefile
```

To train only one class of models, run the following command. Possible model classes are `unet`, `unetpp`, `fpn`, `pspnet`, and `manet`.
```bash
make -f Makefile <model_class>
```

## Results:

Here are all the details collected during the training process. W&B dashboard can be found [here](https://wandb.ai/ankursingh/255_final_project).


| Model                           | Train Loss | Valid Loss | Runtime | Model Size | Params    | Flops    |
| ------------------------------- | ----------:| ----------:| -------:| ----------:| ---------:| --------:|
| MAnet\_resnext50\_32x4d         | 0.583      | 0.625      | 10m 58s | 560.745 MB | 146.917 M | 19.177 G |
| MAnet\_mit\_b2                  | 0.545      | 0.588      | 8m 26s  | 133.443 MB | 34.977 M  | 7.582 G  |
| MAnet\_timm-efficientnet-b5     | 0.723      | 0.7323     | 10m 24s | 146.462 MB | 37.164 M  | 6.480 G  |
| MAnet\_resnet50                 | 0.6075     | 0.6348     | 9m 10s  | 562.702 MB | 147.445 M | 18.975 G |
| PSPNet\_resnext50\_32x4d        | 0.6011     | 0.6744     | 6m 13s  | 91.552 MB* | 2.362 M   | 3.194 G  |
| PSPNet\_mit\_b2                 | 0.6333     | 0.6424     | 6m 55s  | 93.489 MB  | 24.506 M  | 4.433 G  |
| PSPNet\_timm-efficientnet-b5    | 0.8233     | 0.8041     | 6m 41s  | 109.661 MB | 0.6912 M* | 1.150 G* |
| PSPNet\_resnet50                | 0.6384     | 0.691      | 6m 05s* | 93.509 MB  | 2.395 M   | 3.150 G  |
| FPN\_resnet50\_32x4d            | 0.623      | 0.6522     | 7m 50s  | 97.888 MB  | 25.590 M  | 8.089 G  |
| FPN\_timm-efficientnet-b5       | 0.7048     | 0.7219     | 9m 43s  | 115.760 MB | 29.118 M  | 5.200 G  |
| FPN\_resnet50                   | 0.6209     | 0.6576     | 26m 15s | 99.845 MB  | 26.119 M  | 7.887 G  |
| UnetPlusPlus\_resnext50\_32x4d  | 0.4779*    | 0.5307*    | 12m 31s | 185.157 MB | 48.463 M  | 58.084 G |
| UnetPlusPlus\_timm-efficient-b5 | 0.6652     | 0.6769     | 11m 30s | 122.418 MB | 30.862 M  | 11.781 G |
| UnetPlusPlus\_resnet50          | 0.4832     | 0.5404     | 10m 46s | 187.114 MB | 48.991 M  | 57.882 G |
| Unet\_resnext50\_32x4d          | 0.567      | 0.6115     | 8m 37s  | 122.331 MB | 31.998 M  | 11.229 G |
| Unet\_mit\_b2                   | 0.5218     | 0.5764     | 8m 03s  | 104.843 MB | 27.482 M  | 7.208 G  |
| Unet\_timm-efficientnet-b5      | 0.694      | 0.7165     | 9m 52s  | 119.766 MB | 30.168 M  | 6.245 G  |
| Unet\_resnet50                  | 0.5822     | 0.6122     | 6m 59s  | 124.288 MB | 32.526 M  | 11.028 G |


Here are some insights from the table

- 


## Fastai + SMP

Integrating Fastai with SMP was the most fun and challenging part. In the process, I ended up learning more about the Fastai Learner and its internals. 

I created a custom `seg_learner` function which is inspired by `unet_learner`. 

```python
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
    assert (n_out), "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    
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

```

As you can see, it takes `arch` and `encoder` as input and created a SMP model. There are some other functions that had to be updated to make it work like `segment_split`, `Learner.fine_tune`  method, etc. 



## References

- Fastai `Learner` class : https://github.com/fastai/fastai/blob/master/fastai/learner.py#L97
- SMP `create_model` function : https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/\_\_init\_\_.py#L23
- [Keypoint regression with heatmaps in fastai v2](https://elte.me/2021-03-10-keypoint-regression-fastai) To better understand fastai internals, I used this as inspiration to learn and make fastai operate with other libraries and other problem types.