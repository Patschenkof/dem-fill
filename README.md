# Void Filling of Digital Elevation Models with Deep Generative Models

[Our ArXiv submission]()

Examples

<p>
<img src="examples/rex01m-min.png" width="25%" />
<img src="examples/rex01b-min.png" width="25%" />
<img style="float:right;" src="examples/rex03m-min.png" width="25%" />
<img style="float:right;" src="examples/rex03b-min.png" width="25%" />
</p>

<p>
<img src="examples/rex09m-min.png" width="24%" />
<img src="examples/rex09o-min.png" width="24%" />
<img style="float:right;" src="examples/rex07m-min.png" width="24%" />
<img style="float:right;" src="examples/rex07o-min.png" width="24%" />
</p>

Description of examples

## Run

0. Requirements:
    * Install python3.
    * Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).
    * Install tensorflow toolkit [neuralgym](https://github.com/konstantg/neuralgym) (run `pip install git+https://github.com/konstantg/neuralgym`).
1. Training:
    * Prepare training images filelist and shuffle it ([example](https://github.com/JiahuiYu/generative_inpainting/issues/15)).
    * Modify [inpaint.yml](/inpaint.yml) to set DATA_FLIST, LOG_DIR, IMG_SHAPES and other parameters.
    * Run `python train.py`.
2. Resume training:
    * Modify MODEL_RESTORE flag in [inpaint.yml](/inpaint.yml). E.g., MODEL_RESTORE: 20180115220926508503_places2_model.
    * Run `python train.py`.
3. Testing:
    * Run `python test.py --image examples/input.png --mask examples/mask.png --output examples/output.png --checkpoint model_logs/your_model_dir`.
4. Still have questions?
    * If you still have questions (e.g.: How filelist looks like? How to use multi-gpus? How to do batch testing?), please first search over closed issues. If the problem is not solved, please open a new issue.

## Pretrained models

[Norway Landscape]() | [Norway Cities]()

Description of pretrained models (input resolution and void size). Examples of usage.

```bash
# Norway Landscape
python test.py --image input.tif --mask mask.png --output output.tif (or png) --checkpoint_dir model_logs/checkpoint/
# Norway Cities
python test.py --image input.tif --mask mask.png --output output.tif (or png) --checkpoint_dir model_logs/checkpoint/
```

## License

CC 4.0 Attribution-NonCommercial International

The software is for educational and academic research purpose only.

# Additional comments

## Citing

```
how to cite
```
