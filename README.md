# Void Filling of Digital Elevation Models with Deep Generative Models

This GitHub repository implements and evaluates the method described in the paper [1], which is an adaptation to the context of Digital Elevation Models (DEMs) from the method DeepFill described in [2]. In addition pre-trained models are provided, as well as the DEMs used for the evaluation of the method.

[1] K. Gavriil, O.J.D. Barrowclough, G. Muntingh, _Void Filling of Digital Elevation Models with Deep Generative Models_, available on the [ArXiv]().

[2] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang, _Generative image inpainting with contextual attention_, in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

---

<p>
<img src="examples/rex01m-min.png" width="24.5%" />
<img src="examples/rex01b-min.png" width="24.5%" />
<img src="examples/rex03m-min.png" width="24.5%" />
<img src="examples/rex03b-min.png" width="24.5%" />
</p>

<p>
<img src="examples/rex09m-min.png" width="24.5%" />
<img src="examples/rex09o-min.png" width="24.5%" />
<img src="examples/rex07m-min.png" width="24.5%" />
<img src="examples/rex07o-min.png" width="24.5%" />
</p>

Selection of results of the DeepDEMFill void filling method for Digital Elevation Models.

---

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
    * If you still have questions (e.g.: What does filelist looks like? How to use multi-gpus? How to do batch testing?), please first search over closed issues. If the problem is not solved, please open a new issue.

## Pretrained models

[Norway Landscape](https://drive.google.com/open?id=1cxV9nQBQm410BPxS9NjecU5dHZeqZcDX) | [Norway Cities](https://drive.google.com/open?id=1iSdQ28W4GDuBUweMXGtYgNu7Z_XQ24eN)

Download the model directories and put them under `model_logs/` directory

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

## Acknowledgements

We adapted the GitHub repository [generative_inpainting](https://github.com/JiahuiYu/generative_inpainting) to the setting of Digital Elevation Models. The open source C++ library [GoTools](https://github.com/SINTEF-Geometry/GoTools) was used for generating the LR B-spline data. Data provided courtesy Norwegian Mapping Authorities (www.hoydedata.no), copyright Kartverket (CC BY 4.0). This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 675789. This projected was also supported by an IKTPLUSS grant, project number 270922, from the Research Council of Norway.

## Citing

Arxiv

```
how to cite
```
