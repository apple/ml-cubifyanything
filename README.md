# CA-1M and Cubify Anything

This repository includes the public implementation of Cubify Transformer and the
associated CA-1M dataset (incl. the derived CA-VQA annotations).

## Papers

**Apple**

[Cubify Anything: Scaling Indoor 3D Object Detection](https://arxiv.org/abs/2412.04458)

Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, Afshin Dehghan

**CVPR 2025**

![Teaser](teaser.jpg?raw=true "Teaser")

[MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](https://arxiv.org/abs/2503.13111)

Erik Daxberger, Nina Wenzel, David Griffiths, Haiming Gang, Justin Lazarow, Gefen Kohavi, Kai Kang, Marcin Eichner, Yinfei Yang, Afshin Dehghan, Peter Grasch

**ICCV 2025**

![Teaser (CA-VQA)](teaser_cavqa.png?raw=true "Teaser (CA-VQA)")

## Repository Overview

This repository includes:

1. Links to the underlying data and annotations of the CA-1M dataset (incl. the derived CA-VQA annotations).
2. Links to released models of the Cubify Transformer (CuTR) model from the Cubify Anything paper.
3. Basic readers and inference code to run CuTR on the provided data.
4. Basic support for using images captured from own device using the NeRF Capture app.

## Installation

We recommend Python 3.10 and a recent 2.x build of PyTorch. We include a `requirements.txt` which should encapsulate
all necessary dependencies. Please make sure you have `torch` installed first, e.g.,:

```
pip install torch torchvision
```

Then, within the root of the repository:

```
pip install -r requirements.txt
pip install -e .
```

## CA-1M versus ARKitScenes?

This work is related to [ARKitScenes](https://machinelearning.apple.com/research/arkitscenes). We generally share
the same underlying captures. Some notable differences in CA-1M:

1. Each scene has been exhaustively annotated with class-agnostic 3D boxes. We release these in the laser scanner's coordinate frame.
2. For each frame in each capture, we include "per-frame" 3D box ground-truth which was produced using the rendering
   process outlined in the Cubify Anything paper. These annotations are, therefore, *independent* of any pose.

Some other nice things:

1. We release the GT poses (registered to laser scanner) for every frame in each capture.
2. We release the GT depth (rendered from laser scanner) at 512 x 384 for every frame in each capture.
3. Each frame has been already oriented into an upright position.

**NOTE:** CA-1M will only include captures which were successfully registered to the laser scanner. Therefore
not every capture including in ARKitScenes will be present in CA-1M.

## Downloading the CA-1M data

### License

All data is released under the [CC-by-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/).

All links to the data are contained in `data/train.txt` and `data/val.txt`. You can use `curl` to download all files
listed. If you don't need the whole dataset in advance, you can either explicitly pass these
links or pass the split's `txt` file itself and use the `--video-ids` argument to filter the desired videos.

If you pass the `txt` file, please note that file will be cached under `data/[split]`.

### CA-VQA

The data links for CA-VQA are contained in `data/cavqa/val.txt` for the `val` split and in `data/cavqa/train_[task].txt` for the `train` splits of each task (`binary`, `cardinality`, `grounding2d`, `grounding3d`, `multichoice`, `regression`). 


## Understanding the CA-1M data

CA-1M is released in WebDataset format. Therefore, it is essentially a fancy tar archive
*per* capture (i.e., a video). Therefore, a single archive `ca1m-[split]-XXXXXXX.tar` corresponds to all data
of capture XXXXXXXX.

Both splits are released at full frame rate.

All data should be neatly loaded by `CubifyAnythingDataset`. Please refer to `dataset.py` for more
specifics on how to read/parse data on disk. Some general pointers:

```python
[video_id]/[integer_timestamp].wide/image.png               # A 1024x768 RGB image corresponding to the main camera.
[video_id]/[integer_timestamp].wide/depth.png               # A 256x192 depth image stored as a UInt16 (as millimeters) derived from the capture device's onboard LiDAR (ARKit depth).
[video_id]/[integer_timestamp].wide/depth/confidence.tiff   # A 256x192 confidence image storing the [0, 1] confidence value of each depth measurement (currently unused).
[video_id]/[integer_timestamp].wide/instances.json          # A list of GT instances alongside their 3D boxes (i.e., the resulting of the GT rendering process).
[video_id]/[integer_timestamp].wide/T_gravity.json          # A rotation matrix which encodes the pitch/roll of the camera, which we assume is known (e.g., IMU).

[video_id]/[integer_timestamp].gt/RT.json                   # A 4x4 (row major) JSON-encoded matrix corresponding to the registered pose in the laser-scanner space.
[video_id]/[integer_timestamp].gt/depth.png                 # A 512x384 depth image stored as a UInt16 (as millimeters) derived from the FARO laser scanner registration.

```

Note that since we have already oriented the images, these dimensions may be transposed. GT depth may have 0 values which corresponding to unregistered points.

An additional file is included as `[video_id]/world.gt/instances.json` which corresponds to the full world set of 3D annotations from which
the per-frame labels are generated from. These instances include some structural labels: `wall`, `floor`, `ceiling`, `door_frame` which
might aid in rendering.

### CA-VQA

We provide CA-VQA across the different tasks (`binary`, `cardinality`, `grounding2d`, `grounding3d`, `multichoice`, `regression`) in the format described below.

#### Val split
The `val` split of CA-VQA can be loaded (after unarchiving) via HuggingFace datasets:
```python
import datasets
cavqa_val = datasets.load_from_disk("[path_to_cavqa_val_dir]")
```

This will provide a dictionary where each entry is a task's dataset with the following features:
```python
id: str                             # Unique ID of the data example.
question: str                       # The question.
answer: str                         # The ground truth answer of the question.
reference_frame: dict               # The frame that the question refers to.
support_frame_[1-4]: dict           # The preceding frames; order: [4, 3, 2, 1, reference].

# Features of each frame.
image['path']: str                  # Path to the RGB image (png) within the `images` folder.
depth_map_ground_truth['path']: str # Path to the ground-truth depth map.
depth_map_arkit['path']: str        # Path to the ARKit depth map.
depth_map_monocular['path']: str    # Path to the monocular depth map.
pose: List[List[float]]             # The pose (relative to the reference).
intrinsics: List[List[float]]       # The intrinsics.
```

#### Train split
The `train` split of each CA-VQA task can be loaded via TFDS:
```python
import tensorflow_datasets as tfds
builder = tfds.builder_from_directory("[path_to_cavqa_train_dir]/[task]/1.0.0")
cavqa_task_train = builder.as_dataset(split="train")
```

This will provide a dataset with the following features:
```python
questions: Tensor[str]                # The questions (referring to the reference frame).
answers: Tensor[str]                  # The ground truth answers of the questions.
images: Tensor[str]                   # The RGB images, with the reference frame coming last.
depth_maps_ground_truth: Tensor[str]  # The ground-truth depth maps (same frame order as `images`).
depth_maps_arkit: Tensor[str]         # The ARKit depth maps (same frame order as `images`).
depth_maps_monocular: Tensor[str]     # The monocular depth maps (same frame order as `images`).
poses: Tensor[float]                  # The poses (same frame order as `images`).
intrinsics: Tensor[float]             # The intrinsics (same frame order as `images`).
```

## Visualization of the CA-1M data

We include visualization support for CA-1M using [rerun](https://rerun.io). Visualization should happen
automatically. If you wish to not run any models, but only visualize the data, use `--viz-only`.

During inference, you may wish to inspect the 3D accuracy of the predictions. We support
visualizing the predictions on the GT point cloud (derived from Faro depth) when using
the `--viz-on-gt-points` flag.

### Sample command

``` bash
python tools/demo.py [path_to_downloaded_data]/ca1m-val-42898570.tar --viz-only
```

``` bash
python tools/demo.py data/train.txt --viz-only --video-ids 45261548
```

## Skipping Frames

The data is provided at a high frame rate, so using `--every-nth-frame N` will only
process every N frames.

## Running the CuTR models

All models are released under the Apple ML Research Model Terms of Use in [LICENSE_MODEL](LICENSE_MODEL).

1. [RGB-D](https://ml-site.cdn-apple.com/models/cutr/cutr_rgbd.pth)
2. [RGB](https://ml-site.cdn-apple.com/models/cutr/cutr_rgb.pth)

Models can be provided to `demo.py` using the `--model-path` argument. We detect whether this is an RGB
or RGB-D model and disable depth accordingly.

### RGB-D

The first variant of CuTR expects an RGB image and a metric depth map. We train on ARKit depth,
although you may find it works with other metric depth estimators as well.

#### Sample Command

If your computer is MPS enabled:

``` bash
python tools/demo.py data/val.txt --video-ids 42898570 --model-path [path_to_models]/cutr_rgbd.pth --viz-on-gt-points --device mps
```

If your computer is CUDA enabled:

``` bash
python tools/demo.py data/val.txt --video-ids 42898570 --model-path [path_to_models]/cutr_rgbd.pth --viz-on-gt-points --device cuda
```

Otherwise:

``` bash
python tools/demo.py data/val.txt --video-ids 42898570 --model-path [path_to_models]/cutr_rgbd.pth --viz-on-gt-points --device cpu
```

### RGB Only

The second variant of CuTR expects an RGB image alone and attempts to derive the metric scale of
the scene from the image itself.

#### Sample Command

If your device is MPS enabled:

``` bash
python tools/demo.py data/val.txt --video-ids 42898570 --model-path [path_to_models]/cutr_rgb.pth --viz-on-gt-points --device mps
```

## Run on captures from your own device

We also have basic support for running on RGB/Depth captured from your own device.

1. Make sure you have [NeRF Capture](https://apps.apple.com/au/app/nerfcapture/id6446518379) installed on your device
2. Start the NeRF Capture app *before* running `demo.py` (force quit and reopen if for some reason things stop working or a connection is not made).
3. Run the normal commands but pass "stream" instead of the usual tar/folder path.
4. Hit "Send" in the app to send a frame for inference. This will be visualized in the rerun window.

We will continue to print "Still waiting" to show liveliness.

If you have a device equipped with LiDAR, you can use this combined with the RGB-D models, otherwise, you can
only use the RGB only model.

#### RGB-D (on MPS)

``` bash
python tools/demo.py stream --model-path [path_to_models]/cutr_rgbd.pth --device mps
```

#### RGB (on MPS)

``` bash
python tools/demo.py stream --model-path [path_to_models]/cutr_rgb.pth --device mps
```

## Citations

If you use CA-1M or CuTR in your research, please use the following entry:

```
@article{lazarow2024cubify,
  title={Cubify Anything: Scaling Indoor 3D Object Detection},
  author={Lazarow, Justin and Griffiths, David and Kohavi, Gefen and Crespo, Francisco and Dehghan, Afshin},
  journal={arXiv preprint arXiv:2412.04458},
  year={2024}
}
```

If you use CA-VQA in your research, please use the following entry:

```
@article{daxberger2025mmspatial,
  title={MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs},
  author={Daxberger, Erik and Wenzel, Nina and Griffiths, David and Gang, Haiming and Lazarow, Justin and Kohavi, Gefen and Kang, Kai and Eichner, Marcin and Yang, Yinfei and Dehghan, Afshin and Grasch, Peter},
  journal={arXiv preprint arXiv:2503.13111},
  year={2025}
}
```

## License

The sample code is released under Apple Sample Code License.

The data is released under [CC-by-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Acknowledgements

We use and acknowledge contributions from multiple open-source projects in [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS).
