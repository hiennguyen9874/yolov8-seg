import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.checks import (
    check_requirements,
    check_yaml,
    check_imgsz,
    print_args,
)
from ultralytics.utils.ops import Profile
from ultralytics.utils import LOGGER, url2file, colorstr, get_default_args
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import (
    get_latest_opset,
    select_device,
    smart_inference_mode,
)

from end2end import End2End, End2EndRoialign
from warp_model import WarpModel


def export_formats():
    # YOLOv5 export formats
    x = [
        ["ONNX", "onnx", ".onnx", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(
                f"{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)"
            )
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            return None, None

    return outer_func


@try_export
def export_onnx(
    model,
    im,
    file,
    opset,
    dynamic,
    simplify,
    dynamic_batch,
    end2end,
    trt,
    topk_all,
    device,
    iou_thres,
    score_thres,
    mask_resolution,
    pooler_scale,
    sampling_ratio,
    image_size,
    cleanup,
    roi_align,
    roi_align_type,
    prefix=colorstr("ONNX:"),
):
    # YOLOv5 ONNX export
    check_requirements(("onnx",))
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = file.with_suffix(".onnx")

    output_names = ["output", "proto"]
    dynamic_axes = None

    if dynamic:
        raise NotImplementedError

    if end2end:
        if trt:
            output_names = [
                "num_dets",
                "det_boxes",
                "det_scores",
                "det_classes",
                "det_masks",
            ]
        else:
            output_names = ["output"]

    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
        }
        output_axes = {
            "output": {0: "batch"},
            "proto": {0: "batch"},
        }

        if end2end:
            if trt:
                output_axes = {
                    "num_dets": {0: "batch"},
                    "det_boxes": {0: "batch"},
                    "det_scores": {0: "batch"},
                    "det_classes": {0: "batch"},
                    "det_masks": {0: "batch"},
                }
            else:
                output_axes = {
                    "output": {0: "num_dets"},
                }
        dynamic_axes.update(output_axes)

    if end2end:
        if roi_align:
            model = End2EndRoialign(
                model=model,
                max_obj=topk_all,
                iou_thres=iou_thres,
                score_thres=score_thres,
                nc=len(model.names),
                mask_resolution=mask_resolution,
                pooler_scale=pooler_scale,
                sampling_ratio=sampling_ratio,
                device=device,
                trt=trt,
                max_wh=max(image_size),
                roi_align_type=roi_align_type,
            )
        else:
            model = End2End(
                model=model,
                max_obj=topk_all,
                iou_thres=iou_thres,
                score_thres=score_thres,
                nc=len(model.names),
                pooler_scale=pooler_scale,
                device=device,
                trt=trt,
                max_wh=max(image_size),
            )

    model.to(device)

    torch.onnx.export(
        # model.cpu()
        # if (dynamic or dynamic_batch)
        # else model,  # --(dynamic or dynamic_batch) only compatible with cpu
        # im.cpu() if (dynamic or dynamic_batch) else im,
        model,
        im,
        f,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        # do_constant_folding=False,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {
        "stride": int(max(model.model.stride if end2end else model.stride)),
        "names": model.model.names if end2end else model.names,
    }
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(
                ("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.4.1")
            )
            import onnxsim

            LOGGER.info(
                f"{prefix} simplifying with onnx-simplifier {onnxsim.__version__}..."
            )
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"{prefix} simplifier failure: {e}")

    if cleanup:
        try:
            import onnx_graphsurgeon as gs

            LOGGER.info(
                f"\nStarting to cleanup ONNX using onnx_graphsurgeon {gs.__version__}..."
            )

            graph = gs.import_onnx(model_onnx)
            graph = graph.cleanup().toposort()
            model_onnx = gs.export_onnx(graph)

            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"Cleanup failure: {e}")

    return f, model_onnx


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    nc=80,
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLOv5 Detect() inplace=True
    keras=False,  # use Keras
    int8=False,  # CoreML/TF INT8 quantization
    dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    simplify=False,  # ONNX: simplify model
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
    topk_all=100,  # TF.js NMS: topk for all classes to keep
    iou_thres=0.45,  # TF.js NMS: IoU threshold
    conf_thres=0.25,  # TF.js NMS: confidence threshold
    mask_resolution=56,
    pooler_scale=0.25,
    sampling_ratio=0,
    dynamic_batch=False,
    end2end=False,
    trt=False,
    cleanup=False,
    roi_align=False,
    roi_align_type=0,
):
    t = time.time()
    file = Path(
        url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights
    )  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert (
            device.type != "cpu"
        ), "--half only compatible with GPU export, i.e. use --device 0"
        assert (
            not dynamic
        ), "--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both"

    dynamic = False if dynamic_batch else dynamic
    dynamic = False if end2end else dynamic

    model = WarpModel(weights=weights, nc=nc, dynamic=dynamic, export=True)
    model.to(device)

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_imgsz(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(
        device
    )  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()  # training mode = no Detect() layer grid construction

    for _ in range(2):
        y = model(im)  # dry runs

    if half:
        im, model = im.half(), model.half()  # to FP16

    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)"
    )

    # Exports
    f = [""] * 10  # exported filenames
    warnings.filterwarnings(
        action="ignore", category=torch.jit.TracerWarning
    )  # suppress TracerWarning

    f[2], _ = export_onnx(
        model=model,
        im=im,
        file=file,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        cleanup=cleanup,
        dynamic_batch=dynamic_batch,
        end2end=end2end,
        trt=trt,
        topk_all=topk_all,
        device=device,
        iou_thres=iou_thres,
        score_thres=conf_thres,
        mask_resolution=mask_resolution,
        pooler_scale=pooler_scale,
        sampling_ratio=sampling_ratio,
        image_size=imgsz,
        roi_align=roi_align,
        roi_align_type=roi_align_type,
    )

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        h = "--half" if half else ""  # --half FP16 inference arg
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f"\nDetect:          python detect.py --weights {f[-1]} {h}"
            f"\nValidate:        python val.py --weights {f[-1]} {h}"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')"
            f"\nVisualize:       https://netron.app"
        )
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image (h, w)",
    )
    parser.add_argument("--nc", type=int, default=80, help="Num classes")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--half", action="store_true", help="FP16 half-precision export"
    )
    parser.add_argument(
        "--inplace", action="store_true", help="set YOLOv5 Detect() inplace=True"
    )
    parser.add_argument(
        "--int8", action="store_true", help="CoreML/TF INT8 quantization"
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes"
    )
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument(
        "--workspace", type=int, default=4, help="TensorRT: workspace size (GB)"
    )
    parser.add_argument(
        "--topk-all",
        type=int,
        default=100,
        help="TF.js NMS: topk for all classes to keep",
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="TF.js NMS: IoU threshold"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="TF.js NMS: confidence threshold"
    )
    parser.add_argument(
        "--dynamic-batch", action="store_true", help="ONNX: dynamic batching"
    )
    parser.add_argument("--end2end", action="store_true", help="ONNX: NMS")
    parser.add_argument("--trt", action="store_true", help="ONNX: TRT")
    parser.add_argument("--cleanup", action="store_true", help="ONNX: Cleanup")
    parser.add_argument(
        "--mask-resolution", type=int, default=56, help="ONNX: Roialign mask-resolution"
    )
    parser.add_argument(
        "--pooler-scale",
        type=float,
        default=0.25,
        help="ONNX: Roialign scale, scale = proto shape / input shape",
    )
    parser.add_argument(
        "--sampling-ratio", type=int, default=0, help="ONNX: Roialign sampling ratio"
    )
    parser.add_argument(
        "--roi-align",
        action="store_true",
        help="ONNX: Crop And Resize mask using roialign",
    )
    parser.add_argument(
        "--roi-align-type",
        type=int,
        default=0,
        help="ONNX: Roialign type, 0: RoiAlign, 1: RoIAlignDynamic_TRT, 2: RoIAlign2Dynamic_TRT",
    )
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
