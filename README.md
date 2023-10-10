# Export YOLOv8-seg End2End

## Export to ONNX

- Without RoiAlign: `python3 export.py --weights yolov8l-seg.pt --imgsz 640 640 --batch-size 1 --device 1 --dynamic --simplify --opset 14 --dynamic-batch --cleanup --topk-all 3000 --iou-thres 0.65 --conf-thres 0.5 --end2end`

- With RoiAlign: `python3 export.py --weights yolov8l-seg.pt --imgsz 640 640 --batch-size 1 --device 1 --dynamic --simplify --opset 14 --dynamic-batch --cleanup --topk-all 3000 --iou-thres 0.65 --conf-thres 0.5 --end2end --mask-resolution 56 --roi-align`

## Export to TensorRT

- Without RoiAlign: `python3 export.py --weights yolov8l-seg.pt --imgsz 640 640 --batch-size 1 --device 1 --dynamic --simplify --opset 14 --dynamic-batch --cleanup --topk-all 3000 --iou-thres 0.65 --conf-thres 0.5 --end2end --trt`

- With RoiAlign: `python3 export.py --weights yolov8l-seg.pt --imgsz 640 640 --batch-size 1 --device 1 --dynamic --simplify --opset 14 --dynamic-batch --cleanup --topk-all 3000 --iou-thres 0.65 --conf-thres 0.5 --end2end --mask-resolution 56 --roi-align --trt`
