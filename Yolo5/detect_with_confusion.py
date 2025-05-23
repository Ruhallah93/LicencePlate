# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np


import pandas as pd

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def get_ground_truth(path):
  """
  path of label as input
  output is (8, 4) representing 8 boxes each in format (x1, y1, x2, y2)
  """
  # path "/content/test/label/00_EIN_32143.txt"
  boxes = []
  classes = []

  with open(path, "r") as f:
    
    for line in f:
      
      cls, x, y, w, h = line.split(" ")
      
      x = float(x) * 332
      w = float(w) * 332
      y = float(y) * 170
      h = float(h) * 170

      x1 = int(x - 0.5 * w)
      y1 = int(y - 0.5 * h)
      x2 = int(x + 0.5 * w)
      y2 = int(y + 0.5 * h)

      cls = int(cls)

      boxes.append((x1, y1, x2, y2))
      classes.append(cls)
  
  return boxes, classes


def get_iou(box1, box2):
  # assumeing boxes are in format (x1, y1, x2, y2)
  inter = {}
  inter["x_left_top"] = max(box1[0], box2[0])
  inter["y_left_top"] = max(box1[1], box2[1])

  inter["x_right_bottom"] = min(box1[2], box2[2])
  inter["y_right_bottom"] = min(box1[3], box2[3])

  if inter["x_left_top"] < inter["x_right_bottom"] and inter["y_left_top"] < inter["y_right_bottom"]: # there is a non-zero intersection  
    iou_area = (inter["x_right_bottom"] - inter["x_left_top"]) * (inter["y_right_bottom"] - inter["y_left_top"])
  else:
    iou_area = 0

  box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
  box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

  iou = iou_area / (box1_area + box2_area - iou_area)

  return iou

def merger(boxes):

  """
  It recieves predicted boxes as input and find boxes with overlaps
  Then choose the box with higher confidence
  It is not guranteed but the output would probably be 8 boxes
  """

  """
    boxes are in format list of tuples
    (x, y, width, height, confidence, label)
    it doesn't matter whether numbers are normalized or not
    because it does not effect IoU
  """


  i = 0 # the predicted box we are checking
  final_det = []

  

  while i < len(boxes) - 1 :
    
    c_x, c_y, c_w, c_h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
    n_x, n_y, n_w, n_h = boxes[i + 1][0], boxes[i + 1][1], boxes[i + 1][2], boxes[i + 1][3]

    current_box = (c_x, c_y, c_w, c_h)
    next_box = (n_x, n_y, n_w, n_h)
    
    if get_iou(current_box, next_box) > 0.6 :

      index_to_remove =  i + 1 if boxes[i][-2] > boxes[i + 1][-2] else i  
      del boxes[index_to_remove]

    else:
      final_det.append(boxes[i])
      i += 1

  final_det.append(boxes[i])
  return final_det


def update_confusion(confusion_matrix, predicted_box, gt_boxes, predicted_cls, gt_cls, missed_boxes):
  """


  """
  ious = []
  for gt_box in gt_boxes:
    ious.append(get_iou(gt_box, predicted_box))
  

  max_iou = np.argmax(ious)

  missed_boxes[max_iou] = -1

  confusion_matrix[gt_cls[max_iou], predicted_cls] += 1

  return confusion_matrix
 
@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'no_labels/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    counter = 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        # confusion_matrix = np.zeros((len(names) + 1, len(names) + 1)) # +1 for unknown

        # plate_chars = path.split("/")[-1].split(".")[0].split("_") # list
        # plate_chars = list(plate_chars[0]) + [plate_chars[1]] + list(plate_chars[2]) # ground-truth list



        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            plate_chars = path.split("/")[-1].split(".")[0].split("_") # list
            plate_chars = list(plate_chars[0]) + [plate_chars[1]] + list(plate_chars[2])
            plate_chars = ''.join(plate_chars)

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                det_sorted = sorted(det, key=lambda box: box[0]) # sort by X coordinate
                
                for i, item in enumerate(det_sorted):
                  det_sorted[i] = det_sorted[i].cpu().numpy()
                
                final_det = merger(det_sorted)
                final_det = sorted(final_det, key=lambda x: x[-2], reverse=True)
                final_det = final_det[:8]

                final_det = sorted(final_det, key=lambda x: x[0])

                predicted_string = ""
                for c in final_det:
                  predicted_string += str(names[int(c[-1])])
                
                # print("plate_chars", plate_chars)
                # print("predicted_string", predicted_string)
                
                if plate_chars == predicted_string:
                  counter += 1
                else:
                  print("detected wrong", plate_chars)


                
                # gt_boxes, gt_cls = get_ground_truth("/content/test/label/" + path.split("/")[-1].split(".")[0] + ".txt")
                
                # missed_boxes = list(range(8))

                # for prediction in det_sorted:
                #   x1, y1, x2, y2, cls = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item(), prediction[-1].item()
                  
                #   predicted_box = (x1, y1, x2, y2)
                #   confusion_matrix = update_confusion(confusion_matrix, predicted_box, gt_boxes, int(cls), gt_cls, missed_boxes)

                # for missed_box in missed_boxes:
                #   if missed_box != -1:
                #     confusion_matrix[gt_cls[missed_box], len(names)] += 1

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in final_det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    # df = pd.DataFrame(confusion_matrix, columns=list(names) + ["unknown"], index=list(names) + ["unknown"])
    # df.to_csv("/content/confusion_matrix.csv")
    
    # print(confusion_matrix)
    print("Number of correct predictions: ", counter)
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'no_labels/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
