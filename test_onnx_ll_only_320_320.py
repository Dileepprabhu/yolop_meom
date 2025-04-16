import os
import cv2
# import torch
import argparse
import onnxruntime as ort
import numpy as np
# import matplotlib.pyplot as plt
# import onnx
# from onnx_opcounter import calculate_params
import time

def resize_unscale(img, new_shape=(320, 320), color=114):

    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def infer_yolop(ort_session,img_path, model_file, input_dim):

    inference_sv_path = os.path.splitext(img_path.replace('images', f'da_headless-det_img_dim_cmpr'))[0]
    os.makedirs(os.path.dirname(inference_sv_path), exist_ok = True)

    save_da_path = f"./{inference_sv_path}_{model_file}_da.jpg"
    save_merge_path = f"./{inference_sv_path}_{model_file}_output.jpg"

    img_bgr = cv2.imread(img_path)
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (input_dim, input_dim))

    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)  # (1, 3,640,640)

    print(img.shape)

    da_seg_out = ort_session.run(
        ['lane_line_seg'],
        input_feed={"images": img}
    )[0]

    # select da & ll segment area.
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    # print(da_seg_mask.shape)

    # da: resize to original size
    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    # cv2.imwrite(save_merge_path, img_merge)
    cv2.imwrite(save_da_path, da_seg_mask)

    print("detect done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="yolop_da-640-640.onnx")
    parser.add_argument('--img_f', type=str, default="./inference/images")
    args = parser.parse_args()

    ort.set_default_logger_severity(4)
    #onnx_path = f"./weights/{args.weight}"

    onnx_paths = [
        './weights/yolop-320-320.onnx',
        #'./weights/yolop-640-640.onnx',
        #'./weights/yolop-da-ll-320-320.onnx',
        #'./weights/yolop_da_ll-640-640.onnx',
        #'./weights/yolop_det_da-640-640.onnx',

    ]

    for onnx_path in onnx_paths[:]:

        # model = onnx.load_model(onnx_path)
        # params = calculate_params(model)
        # print(f'The model has {params} parameters')
        # del model

        img_dim = int(os.path.splitext(os.path.basename(onnx_path))[0].split('-')[-1])

        ort_session = ort.InferenceSession(onnx_path)
        filename = os.path.splitext(os.path.basename(onnx_path))[0]
        print(f"Load {onnx_path} done!")

        file_list = os.listdir(args.img_f)

        for i, img_name in enumerate(file_list[:]):
            img_path = os.path.join(args.img_f, img_name)
            if os.path.isdir(img_path):
                continue
            start_time = time.time()
            infer_yolop(ort_session, img_path=img_path, model_file = filename, input_dim = img_dim)
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time for image {img_name}: {inference_time:.4f} seconds")

    """
    PYTHONPATH=. python3 ./test_onnx.py --weight yolop-640-640.onnx --img test.jpg
    """