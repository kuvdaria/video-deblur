import gc
import os
from pathlib import Path

import cv2
import numpy as np
import skvideo.io
import torch
import wget
from flask import Flask, request, send_file
from skvideo.io import FFmpegWriter
from tqdm import tqdm


os.system('bash setup.sh')


import EDVR.codes.data.util as data_util
import EDVR.codes.models.archs.EDVR_arch as EDVR_arch
import EDVR.codes.utils.util as util





app = Flask(__name__)
# ---model-specs-----
data_mode = 'blur'
chunk_size = 100
stage = 1
device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
flip_test = False
url = 'https://drive.google.com/uc?export=download&id=1ZCl0aU8isEnUCsUYv9rIZZQrGo7vBFUH'
model_path = 'EDVR_REDS_deblur_L.pth'
if not os.path.exists(model_path):
    model_path = wget.download(url)
print('Model Used: ', model_path)
predeblur, HR_in = True, True
N_in = 5
back_RBs = 40
model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# ---other-specs---
crop_border = 0
# border frames when evaluate
border_frame = N_in // 2
# temporal padding mode
padding = 'replicate'
save_imgs = True
num_to_pr = 30
if not os.path.exists('videos'):
    os.mkdir('videos')


def clean_mem():
    gc.collect()


def preProcess(frame, multiple):
    img = frame
    h, w = img.shape[:2]
    # resize so they are multiples of 4 or 16 (for blurred)
    h = h - h % multiple
    w = w - w % multiple
    img = img.reshape((h, w, 3))

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_img_seq(img_l):
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        file = request.files['file']
        video_path = 'videos/new_video.mp4'
        file.save(video_path)
        fps = skvideo.io.ffprobe(video_path)['video']['@avg_frame_rate']
        writer = FFmpegWriter('videos/video_done.mp4',
                              outputdict={"-vcodec": "libx264",
                                          "-crf": '17',
                                          "-pix_fmt": "yuv420p",
                                          "-framerate": fps})
        cap = cv2.VideoCapture(video_path)

        num = 0
        frames = []
        end = False
        while True:
            if end:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                end = True
            else:
                num += 1
                frame = preProcess(frame, 16)
                frames.append(frame)

            if (num % num_to_pr == 0 or end) and frames != []:
                clean_mem()
                imgs_LQ = read_img_seq(frames)
                max_idx = len(frames)

                for img_idx, fr in tqdm(enumerate(frames)):
                    select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
                    imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).\
                        to(device)

                    output = util.single_forward(model, imgs_in)
                    output = util.tensor2img(output.squeeze(0))
                    writer.writeFrame(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                frames = []

        writer.close()
        cap.release()
        print('deblur done')

        video = 'videos/new_video.mp4'
        audio_file = Path(str(video).replace('.mp4', '.aac'))
        if audio_file.exists():
            audio_file.unlink()

        os.system(
            'ffmpeg -y -i "'
            + str(video)
            + '" -vn -acodec copy "'
            + str(audio_file)
            + '"'
        )

        out_path = 'videos/video_done.mp4'
        result_path = 'videos/video_done_aud.mp4'
        if audio_file.exists:
            os.system(
                'ffmpeg -y -i "'
                + str(out_path)
                + '" -i "'
                + str(audio_file)
                + '" -shortest -c:v copy -c:a aac -b:a 256k -strict -2 "'
                + str(result_path)
                + '"'
            )
        return send_file('videos/video_done_aud.mp4', attachment_filename='video_done_aud.mp4')


if __name__ == '__main__':
    app.run()
