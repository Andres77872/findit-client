import time
import io
import zlib
import numpy as np
import cv2
import threading
import requests

from findit_client.exceptions import (ImageNotLoadedException,
                                      ImageNotFetchedException,
                                      ImageSizeTooBigException,
                                      ImageRemoteNotAsImageContentTypeException,
                                      ImageRemoteNoContentLengthFoundException, TooFewSearchResultsException)
from findit_client.models import ImageSearchResponseModel


def resize(
        img: np.ndarray,
        width=256,
        height=256,
        padding=255
) -> np.ndarray:
    p = max(img.shape[:2] / np.array([height, width]))
    r = img.shape[:2] / p
    img = cv2.resize(img, (int(r[1]), int(r[0])), cv2.INTER_NEAREST)
    re = np.zeros((height, width, 3)) + padding
    offset = np.array((np.array(re.shape[:2]) - np.array(img.shape[:2])) / 2, dtype=np.int32)
    re[offset[0]:offset[0] + img.shape[0], offset[1]:offset[1] + img.shape[1]] = img

    return re


def normalize(img, normalization_mode=True):
    if normalization_mode:
        img = img / 255
    else:
        img = (img / 127.5) - 1
    return img


def load(
        img: np.ndarray,
        width: int = 448,
        height: int = 448,
        normalization_mode: bool | None = None,
        color_schema_rgb: bool = False,
        padding_color: int = 255,
        mode: str = None,
        origin: str = None
) -> np.ndarray:
    if img is None:
        raise ImageNotLoadedException(mode=mode, origin=origin)

    if color_schema_rgb:
        img = img[..., ::-1]

    if normalization_mode is not None:
        img = normalize(img=img,
                        normalization_mode=normalization_mode)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = resize(img, width=width, height=height, padding=padding_color)
    return img[None]


def build_masonry_collage(results: ImageSearchResponseModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if results.results.count < 4:
        raise TooFewSearchResultsException()

    col = [
        [0, []], [0, []], [0, []], [0, []]
    ]

    def resize_local(img, sz=256):
        h, w, _ = img.shape

        h = int(h / (w / sz))
        w = int(w / (w / sz))

        return cv2.resize(img, (w, h)), h

    def run(url):
        rq = sess.get(url, timeout=5, stream=True)
        arr = np.asarray(bytearray(rq.content), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        input_image, h = resize_local(img)

        col.sort(key=lambda x: x[0])
        col[0][0] += h
        col[0][1].append(input_image / 255)

    th = []
    for idx, i in enumerate(results.results.data):
        th.append(threading.Thread(target=run, args=[i[0].img + '?access']))
        th[-1].start()
    for i in th:
        i.join()

    hm = min(col, key=lambda x: x[0])[0]
    hm = min([hm, 1024])
    for x, i in enumerate(col):
        col[x] = np.concatenate(i[1], axis=0)[:hm]
    col = np.concatenate(col, axis=1)

    p5 = resize_local(col, 512)[0]
    p2 = resize_local(col, 256)[0]

    return col, p5, p2


def load_file_image(image_file: str,
                    **kwargs) -> tuple[np.ndarray, float]:
    """
    :param image_file: Image as numpy array or image URL
    :param width: Image width destination (optional)
    :param height: Image height destination (optional)
    :param normalization_mode: Normalization mode True=[0-1], False=[-1,1], None=[0-255]
    :param color_schema_bgr: Color schema
    :param padding_color: Color padding in the resize
    :return: A tuple of (raw image (if selected), raw shape, resized image)
    """
    st = time.time()
    i = load(img=cv2.imread(image_file),
             mode='file',
             origin=image_file,
             **kwargs)
    return i, time.time() - st


sess = requests.Session()
sess.headers['User-Agent'] = 'findit.moe client -> https://findit.moe'


def load_url_image(url: str,
                   **kwargs) -> tuple[np.ndarray, float]:
    """
    :param url: Image as numpy array or image URL
    :param width: Image width destination (optional)
    :param height: Image height destination (optional)
    :param normalization_mode: Normalization mode True=[0-1], False=[-1,1], None=[0-255]
    :param color_schema_bgr: Color schema
    :param padding_color: Color padding in the resize
    :return: A tuple of (raw image (if selected), raw shape, resized image)
    """
    st = time.time()
    rq = sess.get(url, timeout=2, stream=True)
    if rq.status_code != 200:
        raise ImageNotFetchedException(origin=url)

    if 'image' not in rq.headers['Content-Type']:
        raise ImageRemoteNotAsImageContentTypeException(origin=url)
    if 'Content-Length' not in rq.headers:
        raise ImageRemoteNoContentLengthFoundException(origin=url)
    if int(rq.headers['Content-Length']) > 8000000:
        raise ImageSizeTooBigException(origin=url,
                                       limit=8000000,
                                       size=int(rq.headers['Content-Length']))

    arr = np.asarray(bytearray(rq.content), dtype=np.uint8)
    i = load(img=cv2.imdecode(arr, -1),
             mode='file',
             origin=url,
             **kwargs)
    return i, time.time() - st


def load_bytes_image(image_file: bytes,
                     **kwargs) -> tuple[np.ndarray, float]:
    """
    :param image_file: Image as numpy array or image URL
    :param width: Image width destination (optional)
    :param height: Image height destination (optional)
    :param normalization_mode: Normalization mode True=[0-1], False=[-1,1], None=[0-255]
    :param color_schema_bgr: Color schema
    :param padding_color: Color padding in the resize
    :return: A tuple of (raw image (if selected), raw shape, resized image)
    """
    st = time.time()
    arr = np.asarray(bytearray(image_file), dtype=np.uint8)
    i = load(img=cv2.imdecode(arr, -1),
             mode='file',
             origin=image_file,
             **kwargs)
    return i, time.time() - st


def compress_nparr(nparr: np.ndarray):
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed  # , len(uncompressed), len(compressed)


def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring)))
