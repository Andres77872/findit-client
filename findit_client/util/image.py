import asyncio
import io
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import cv2
import numpy as np
from PIL import Image

from findit_client.exceptions import (ImageNotLoadedException,
                                      ImageNotFetchedException,
                                      ImageSizeTooBigException,
                                      ImageRemoteNotAsImageContentTypeException,
                                      ImageRemoteNoContentLengthFoundException,
                                      TooFewSearchResultsException)
from findit_client.models import ImageSearchResponseModel
from findit_client.util.pixiv import check_login, login_in, get_image
from findit_client.util.validations import validate_load_image

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor()
# Global session
sess: aiohttp.ClientSession = None


async def ensure_session():
    global sess
    await init_sess()


async def init_sess():
    """Initialize the global aiohttp session."""
    global sess
    sess = aiohttp.ClientSession(headers={'User-Agent': 'findit.moe client -> https://findit.moe'})


async def close_sess():
    """Close the global aiohttp session."""
    if sess:
        await sess.close()


def normalize(img, normalization_mode=True):
    if normalization_mode:
        img = img / 255
    else:
        img = (img / 127.5) - 1
    return img


def _load_image_sync(img: bytes,
                     width: int = 448,
                     height: int = 448,
                     normalization_mode: bool | None = None,
                     color_schema_rgb: bool = False,
                     padding_color: int = 255,
                     mode: str = None,
                     origin: str = None
                     ):
    """Synchronous version of image loading logic (CPU-bound)"""
    try:
        image_bytes = io.BytesIO(img)
        with Image.open(image_bytes) as image:
            # Ensure the image has an alpha channel for transparency
            image = image.convert('RGBA')

            # Get the aspect ratio of the image
            aspect = image.width / image.height
            if aspect > 1:
                new_width = width
                new_height = int(height / aspect)
            else:
                new_width = int(width * aspect)
                new_height = height

            # Resize the image with the aspect ratio
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            np_image = np.array(resized_image)
            if np_image.shape[2] == 4:
                alpha = np_image[:, :, 3:4] / 255.0
                background = np.ones_like(np_image, dtype=np.uint8) * 255
                background[:, :, :3] = [255, 255, 255]
                blended = (np_image * alpha + background * (1 - alpha)).astype(np.uint8)
                bgr_image = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            z = (height, width)
            s = np.array(bgr_image.shape[:2])
            p = max(s / z)
            r = (s / p).astype(int)

            re = np.zeros((*z, 3)) + padding_color

            offset = (z - r) // 2
            re[offset[0]:offset[0] + r[0], offset[1]:offset[1] + r[1]] = bgr_image

            if normalization_mode is not None:
                re = normalize(img=re,
                               normalization_mode=normalization_mode)
            if color_schema_rgb:
                re = re[..., ::-1]
            return re[None]
    except Exception as e:
        print(f"Skipping... e={e}")
        raise ImageNotLoadedException(mode=mode, origin=origin)


async def load(img: bytes,
               width: int = 448,
               height: int = 448,
               normalization_mode: bool | None = None,
               color_schema_rgb: bool = False,
               padding_color: int = 255,
               mode: str = None,
               origin: str = None
               ):
    """Async wrapper for image loading (runs in thread pool)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool,
                                      _load_image_sync,
                                      img, width, height, normalization_mode,
                                      color_schema_rgb, padding_color, mode, origin)


async def build_masonry_collage(results: ImageSearchResponseModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    async def fetch_and_process(url):
        await ensure_session()
        async with sess.get(url + '.png', timeout=5) as rq:
            content = await rq.read()
            # Process in thread pool since cv2 operations are CPU-bound
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(thread_pool, process_image, content)

    def process_image(content):
        arr = np.asarray(bytearray(content), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        return resize_local(img)

    # Create tasks for all image fetches
    tasks = []
    for i in results.results.data:
        tasks.append(fetch_and_process(i[0].img + '?access'))

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)

    # Process results
    for input_image, h in results:
        col.sort(key=lambda x: x[0])
        col[0][0] += h
        col[0][1].append(input_image / 255)

    # Final processing - this is CPU-bound, so run in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, finalize_collage, col)


def finalize_collage(col):
    hm = min(col, key=lambda x: x[0])[0]
    hm = min([hm, 1024])
    for x, i in enumerate(col):
        col[x] = np.concatenate(i[1], axis=0)[:hm]
    col = np.concatenate(col, axis=1)

    resize_local = lambda img, sz=256: (cv2.resize(img, (sz, int(img.shape[0] / (img.shape[1] / sz)))),
                                        int(img.shape[0] / (img.shape[1] / sz)))

    p5 = resize_local(col, 512)[0]
    p2 = resize_local(col, 256)[0]

    return col, p5, p2


@validate_load_image
async def load_file_image(image: str | list[str],
                          **kwargs) -> tuple[np.ndarray, float]:
    """
    :param image: Image as numpy array or image URL
    :param width: Image width destination (optional)
    :param height: Image height destination (optional)
    :param normalization_mode: Normalization mode True=[0-1], False=[-1,1], None=[0-255]
    :param color_schema_bgr: Color schema
    :param padding_color: Color padding in the resize
    :return: A tuple of (raw image (if selected), raw shape, resized image)
    """
    st = time.time()
    # File I/O can be slow, so run in thread pool
    loop = asyncio.get_event_loop()
    file_content = await loop.run_in_executor(thread_pool, lambda: open(image, "rb").read())

    i = await load(img=file_content,
                   mode='file',
                   origin=image,
                   **kwargs)
    return i, time.time() - st


@validate_load_image
async def load_url_image(image: str | list[str],
                         pixiv_credentials: dict = None,
                         get_raw_content: bool = False,
                         **kwargs) -> tuple[np.ndarray, float] | str | bytes:
    """
    :param image: Image as numpy array or image URL
    :param width: Image width destination (optional)
    :param height: Image height destination (optional)
    :param normalization_mode: Normalization mode True=[0-1], False=[-1,1], None=[0-255]
    :param color_schema_bgr: Color schema
    :param padding_color: Color padding in the resize
    :return: A tuple of (raw image (if selected), raw shape, resized image)
    """
    st = time.time()

    if image.startswith('https://i.pximg.net'):
        if not check_login():
            await login_in(**pixiv_credentials)
        content = get_image(url=image)
    else:
        await ensure_session()
        async with sess.get(image, timeout=2) as rq:
            if rq.status != 200:
                raise ImageNotFetchedException(origin=image)

            content_type = rq.headers.get('Content-Type', '')
            if 'image' not in content_type:
                raise ImageRemoteNotAsImageContentTypeException(origin=image)

            content_length = rq.headers.get('Content-Length')
            if not content_length:
                raise ImageRemoteNoContentLengthFoundException(origin=image)

            if int(content_length) > 8000000:
                raise ImageSizeTooBigException(origin=image,
                                               limit=8000000,
                                               size=int(content_length))

            content = await rq.read()

    if get_raw_content:
        return content

    i = await load(img=content,
                   mode='file',
                   origin=image,
                   **kwargs)
    return i, time.time() - st


@validate_load_image
async def load_bytes_image(image: bytes | list[bytes],
                           **kwargs) -> tuple[np.ndarray, float]:
    """
    :param image: Image as numpy array or image URL
    :param width: Image width destination (optional)
    :param height: Image height destination (optional)
    :param normalization_mode: Normalization mode True=[0-1], False=[-1,1], None=[0-255]
    :param color_schema_bgr: Color schema
    :param padding_color: Color padding in the resize
    :return: A tuple of (raw image (if selected), raw shape, resized image)
    """
    st = time.time()
    i = await load(img=image,
                   mode='file',
                   origin='upload',
                   **kwargs)
    return i, time.time() - st


# These functions are CPU-bound and relatively simple, so keeping them synchronous
def compress_nparr(nparr: np.ndarray):
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed


def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring)))
