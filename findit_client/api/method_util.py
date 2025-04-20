import hashlib

import numpy as np

from findit_client.api.api_requests import ApiRequests
from findit_client.exceptions import ImageNotFetchedException
from findit_client.models import ImageSearchResponseModel
from findit_client.util import load_file_image, load_url_image, load_bytes_image
from findit_client.util.image import build_masonry_collage
from findit_client.util.pixiv import get_pixiv_image_url, get_crawler_image
from findit_client.util.zip_file import zip_file


class FindItMethodsUtil:
    def __init__(self,
                 __version__: str,
                 pixiv_credentials: dict,
                 **kwargs):
        self.ApiRequests = ApiRequests(**kwargs)
        self.__version__ = __version__
        self.pixiv_credentials = pixiv_credentials
        # self.client = openai.OpenAI(api_key=__ChatGPT_TOKEN__)

    def random_search_generator(
            self,
            pool: list[str] = None,
            limit: int = 32,
            content: str = 'g'
    ) -> ImageSearchResponseModel:
        return self.ApiRequests.generate_random_response(
            pool=pool,
            limit=limit,
            content=content,
            mode='RANDOM',
            load_image_time=0,
            api_version=self.__version__
        )

    def image_encoder_by_file(
            self,
            img: str,
    ) -> list[float]:
        img_array, _ = load_file_image(img)
        return self.ApiRequests.get_embedding_vector(img_array)

    def image_encoder_by_url(
            self,
            url: str,
    ) -> list[float]:
        img_array, _ = load_url_image(image=url,
                                      pixiv_credentials=self.pixiv_credentials)
        return self.ApiRequests.get_embedding_vector(img_array)

    def image_encoder_by_image_bytes(
            self,
            image_file: bytes,
    ) -> list[float]:
        img_array, _ = load_bytes_image(image_file)
        return self.ApiRequests.get_embedding_vector(img_array)

    def generate_masonry_collage(
            self,
            results: ImageSearchResponseModel
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return build_masonry_collage(results)

    def generate_md5_by_url(self,
                            url: str) -> str:
        content = load_url_image(image=url,
                                 get_raw_content=True,
                                 pixiv_credentials=self.pixiv_credentials)
        return hashlib.md5(bytearray(content)).hexdigest()

    def generate_md5_by_file(self,
                             image_file: bytes) -> str:

        return hashlib.md5(bytearray(image_file)).hexdigest()

    def download_pixiv_image(self,
                             idx: int,
                             token: str = None
                             ):
        urls = get_pixiv_image_url(idx)
        if token is None:
            def retry(n, u):
                for i in ['.png', '.jpg', '.jpeg']:
                    try:
                        r = load_url_image(image=u.replace('.png', i),
                                           get_raw_content=True,
                                           pixiv_credentials=self.pixiv_credentials)
                    except ImageNotFetchedException:
                        continue
                    return n.replace('.png', i), r

            data = []
            for name, url in urls:
                data.append(retry(name, url))
        else:
            data = get_crawler_image(url=urls, token=token)
        return zip_file(data)
