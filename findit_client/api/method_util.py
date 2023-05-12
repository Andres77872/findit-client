import numpy as np

from findit_client.models import ImageSearchResponseModel
from findit_client.api.api_requests import ApiRequests
from findit_client.util import load_file_image, load_url_image, load_bytes_image

from findit_client.util.image import build_masonry_collage


class FindItMethodsUtil:
    def __init__(self,
                 __version__: str,
                 **kwargs):
        self.ApiRequests = ApiRequests(**kwargs)
        self.__version__ = __version__

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
        img_array, _ = load_url_image(url)
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
