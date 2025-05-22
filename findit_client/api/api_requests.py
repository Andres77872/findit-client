import functools

import numpy as np

from findit_client.api import conextions
from findit_client.api.const import RANDOM_GENERATOR_API_PATH
from findit_client.models import ImageSearchResponseModel
from findit_client.models.builder import build_tagger_response
from findit_client.models.model_tagger import TaggerResponseModel


def async_cache_function():
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator


class ApiRequests:
    def __init__(
            self,
            url_api_embedding: str,
            url_api_back_search: str,
            cache_decorator=None,
            **kwargs
    ):
        self.cache_decorator = cache_decorator or async_cache_function
        self.url_api_embedding = url_api_embedding
        self.url_api_back_search = url_api_back_search
        self.url_image_backend = kwargs.get('url_image_backend', RANDOM_GENERATOR_API_PATH)

        # Wrap only specific methods
        self.embedding_request = self.cache_decorator()(conextions.embedding_request)
        self.search_by_vector = self.cache_decorator()(conextions.search_by_vector)
        self.search_by_id = self.cache_decorator()(conextions.search_by_id)
        self.search_by_scroll = self.cache_decorator()(conextions.search_scroll)
        self.tagger_by_file_request = self.cache_decorator()(conextions.tagger_by_file_request)
        self.get_vector_by_id_request = self.cache_decorator()(conextions.get_vector_by_id_request)
        self.tagger_by_vector_request = self.cache_decorator()(conextions.tagger_by_vector_request)
        self.random_search_request = self.cache_decorator()(conextions.random_search_request)
        self.search_by_string_request = self.cache_decorator()(conextions.search_by_string_request)
        self.embedding_clip_text_request = self.cache_decorator()(conextions.embedding_clip_text_request)

    async def search_by_ndarray_image_input(
            self,
            img_array: np.ndarray,
            **kwargs
    ) -> ImageSearchResponseModel:
        vector, tm = await self.embedding_request(
            img_array=img_array,
            url_api_embedding=self.url_api_embedding)
        return await self.search_by_vector(
            url=self.url_api_back_search,
            vector=vector,
            embedding_time=tm,
            **kwargs
        )

    async def generate_random_response(
            self,
            **kwargs
    ) -> ImageSearchResponseModel:
        return await self.random_search_request(
            embedding_time=0,
            url_image_backend=self.url_image_backend,
            **kwargs
        )

    async def search_by_vector_input(
            self,
            vector: list,
            **kwargs
    ) -> ImageSearchResponseModel:
        return await self.search_by_vector(
            url=self.url_api_back_search,
            vector=vector,
            embedding_time=0,
            **kwargs
        )

    async def search_by_booru_image_id(
            self,
            **kwargs
    ) -> ImageSearchResponseModel:
        return await self.search_by_id(
            url=self.url_api_back_search,
            embedding_time=0,
            **kwargs
        )

    async def search_by_string(
            self,
            text: str,
            **kwargs
    ) -> ImageSearchResponseModel:
        vector, tm = await self.embedding_clip_text_request(
            text=text,
            url_api_embedding=self.url_api_embedding
        )

        return await self.search_by_string_request(
            vector=vector,
            url=self.url_api_back_search,
            embedding_time=tm,
            **kwargs
        )

    async def search_scroll(
            self,
            **kwargs
    ) -> ImageSearchResponseModel:
        return await self.search_by_scroll(
            url=self.url_api_back_search,
            embedding_time=0,
            **kwargs
        )

    async def tagger_by_ndarray_input(
            self,
            img_array: np.ndarray,
            **kwargs
    ) -> TaggerResponseModel:
        tags, tm = await self.tagger_by_file_request(
            img_array=img_array,
            url_api_embedding=self.url_api_embedding)

        return build_tagger_response(
            tags=tags,
            embedding_time=tm,
            **kwargs
        )

    async def tagger_by_booru_image_id(
            self,
            id_vector: int,
            booru_name: str = None,
            **kwargs
    ) -> TaggerResponseModel:
        vector, tm1 = await self.get_vector_by_id_request(
            url=self.url_api_back_search,
            id_vector=id_vector,
            booru_name=booru_name
        )

        tags, tm2 = await self.tagger_by_vector_request(
            vector=vector,
            url_api_embedding=self.url_api_embedding)

        return build_tagger_response(
            tags=tags,
            embedding_time=tm1 + tm2,
            **kwargs
        )

    async def get_embedding_vector(
            self,
            img_array: np.ndarray,
    ) -> list[float]:
        vector, _ = await self.embedding_request(
            img_array=img_array,
            url_api_embedding=self.url_api_embedding)
        return vector
