import functools

import aiohttp
import numpy as np

from findit_client import util
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
            session: aiohttp.ClientSession = None,
            **kwargs
    ):
        self.cache_decorator = cache_decorator or async_cache_function
        self.url_api_embedding = url_api_embedding
        self.url_api_back_search = url_api_back_search
        self.url_image_backend = kwargs.get('url_image_backend', RANDOM_GENERATOR_API_PATH)

        # Store the provided session or create a new one
        self.sess = session
        self._own_session = session is None  # Track if we created our own session

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
        self._load_url_image = self.cache_decorator()(util.load_url_image)
        self._build_masonry_collage = self.cache_decorator()(util.build_masonry_collage)

    async def initialize(self):
        """Initialize the client session if not already provided."""
        if self.sess is None:
            self.sess = aiohttp.ClientSession(
                headers={'User-Agent': 'findit.moe client -> https://findit.moe'}
            )
            self._own_session = True
        return self

    async def close(self):
        """Close the session if we created it."""
        if self._own_session and self.sess is not None:
            await self.sess.close()
            self.sess = None

    async def __aenter__(self):
        """Support using this class as an async context manager."""
        return await self.initialize()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting the context manager."""
        await self.close()

    async def _ensure_session(self):
        """Ensure a session exists before making requests"""
        if self.sess is None:
            await self.initialize()

    async def search_by_ndarray_image_input(
            self,
            img_array: np.ndarray,
            **kwargs
    ) -> ImageSearchResponseModel:
        await self._ensure_session()
        vector, tm = await self.embedding_request(
            session=self.sess,
            img_array=img_array,
            url_api_embedding=self.url_api_embedding)
        return await self.search_by_vector(
            session=self.sess,
            url=self.url_api_back_search,
            vector=vector,
            embedding_time=tm,
            **kwargs
        )

    async def generate_random_response(
            self,
            **kwargs
    ) -> ImageSearchResponseModel:
        await self._ensure_session()
        return await self.random_search_request(
            session=self.sess,
            embedding_time=0,
            url_image_backend=self.url_image_backend,
            **kwargs
        )

    async def search_by_vector_input(
            self,
            vector: list,
            **kwargs
    ) -> ImageSearchResponseModel:
        await self._ensure_session()
        return await self.search_by_vector(
            session=self.sess,
            url=self.url_api_back_search,
            vector=vector,
            embedding_time=0,
            **kwargs
        )

    async def search_by_booru_image_id(
            self,
            **kwargs
    ) -> ImageSearchResponseModel:
        await self._ensure_session()
        return await self.search_by_id(
            session=self.sess,
            url=self.url_api_back_search,
            embedding_time=0,
            **kwargs
        )

    async def search_by_string(
            self,
            text: str,
            **kwargs
    ) -> ImageSearchResponseModel:
        await self._ensure_session()
        vector, tm = await self.embedding_clip_text_request(
            session=self.sess,
            text=text,
            url_api_embedding=self.url_api_embedding
        )

        return await self.search_by_string_request(
            session=self.sess,
            vector=vector,
            url=self.url_api_back_search,
            embedding_time=tm,
            **kwargs
        )

    async def search_scroll(
            self,
            **kwargs
    ) -> ImageSearchResponseModel:
        await self._ensure_session()
        return await self.search_by_scroll(
            session=self.sess,
            url=self.url_api_back_search,
            embedding_time=0,
            **kwargs
        )

    async def tagger_by_ndarray_input(
            self,
            img_array: np.ndarray,
            **kwargs
    ) -> TaggerResponseModel:
        await self._ensure_session()
        tags, tm = await self.tagger_by_file_request(
            session=self.sess,
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
        await self._ensure_session()
        vector, tm1 = await self.get_vector_by_id_request(
            session=self.sess,
            url=self.url_api_back_search,
            id_vector=id_vector,
            booru_name=booru_name
        )

        tags, tm2 = await self.tagger_by_vector_request(
            session=self.sess,
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
        await self._ensure_session()
        vector, _ = await self.embedding_request(
            session=self.sess,
            img_array=img_array,
            url_api_embedding=self.url_api_embedding)
        return vector

    async def load_url_image(
            self,
            image: str,
            **kwargs
    ):
        await self._ensure_session()
        return await self._load_url_image(image=image, sess=self.sess, **kwargs)

    async def build_masonry_collage(
            self,
            **kwargs
    ):
        await self._ensure_session()
        return await self._build_masonry_collage(sess=self.sess, **kwargs)
