import numpy as np

from findit_client.api.conextions import (search_by_vector,
                                          embedding_request,
                                          search_by_id,
                                          search_scroll,
                                          tagger_by_file_request,
                                          get_vector_by_id_request,
                                          tagger_by_vector_request,
                                          random_search_request,
                                          search_by_string_request,
                                          embedding_clip_text_request)
from findit_client.models import ImageSearchResponseModel
from findit_client.models.builder import build_tagger_response
from findit_client.models.model_tagger import TaggerResponseModel


class ApiRequests:
    def __init__(
            self,
            url_api_embedding: str,
            url_api_back_search: str,
            **kwargs
    ):
        self.url_api_embedding = url_api_embedding
        self.url_api_back_search = url_api_back_search

    def search_by_ndarray_image_input(
            self,
            img_array: np.ndarray,
            **kwargs
    ) -> ImageSearchResponseModel:
        vector, tm = embedding_request(
            img_array=img_array,
            url_api_embedding=self.url_api_embedding)
        return search_by_vector(
            url=self.url_api_back_search,
            vector=vector,
            embedding_time=tm,
            **kwargs
        )

    def generate_random_response(
            self,
            **kwargs
    ) -> ImageSearchResponseModel:
        return random_search_request(
            embedding_time=0,
            **kwargs
        )

    def search_by_vector_input(
            self,
            vector: list,
            **kwargs
    ) -> ImageSearchResponseModel:
        return search_by_vector(
            url=self.url_api_back_search,
            vector=vector,
            embedding_time=0,
            **kwargs
        )

    def search_by_booru_image_id(
            self,
            **kwargs
    ) -> ImageSearchResponseModel:
        return search_by_id(
            url=self.url_api_back_search,
            embedding_time=0,
            **kwargs
        )

    def search_by_string(
            self,
            text: str,
            **kwargs
    ) -> ImageSearchResponseModel:
        vector, tm = embedding_clip_text_request(text=text,
                                                 url_api_embedding=self.url_api_embedding)

        return search_by_string_request(
            vector=vector,
            url=self.url_api_back_search,
            embedding_time=tm,
            **kwargs
        )

    def search_scroll(
            self,
            **kwargs
    ) -> ImageSearchResponseModel:
        return search_scroll(
            url=self.url_api_back_search,
            embedding_time=0,
            **kwargs
        )

    def tagger_by_ndarray_input(
            self,
            img_array: np.ndarray,
            **kwargs
    ) -> TaggerResponseModel:
        tags, tm = tagger_by_file_request(
            img_array=img_array,
            url_api_embedding=self.url_api_embedding)

        return build_tagger_response(
            tags=tags,
            embedding_time=tm,
            **kwargs
        )

    def tagger_by_booru_image_id(
            self,
            id_vector: int,
            booru_name: str = None,
            **kwargs
    ) -> TaggerResponseModel:
        vector, tm1 = get_vector_by_id_request(
            url=self.url_api_back_search,
            id_vector=id_vector,
            booru_name=booru_name
        )

        tags, tm2 = tagger_by_vector_request(
            vector=vector,
            url_api_embedding=self.url_api_embedding)

        return build_tagger_response(
            tags=tags,
            embedding_time=tm1 + tm2,
            **kwargs
        )

    def get_embedding_vector(
            self,
            img_array: np.ndarray,
    ) -> list[float]:
        vector, _ = embedding_request(
            img_array=img_array,
            url_api_embedding=self.url_api_embedding)
        return vector
