from ArZypher import arzypher_decoder

from findit_client.api.api_requests import ApiRequests
from findit_client.api.const import ID_TO_BOORU, X_query_arzypher_params
from findit_client.exceptions import QueryCantBeDecodedException
from findit_client.models import ImageSearchResponseModel
from findit_client.util import load_file_image, load_url_image, load_bytes_image


class FindItMethodsSearch:
    def __init__(self,
                 __version__: str,
                 **kwargs):
        self.ApiRequests = ApiRequests(**kwargs)
        self.__version__ = __version__

    async def by_file(
            self,
            img: str | list[str],
            pool: list[str] = None,
            limit: int = 32,
            rating: list[str] = None,
            **kwargs
    ) -> ImageSearchResponseModel:
        img_array, tm = await load_file_image(img)
        return await self.ApiRequests.search_by_ndarray_image_input(
            img_array=img_array,
            pool=pool,
            limit=limit,
            rating=rating,
            mode='FILE',
            load_image_time=tm,
            api_version=self.__version__,
            **kwargs
        )

    async def by_url(
            self,
            url: str,
            pool: list[str] = None,
            limit: int = 32,
            rating: list[str] = None,
            **kwargs
    ) -> ImageSearchResponseModel:
        img_array, tm = await load_url_image(url)
        return await self.ApiRequests.search_by_ndarray_image_input(
            img_array=img_array,
            pool=pool,
            limit=limit,
            rating=rating,
            mode='URL',
            load_image_time=tm,
            api_version=self.__version__,
            **kwargs
        )

    async def by_image_bytes(
            self,
            img: bytes,
            pool: list[str] = None,
            limit: int = 32,
            rating: list[str] = None,
            **kwargs
    ) -> ImageSearchResponseModel:
        img_array, tm = await load_bytes_image(img)
        return await self.ApiRequests.search_by_ndarray_image_input(
            img_array=img_array,
            pool=pool,
            limit=limit,
            rating=rating,
            mode='BIN_FILE',
            load_image_time=tm,
            api_version=self.__version__,
            **kwargs
        )

    async def by_booru_image_id(
            self,
            booru_name: str,
            image_id: int,
            pool: list[str] = None,
            limit: int = 32,
            rating: list[str] = None,
            **kwargs
    ) -> ImageSearchResponseModel:
        return await self.ApiRequests.search_by_booru_image_id(
            id_vector=image_id,
            pool_vector=booru_name,
            pool=pool,
            limit=limit,
            rating=rating,
            mode='QUERY',
            load_image_time=0,
            api_version=self.__version__,
            **kwargs
        )

    async def by_query(
            self,
            query: str,
            pool: list[str] = None,
            limit: int = 32,
            rating: list[str] = None,
            **kwargs
    ) -> ImageSearchResponseModel:
        dec, _ = arzypher_decoder(**X_query_arzypher_params,
                                  encoded=query)
        if dec == [0]:
            raise QueryCantBeDecodedException(query=query)
        (booru_id, image_id) = dec

        return await self.ApiRequests.search_by_booru_image_id(
            id_vector=image_id,
            pool_vector=ID_TO_BOORU[booru_id],
            pool=pool,
            limit=limit,
            rating=rating,
            mode='QUERY',
            load_image_time=0,
            api_version=self.__version__,
            **kwargs
        )

    async def by_vector(
            self,
            vector: list,
            pool: list[str] = None,
            limit: int = 32,
            rating: list[str] = None,
            **kwargs
    ) -> ImageSearchResponseModel:
        return await self.ApiRequests.search_by_vector_input(
            vector=vector,
            pool=pool,
            limit=limit,
            rating=rating,
            mode='VECTOR',
            load_image_time=0,
            api_version=self.__version__,
            **kwargs
        )

    async def by_text(
            self,
            text: str,
            pool: list[str] = None,
            limit: int = 32,
            rating: list[str] = None,
            **kwargs
    ) -> ImageSearchResponseModel:
        return await self.ApiRequests.search_by_string(
            use_sem=True,
            text=text,
            pool=pool,
            limit=limit,
            rating=rating,
            mode='TEXT',
            load_image_time=0,
            api_version=self.__version__,
            **kwargs
        )

    async def scroll(
            self,
            scroll_token: str,
            limit: int = 32,
            rating: list[str] = None,
            **kwargs
    ) -> ImageSearchResponseModel:
        return await self.ApiRequests.search_scroll(
            limit=limit,
            rating=rating,
            scroll_token=scroll_token,
            mode='SCROLL',
            load_image_time=0,
            api_version=self.__version__,
            **kwargs
        )
