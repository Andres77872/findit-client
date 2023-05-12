from findit_client.api.const import BOORUS_NAMES_STR
from findit_client.models import ImageSearchResponseModel
from findit_client.api.api_requests import ApiRequests


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
        if pool is None:
            pool = BOORUS_NAMES_STR
        return self.ApiRequests.generate_random_response(
            pool=pool,
            total=limit,
            content=content,
            mode='RANDOM',
            load_image_time=0,
            api_version=self.__version__
        )
