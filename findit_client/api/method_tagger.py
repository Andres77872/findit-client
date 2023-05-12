from ArZypher import arzypher_decoder

from findit_client.api.const import BOORUS_NAMES_STR, ID_TO_BOORU, X_query_arzypher_params_
from findit_client.exceptions import SearchBooruNotFound
from findit_client.models import ImageSearchResponseModel
from findit_client.models.model_tagger import TaggerResponseModel
from findit_client.util import load_file_image, load_url_image, load_bytes_image
from findit_client.api.api_requests import ApiRequests


class FindItMethodsTagger:
    def __init__(self,
                 __version__: str,
                 **kwargs):
        self.ApiRequests = ApiRequests(**kwargs)
        self.__version__ = __version__

    def by_file(
            self,
            img: str,
            th_rating: float = 0.35,
            th_character: float = 0.8,
            th_general: float = 0.5
    ) -> TaggerResponseModel:
        img_array, tm = load_file_image(img)
        return self.ApiRequests.tagger_by_ndarray_input(
            img_array=img_array,
            th_rating=th_rating,
            th_character=th_character,
            th_general=th_general,
            mode='FILE',
            load_image_time=tm,
            api_version=self.__version__
        )

    def by_url(
            self,
            url: str,
            th_rating: float = 0.35,
            th_character: float = 0.8,
            th_general: float = 0.5
    ) -> TaggerResponseModel:
        img_array, tm = load_url_image(url)
        return self.ApiRequests.tagger_by_ndarray_input(
            img_array=img_array,
            th_rating=th_rating,
            th_character=th_character,
            th_general=th_general,
            mode='URL',
            load_image_time=tm,
            api_version=self.__version__
        )

    def by_image_bytes(
            self,
            img: bytes,
            th_rating: float = 0.35,
            th_character: float = 0.8,
            th_general: float = 0.5
    ) -> TaggerResponseModel:
        img_array, tm = load_bytes_image(img)
        return self.ApiRequests.tagger_by_ndarray_input(
            img_array=img_array,
            th_rating=th_rating,
            th_character=th_character,
            th_general=th_general,
            mode='BIN_FILE',
            load_image_time=tm,
            api_version=self.__version__
        )

    def by_booru_image_id(
            self,
            booru_name: str,
            image_id: int,
            th_rating: float = 0.35,
            th_character: float = 0.8,
            th_general: float = 0.5
    ) -> TaggerResponseModel:
        if booru_name not in BOORUS_NAMES_STR:
            raise SearchBooruNotFound
        return self.ApiRequests.tagger_by_booru_image_id(
            id_vector=image_id,
            pool=booru_name,
            th_rating=th_rating,
            th_character=th_character,
            th_general=th_general,
            mode='BOORU_IMAGE_ID',
            load_image_time=0,
            api_version=self.__version__
        )

    def by_query(
            self,
            query: str,
            th_rating: float = 0.35,
            th_character: float = 0.8,
            th_general: float = 0.5
    ) -> TaggerResponseModel:
        (booru_id, image_id), _ = arzypher_decoder(**X_query_arzypher_params_,
                                                   encoded=query)
        return self.ApiRequests.tagger_by_booru_image_id(
            id_vector=image_id,
            pool=ID_TO_BOORU[booru_id],
            th_rating=th_rating,
            th_character=th_character,
            th_general=th_general,
            mode='QUERY',
            load_image_time=0,
            api_version=self.__version__
        )
