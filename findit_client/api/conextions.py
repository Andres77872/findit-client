import numpy as np
import requests
from ArZypher import arzypher_decoder

from findit_client.exceptions import (EmbeddingException,
                                      RemoteRawSearchException,
                                      QueryCantBeDecodedException)
from findit_client.models.model_search import ImageSearchResponseModel
from findit_client.models.builder import (build_search_response,
                                          build_random_search_response)
from findit_client.util.image import compress_nparr, uncompress_nparr
from findit_client.api.const import (EMBEDDING_SEARCH_API_PATH,
                                     SEARCH_BY_VECTOR_API_PATH,
                                     RANDOM_GENERATOR_API_PATH,
                                     SEARCH_BY_ID_API_PATH,
                                     SEARCH_SCROLL_API_PATH,
                                     TAGGER_BY_FILE_API_PATH,
                                     EMBEDDING_GET_VECTOR_API_PATH,
                                     TAGGER_BY_VECTOR_API_PATH,
                                     BOORU_TO_ID,
                                     X_scroll_arzypher_params,
                                     EMBEDDING_GET_VECTOR_TEXT_API_PATH)
from findit_client.util.validations import validate_params

sess = requests.Session()
sess.verify = True
sess.headers['User-Agent'] = 'findit.moe client -> https://findit.moe'


def search_response(url: str,
                    js: dict = None,
                    scroll_content: list = None,
                    use_sem: bool = False,
                    **kwargs) -> ImageSearchResponseModel | None:
    j = js if js else kwargs
    j['use_sem'] = use_sem
    if (resp := sess.post(url, json=j)) and resp.status_code == 200:
        results = resp.json()
        # print(results)
        tm = resp.elapsed.microseconds / 1000000 - results['time']
        return build_search_response(
            results=results,
            latency_search=tm,
            scroll_content=scroll_content,
            **kwargs
        )
    return None


def nn_model_request(url: str, nparr: np.ndarray) -> tuple[list, float] | None:
    if (resp := sess.post(url=url, files={'obj': compress_nparr(nparr)})) and resp.status_code == 200:
        return uncompress_nparr(resp.content)[0].tolist(), resp.elapsed.microseconds / 1000000
    return None


def embedding_request(
        img_array: np.ndarray,
        url_api_embedding: str
) -> tuple[list, float]:
    url = url_api_embedding + EMBEDDING_SEARCH_API_PATH
    if (rp := nn_model_request(url, img_array)) and rp:
        return rp
    raise EmbeddingException(origin=url)


def embedding_text_request(
        text: str,
        url_api_embedding: str
) -> tuple[list, float]:
    url = url_api_embedding + EMBEDDING_GET_VECTOR_TEXT_API_PATH
    if (resp := sess.post(url=url, data={'text': text})) and resp.status_code == 200:
        return resp.json(), resp.elapsed.microseconds / 1000000
    raise EmbeddingException(origin=url)


def tagger_by_file_request(
        img_array: np.ndarray,
        url_api_embedding: str
) -> tuple[list, float]:
    url = url_api_embedding + TAGGER_BY_FILE_API_PATH
    if (rp := nn_model_request(url, img_array)) and rp:
        return rp
    raise EmbeddingException(origin=url)


def tagger_by_vector_request(
        vector: np.ndarray,
        url_api_embedding: str
) -> tuple[list, float]:
    url = url_api_embedding + TAGGER_BY_VECTOR_API_PATH
    if (rp := nn_model_request(url, vector)) and rp:
        return rp
    raise EmbeddingException(origin=url)


@validate_params
def random_search_request(
        limit: int = 32,
        pool: list[str] = None,
        content: str = 'g',
        **kwargs
) -> ImageSearchResponseModel:
    g = f'?total={limit}&'
    if pool is not None:
        g += 'booru=' + ','.join([str(BOORU_TO_ID[x]) for x in pool]) + '&'
    if content is not None:
        g += 'content=' + content + '&'

    if (resp := sess.get(RANDOM_GENERATOR_API_PATH + g)) and resp.status_code == 200:
        results = resp.json()
        return build_random_search_response(
            results=results,
            latency_search=resp.elapsed.microseconds / 1000000,
            **kwargs
        )
    raise RemoteRawSearchException(origin=RANDOM_GENERATOR_API_PATH + g)


@validate_params
def search_by_vector(
        url: str,
        **kwargs
) -> ImageSearchResponseModel:
    if (sh := search_response(url + SEARCH_BY_VECTOR_API_PATH, **kwargs)) and sh:
        return sh
    raise RemoteRawSearchException(origin=url)


@validate_params
def search_by_string_request(
        url: str,
        **kwargs
) -> ImageSearchResponseModel:
    if (sh := search_response(url + SEARCH_BY_VECTOR_API_PATH, **kwargs)) and sh:
        return sh
    raise RemoteRawSearchException(origin=url)


@validate_params
def search_by_id(
        url: str,
        **kwargs
) -> ImageSearchResponseModel:
    if (sh := search_response(url + SEARCH_BY_ID_API_PATH, **kwargs)) and sh:
        return sh
    raise RemoteRawSearchException(origin=url)


@validate_params
def search_scroll(
        url: str,
        scroll_token: str,
        limit: str,
        **kwargs
) -> ImageSearchResponseModel:
    content, _ = arzypher_decoder(**X_scroll_arzypher_params,
                                  encoded=scroll_token)
    if content == [0]:
        raise QueryCantBeDecodedException(query=scroll_token)

    # print(content)

    boorus_index = [content[i + 1] if content[i] == 1 else -1 for i in range(2, len(content), 2)]

    # scroll_content = [content[i + 1] if content[i] == 1 else 0 for i in range(2, len(content), 2)]

    # print(boorus_index)
    # print(scroll_content)

    js = {
        'vector_id': content[0],
        'count': content[1],
        'limit': limit,
        'boorus_index': boorus_index
    }

    # print(js)

    if (sh := search_response(url + SEARCH_SCROLL_API_PATH, js, boorus_index, **kwargs)) and sh:
        return sh
    raise RemoteRawSearchException(origin=url)


@validate_params
def get_vector_by_id_request(
        url: str,
        id_vector: int,
        booru_name: str
) -> tuple[np.ndarray, float]:
    url = url + EMBEDDING_GET_VECTOR_API_PATH
    if (resp := sess.post(url, json={'id_vector': id_vector, 'pool': booru_name})) and resp.status_code == 200:
        results = resp.json()
        return np.array(results)[None], resp.elapsed.microseconds / 1000000
    raise RemoteRawSearchException(origin=url)
