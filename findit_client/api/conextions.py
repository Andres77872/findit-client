import time

import numpy as np
import requests
from ArZypher import arzypher_decoder

from findit_client.exceptions import EmbeddingException, RemoteRawSearchException
from findit_client.models.model_search import ImageSearchResponseModel
from findit_client.models.builder import build_search_response, build_random_search_response
from findit_client.util.image import compress_nparr, uncompress_nparr
from findit_client.api.const import (EMBEDDING_SEARCH_API_PATH,
                                     SEARCH_BY_VECTOR_API_PATH,
                                     RANDOM_GENERATOR_API_PATH,
                                     SEARCH_BY_ID_API_PATH,
                                     SEARCH_SCROLL_API_PATH,
                                     TAGGER_BY_FILE_API_PATH,
                                     EMBEDDING_GET_VECTOR_API_PATH,
                                     TAGGER_BY_VECTOR_API_PATH,
                                     BOORU_TO_ID, X_scroll_arzypher_params)


# def wtime(func):
#     def wrapper(*args, **kwargs):
#         st = time.time()
#         print(kwargs)
#         num_sum = func(*args, **kwargs)
#         print(time.time() - st)
#         return num_sum
#
#     return wrapper
#

def embedding_request(
        img_array: np.ndarray,
        url_api_embedding: str
) -> tuple[list, float]:
    st = time.time()
    url = url_api_embedding + EMBEDDING_SEARCH_API_PATH
    resp = requests.post(url=url, files={'obj': compress_nparr(img_array)})
    if resp.status_code != 200:
        raise EmbeddingException(origin=url)
    return uncompress_nparr(resp.content)[0].tolist(), time.time() - st


def random_search_request(
        total: int = 32,
        pool: list[str] = None,
        content: str = 'g',
        **kwargs
) -> ImageSearchResponseModel:
    st = time.time()

    if type(total) == str:
        total = int(total)
    if not 0 < total <= 128:
        total = 32

    g = f'?total={total}&'
    if pool is not None:
        g += 'booru=' + ','.join([str(BOORU_TO_ID[x]) for x in pool]) + '&'
    if content is not None:
        g += 'content=' + content + '&'

    resp = requests.get(RANDOM_GENERATOR_API_PATH + g)
    if resp.status_code != 200:
        raise RemoteRawSearchException(origin=RANDOM_GENERATOR_API_PATH + g)
    results = resp.json()
    tm = time.time() - st
    return build_random_search_response(
        results=results,
        latency_search=tm,
        **kwargs
    )


def search_by_vector(
        url: str,
        **kwargs
) -> ImageSearchResponseModel:
    st = time.time()
    url = url + SEARCH_BY_VECTOR_API_PATH
    resp = requests.post(url, json=kwargs)
    if resp.status_code != 200:
        raise RemoteRawSearchException(origin=url)
    results = resp.json()
    tm = time.time() - st - results['time']
    return build_search_response(
        results=results,
        latency_search=tm,
        **kwargs
    )


def search_by_id(
        url: str,
        **kwargs
) -> ImageSearchResponseModel:
    st = time.time()
    url = url + SEARCH_BY_ID_API_PATH
    resp = requests.post(url, json=kwargs)
    if resp.status_code != 200:
        raise RemoteRawSearchException(origin=url)
    results = resp.json()
    tm = time.time() - st - results['time']
    return build_search_response(
        results=results,
        latency_search=tm,
        **kwargs
    )


def search_scroll(
        url: str,
        scroll_token: str,
        limit: str,
        **kwargs
) -> ImageSearchResponseModel:
    content, _ = arzypher_decoder(**X_scroll_arzypher_params,
                                  encoded=scroll_token)

    boorus_index = [
        content[3] if content[2] == 1 else -1,
        content[5] if content[4] == 1 else -1,
        content[7] if content[6] == 1 else -1,
        content[9] if content[8] == 1 else -1,
        content[11] if content[10] == 1 else -1,
        content[13] if content[12] == 1 else -1,
        content[15] if content[14] == 1 else -1,
    ]

    js = {
        'vector_id': content[0],
        'count': content[1],
        'limit': limit,
        'boorus_index': boorus_index
    }

    st = time.time()
    url = url + SEARCH_SCROLL_API_PATH
    resp = requests.post(url, json=js)
    if resp.status_code != 200:
        raise RemoteRawSearchException(origin=url)
    results = resp.json()
    tm = time.time() - st - results['time']
    return build_search_response(
        results=results,
        latency_search=tm,
        **kwargs
    )


def tagger_by_file_request(
        img_array: np.ndarray,
        url_api_embedding: str
) -> tuple[list, float]:
    st = time.time()
    url = url_api_embedding + TAGGER_BY_FILE_API_PATH
    resp = requests.post(url=url, files={'obj': compress_nparr(img_array)})
    if resp.status_code != 200:
        raise EmbeddingException(origin=url)
    return uncompress_nparr(resp.content)[0].tolist(), time.time() - st


def tagger_by_vector_request(
        vector: np.ndarray,
        url_api_embedding: str
) -> tuple[list, float]:
    st = time.time()
    url = url_api_embedding + TAGGER_BY_VECTOR_API_PATH
    resp = requests.post(url=url, files={'obj': compress_nparr(vector)})
    if resp.status_code != 200:
        raise EmbeddingException(origin=url)
    return uncompress_nparr(resp.content)[0].tolist(), time.time() - st


def get_vector_by_id_request(
        url: str,
        id_vector: int,
        pool: str
) -> tuple[np.ndarray, float]:
    st = time.time()
    url = url + EMBEDDING_GET_VECTOR_API_PATH
    resp = requests.post(url, json={'id_vector': id_vector, 'pool': pool})
    if resp.status_code != 200:
        raise RemoteRawSearchException(origin=url)
    results = resp.json()
    tm = time.time() - st
    return np.array(results)[None], tm
