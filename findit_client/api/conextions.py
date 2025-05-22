import aiohttp
import numpy as np
from ArZypher import arzypher_decoder

from findit_client.api.const import (
    EMBEDDING_SEARCH_API_PATH,
    SEARCH_BY_VECTOR_API_PATH,
    SEARCH_BY_ID_API_PATH,
    SEARCH_SCROLL_API_PATH,
    TAGGER_BY_FILE_API_PATH,
    EMBEDDING_GET_VECTOR_BY_BOORU_API_PATH,
    TAGGER_BY_VECTOR_API_PATH,
    X_scroll_arzypher_params,
    EMBEDDING_GET_VECTOR_CLIP_TEXT_API_PATH,
    EMBEDDING_GET_VECTOR_BY_ID_FILE_API_PATH,
)
from findit_client.exceptions import (
    EmbeddingException,
    RemoteRawSearchException,
    QueryCantBeDecodedException,
)
from findit_client.models.builder import (
    build_search_response,
    build_random_search_response,
)
from findit_client.models.model_search import ImageSearchResponseModel
from findit_client.util.image import compress_nparr, uncompress_nparr
from findit_client.util.validations import validate_params

sess: aiohttp.ClientSession = None


async def ensure_session():
    global sess
    await init_sess()


async def init_sess():
    global sess
    sess = aiohttp.ClientSession(
        headers={'User-Agent': 'findit.moe client -> https://findit.moe'}
    )


async def close_sess():
    await sess.close()


async def search_response(url: str, js: dict = None, **kwargs) -> ImageSearchResponseModel | None:
    j = {}
    if js:
        j.update(js)
    if kwargs:
        j.update(kwargs)
    await ensure_session()
    async with sess.post(url, json=j) as resp:
        if resp.status == 200:
            results = await resp.json()
            elapsed = resp.elapsed.total_seconds() if hasattr(resp, 'elapsed') else 0
            tm = elapsed - results.get('time', 0)
            return build_search_response(
                results=results,
                latency_search=tm,
                **kwargs
            )
    return None


async def nn_model_request(url: str, nparr: np.ndarray) -> tuple[list, float] | None:
    data = compress_nparr(nparr)
    form = aiohttp.FormData()
    form.add_field('obj', data)
    await ensure_session()
    async with sess.post(url=url, data=form) as resp:
        if resp.status == 200:
            content = await resp.read()
            elapsed = resp.elapsed.total_seconds() if hasattr(resp, 'elapsed') else 0
            return uncompress_nparr(content)[0].tolist(), elapsed
    return None


async def embedding_request(
        img_array: np.ndarray,
        url_api_embedding: str
) -> tuple[list, float]:
    url = url_api_embedding + EMBEDDING_SEARCH_API_PATH
    rp = await nn_model_request(url, img_array)
    if rp:
        return rp
    raise EmbeddingException(origin=url)


async def embedding_clip_text_request(
        text: str,
        url_api_embedding: str
) -> tuple[list, float]:
    url = url_api_embedding + EMBEDDING_GET_VECTOR_CLIP_TEXT_API_PATH
    data = {'text': text}
    await ensure_session()
    async with sess.post(url=url, data=data) as resp:
        if resp.status == 200:
            vec = await resp.json()
            elapsed = resp.elapsed.total_seconds() if hasattr(resp, 'elapsed') else 0
            return vec, elapsed
    raise EmbeddingException(origin=url)


async def tagger_by_file_request(
        img_array: np.ndarray,
        url_api_embedding: str
) -> tuple[list, float]:
    url = url_api_embedding + TAGGER_BY_FILE_API_PATH
    rp = await nn_model_request(url, img_array)
    if rp:
        return rp
    raise EmbeddingException(origin=url)


async def tagger_by_vector_request(
        vector: np.ndarray,
        url_api_embedding: str
) -> tuple[list, float]:
    url = url_api_embedding + TAGGER_BY_VECTOR_API_PATH
    rp = await nn_model_request(url, vector)
    if rp:
        return rp
    raise EmbeddingException(origin=url)


@validate_params
async def random_search_request(
        url_image_backend: str,
        limit: int = 32,
        pool: list[str] = None,
        content: str = 'g',
        **kwargs
) -> ImageSearchResponseModel:
    g = f'?total={limit}&'
    if pool is not None:
        g += 'booru=' + ','.join(pool) + '&'
    if content is not None:
        g += 'content=' + content + '&'
    await ensure_session()
    async with sess.get(url_image_backend + '/random' + g) as resp:
        if resp.status == 200:
            results = await resp.json()
            elapsed = resp.elapsed.total_seconds() if hasattr(resp, 'elapsed') else 0
            return build_random_search_response(
                results=results,
                latency_search=elapsed,
                **kwargs
            )
    raise RemoteRawSearchException(origin=url_image_backend + g)


@validate_params
async def search_by_vector(
        url: str,
        **kwargs
) -> ImageSearchResponseModel:
    sh = await search_response(url + SEARCH_BY_VECTOR_API_PATH, **kwargs)
    if sh:
        return sh
    raise RemoteRawSearchException(origin=url)


@validate_params
async def search_by_string_request(
        url: str,
        **kwargs
) -> ImageSearchResponseModel:
    sh = await search_response(url + SEARCH_BY_VECTOR_API_PATH, **kwargs)
    if sh:
        return sh
    raise RemoteRawSearchException(origin=url)


@validate_params
async def search_by_id(
        url: str,
        **kwargs
) -> ImageSearchResponseModel:
    sh = await search_response(url + SEARCH_BY_ID_API_PATH, **kwargs)
    if sh:
        return sh
    raise RemoteRawSearchException(origin=url)


@validate_params
async def search_scroll(
        url: str,
        scroll_token: str,
        limit: str,
        rating: list,
        **kwargs
) -> ImageSearchResponseModel:
    content, _ = arzypher_decoder(**X_scroll_arzypher_params,
                                  encoded=scroll_token)
    if content == [0]:
        raise QueryCantBeDecodedException(query=scroll_token)

    js = {
        'vector_id': content[0],
        'count': content[1],
        'limit': limit,
        'scroll': content[1],
        'rating': rating
    }

    sh = await search_response(url + SEARCH_SCROLL_API_PATH, js, **kwargs)
    if sh:
        return sh
    raise RemoteRawSearchException(origin=url)


@validate_params
async def get_vector_by_id_request(
        url: str,
        id_vector: int,
        booru_name: str
) -> tuple[np.ndarray, float]:
    await ensure_session()
    if booru_name:
        url = url + EMBEDDING_GET_VECTOR_BY_BOORU_API_PATH
        payload = {'id_vector': id_vector, 'pool': booru_name}
        async with sess.post(url, json=payload) as resp:
            if resp.status == 200:
                results = await resp.json()
                elapsed = resp.elapsed.total_seconds() if hasattr(resp, 'elapsed') else 0
                return np.array(results)[None], elapsed
    else:
        url = url + EMBEDDING_GET_VECTOR_BY_ID_FILE_API_PATH + '/' + str(id_vector)
        async with sess.post(url) as resp:
            if resp.status == 200:
                results = await resp.json()
                elapsed = resp.elapsed.total_seconds() if hasattr(resp, 'elapsed') else 0
                return np.array(results)[None], elapsed
    raise RemoteRawSearchException(origin=url)
