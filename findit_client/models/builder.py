import json
import time

import requests
from ArZypher import arzypher_encoder
from findit_client.api.const import (BOORU_TO_ID,
                                     BOORU_SOURCE_URL,
                                     URL_IMAGE_PROVIDER,
                                     X_image_arzypher_params,
                                     X_query_arzypher_params,
                                     X_scroll_arzypher_params,
                                     ID_TO_BOORU)
from findit_client.models import ImageSearchResponseModel
from findit_client.models.model_search import ImageSearchResultRaw
from findit_client.models.model_tagger import TaggerResponseModel


def build_search_response(results: dict,
                          warning=None,
                          **kwargs) -> ImageSearchResponseModel:
    if warning is None:
        warning = []
    st = time.time()
    rl = []
    for i in results['results']['data']:
        _r = []
        p224, _ = arzypher_encoder(**X_image_arzypher_params,
                                   params_data=[i[0], 0])
        p512, _ = arzypher_encoder(**X_image_arzypher_params,
                                   params_data=[i[0], 1])
        for _p in i[2]:
            query, _ = arzypher_encoder(**X_query_arzypher_params,
                                        params_data=[_p['id_booru'], _p['post_id']])
            _r.append({
                'id': _p['post_id'],
                'source': BOORU_SOURCE_URL[_p['id_booru'] - 1].format(_p['post_id']),
                'preview': f'{URL_IMAGE_PROVIDER[0]}/{p224}',
                'img': f'{URL_IMAGE_PROVIDER[0]}/{p512}',
                'score': i[1],
                'pool': ID_TO_BOORU[_p['id_booru']],
                'query': query,
                'size': i[4],
                'color': i[5]
            })
        rl.append({'content': _r, 'vector': i[3]})

    ok_count = [
        results['results']['search_id'],
        results['results']['count_total']
    ]

    scroll_token, _ = arzypher_encoder(**X_scroll_arzypher_params,
                                       params_data=ok_count)

    dc = {
        'search_meta': ImageSearchResultRaw(**results),
        'scroll_token': scroll_token,
        'status': {
            'code': 'OK' if len(warning) == 0 else 'WARNING',
            'msg': warning
        },
        'post_process_time': time.time() - st,
        'results': {
            'count': len(rl),
            'data': rl
        },
        **kwargs
    }
    return ImageSearchResponseModel(**dc)


def build_random_search_response(results: list[dict],
                                 warning=None,
                                 **kwargs) -> ImageSearchResponseModel:
    if warning is None:
        warning = []
    st = time.time()
    rl = []
    for _p in results:
        _r = []
        query, _ = arzypher_encoder(**X_query_arzypher_params,
                                    params_data=[BOORU_TO_ID[_p['X-Booru-name']], int(_p['X-Image-Id'])])
        _r.append({
            'id': _p['X-Image-Id'],
            'source': BOORU_SOURCE_URL[BOORU_TO_ID[_p['X-Booru-name']] - 1].format(_p['X-Image-Id']),
            'preview': _p['url']['224'],
            'img': _p['url']['512'],
            'score': 0,
            'pool': _p['X-Booru-name'],
            'query': query,
            'size': [_p['X-Width'], _p['X-Height']],
            'color': _p['X-Color']
        })
        rl.append({'content': _r, 'vector': []})

    dc = {
        'search_meta': {
            'time': 0,
            'latency_search': 0,
            'search_version': '0',
            'status': {'code': 'OK', 'msg': []},
            'time_groping': 0,
            'qdrant_meta': {
                'time': 0,
                'qdrant_version': '0',
                'status': {'code': 'OK', 'msg': []},
                'config': {
                    'limit': 0,
                    'pools': [],
                    'vector': []
                }
            }
        },
        'scroll_token': '',
        'status': {
            'code': 'OK' if len(warning) == 0 else 'WARNING',
            'msg': warning
        },
        'post_process_time': time.time() - st,
        'results': {
            'count': len(rl),
            'data': rl
        },
        **kwargs
    }
    return ImageSearchResponseModel(**dc)


resp = requests.get('https://models.arz.ai/tags_SW_CN_V2.tf.json')
tags = json.loads(resp.text)
tags_id = {tags[x]: x for x in tags}


def tag_parse(r: list, threshold: float, start: int) -> list:
    a = [(x + start, y) for x, y in enumerate(r) if y >= threshold]

    data = [[tags_id[x], float(y)] for x, y in a]
    data.sort(key=lambda x: x[1])

    data_dc = [{'tag': t, 'score': s} for t, s in data[::-1]]
    return data_dc


def build_tagger_response(tags: list,
                          th_rating: float,
                          th_character: float,
                          th_general: float,
                          warning: list = [],
                          **kwargs) -> TaggerResponseModel:
    st = time.time()

    tgs_rating = tag_parse(tags[:4], th_rating, 0)
    if len(tgs_rating) == 0:
        tgs_rating = [tag_parse(tags[:4], 0, 0)[0]]
    tgs_general = tag_parse(tags[4:6951], th_general, 4)
    tgs_character = tag_parse(tags[6951:], th_character, 6951)

    dc = {
        'status': {
            'code': 'OK' if len(warning) == 0 else 'WARNING',
            'msg': warning
        },
        'post_process_time': time.time() - st,
        'results': {
            'count': len(tgs_rating) + len(tgs_general) + len(tgs_character),
            'data': {
                'rating': tgs_rating,
                'general': tgs_general,
                'character': tgs_character,
            }
        },
        **kwargs
    }

    return TaggerResponseModel(**dc)
