import numpy as np

from findit_client.api.const import BOORUS_NAMES_STR, BOORU_TO_ID
from findit_client.exceptions import SearchBooruNotFound, SearchRatingNotFound


def validate_params(func):
    def wrapper(*args, **kwargs):
        if 'limit' in kwargs:
            total = kwargs['limit']
            if not isinstance(total, int):
                total = int(total)
            if not 0 <= total <= 128:
                total = 32
                kwargs['limit'] = total
        if 'pool' in kwargs:
            if (pool := kwargs['pool']) and pool is not None:
                for p in pool:
                    if p not in BOORUS_NAMES_STR:
                        raise SearchBooruNotFound(booru=p)
            else:
                kwargs['pool'] = BOORUS_NAMES_STR

        if 'rating' in kwargs:
            if (rating := kwargs['rating']) and rating is not None:
                for r in rating:
                    if r not in {'g', 's', 'q', 'e'}:
                        raise SearchRatingNotFound(booru=r)
            else:
                kwargs['rating'] = ['g', 's', 'q', 'e']
        if 'booru_name' in kwargs:
            if kwargs['booru_name']:
                if kwargs['booru_name'] not in BOORUS_NAMES_STR:
                    raise SearchBooruNotFound(booru=kwargs['booru_name'])
                kwargs['booru_name'] = BOORU_TO_ID[kwargs['booru_name']]

        if 'pool_vector' in kwargs:
            if kwargs['pool_vector'] not in BOORUS_NAMES_STR:
                raise SearchBooruNotFound(booru=kwargs['pool_vector'])
            kwargs['pool_vector'] = BOORU_TO_ID[kwargs['pool_vector']]
        num_sum = func(*args, **kwargs)
        return num_sum

    return wrapper


def validate_load_image(func):
    def wrapper(image: str | bytes | list[str | bytes], *args, **kwargs):
        if isinstance(image, str | bytes):
            res = func(image, *args, **kwargs)
        else:
            res = [func(x, *args, **kwargs) for x in image]
            tm = sum([x[1] for x in res])
            np_image = np.concatenate(np.array([x[0] for x in res]), axis=0, dtype=res[0][0].dtype)
            res = np_image, tm
        return res

    return wrapper
