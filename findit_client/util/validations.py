from findit_client.api.const import BOORUS_NAMES_STR
from findit_client.exceptions import SearchBooruNotFound


def validate_params(func):
    def wrapper(*args, **kwargs):
        if 'limit' in kwargs:
            total = kwargs['limit']
            if not isinstance(total, int):
                total = int(total)
            if not 0 < total <= 128:
                total = 32
                kwargs['limit'] = total
        if 'pool' in kwargs:
            if (pool := kwargs['pool']) and pool is None:
                kwargs['pool'] = BOORUS_NAMES_STR
            else:
                for p in pool:
                    if p not in BOORUS_NAMES_STR:
                        raise SearchBooruNotFound(booru=p)
        if 'booru_name' in kwargs:
            if kwargs['booru_name'] not in BOORUS_NAMES_STR:
                raise SearchBooruNotFound(booru=kwargs['booru_name'])
        num_sum = func(*args, **kwargs)
        return num_sum

    return wrapper
