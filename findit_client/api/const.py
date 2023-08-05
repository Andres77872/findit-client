EMBEDDING_SEARCH_API_PATH = 'pic2encoder'
EMBEDDING_GET_VECTOR_API_PATH = 'get_vector'

EMBEDDING_GET_VECTOR_TEXT_API_PATH = 'pic2encoder_clip_text'

SEARCH_BY_VECTOR_API_PATH = 'search_vector'
SEARCH_BY_ID_API_PATH = 'search_id'
SEARCH_SCROLL_API_PATH = 'search_scroll'
RANDOM_GENERATOR_API_PATH = 'https://img.arz.ai/random'

TAGGER_BY_FILE_API_PATH = 'anime_pic2tag/file'
TAGGER_BY_VECTOR_API_PATH = 'anime_pic2tag/vector'

URL_IMAGE_PROVIDER = [
    'https://img-s4.arz.ai',
    'https://img-s5.arz.ai',
    'https://img-s6.arz.ai',
    'https://img-s7.arz.ai',
    'https://img-s8.arz.ai',
    'https://img-s9.arz.ai',
]

BOORUS_NAMES_STR = [
    'danbooru',
    'gelbooru',
    'zerochan',
    'anime-pictures',
    'yande.re',
    'e-shuushuu',
    'safebooru']

BOORU_TO_ID = {
    'danbooru': 0,
    'sem_danbooru': 0,
    'gelbooru': 1,
    'sem_gelbooru': 1,
    'zerochan': 2,
    'sem_zerochan': 2,
    'anime-pictures': 3,
    'sem_anime-pictures': 3,
    'yande.re': 4,
    'sem_yande.re': 4,
    'e-shuushuu': 5,
    'sem_e-shuushuu': 5,
    'safebooru': 6,
    'sem_safebooru': 6,
}

ID_TO_BOORU = {
    0: 'danbooru',
    1: 'gelbooru',
    2: 'zerochan',
    3: 'anime-pictures',
    4: 'yande.re',
    5: 'e-shuushuu',
    6: 'safebooru'
}

BOORU_SOURCE_URL = [
    'https://danbooru.donmai.us/posts/{0}',
    'https://gelbooru.com/index.php?page=post&s=view&id={0}',
    'https://www.zerochan.net/{0}',
    'https://anime-pictures.net/posts/{0}',
    'https://yande.re/post/show/{0}',
    'https://e-shuushuu.net/image/{0}',
    'https://safebooru.org/index.php?page=post&s=view&id={0}'
]

X_query_arzypher_params = {
    'random_key': None,
    'check_sum': 64,
    'params_keys': [8, 24],  # booru_ID, image_ID
    'padding': None,
    'private_key': None
}
"""
{
    'random_key': None, \n
    'check_sum': 32, \n
    'params_keys': [8, 24],  # booru_ID, image_ID \n
    'padding': None, \n
    'private_key': None
}
"""

X_image_arzypher_params = {
    'random_key': None,
    'check_sum': 15,
    'params_keys': [8, 24, 1],  # booru_ID, image_ID
    'padding': None,
    'private_key': None
}
"""
{
    'random_key': None, \n
    'check_sum': 15, \n
    'params_keys': [8, 24, 1],  # booru_ID, image_ID \n
    'padding': None, \n
    'private_key': None
}
"""

X_scroll_arzypher_params = {
    'random_key': 32,
    'check_sum': 256,
    'params_keys': [32,  # ID
                    16,  # COUNT
                    8, 16,  # [DANBOORU, COUNT]
                    8, 16,  # [GELBOORU, COUNT]
                    8, 16,  # [ZEROCHAN, COUNT]
                    8, 16,  # [ANIME-PICTURES, COUNT]
                    8, 16,  # [YANDE.RE, COUNT]
                    8, 16,  # [E-SHUUSHUU, COUNT]
                    8, 16],  # [SAFEBOORU, COUNT]
    'padding': None,
    'private_key': None
}
"""
{
    'random_key': 32, \n
    'check_sum': 256, \n
    'params_keys': [32,  # ID 
                    16,  # COUNT \n
                    8, 16,  # [DANBOORU, COUNT] \n
                    8, 16,  # [GELBOORU, COUNT] \n
                    8, 16,  # [ZEROCHAN, COUNT] \n
                    8, 16,  # [ANIME-PICTURES, COUNT] \n
                    8, 16,  # [YANDE.RE, COUNT] \n
                    8, 16,  # [E-SHUUSHUU, COUNT] \n
                    8, 16],  # [SAFEBOORU, COUNT] \n
    'padding': None, \n
    'private_key': None
}
"""
