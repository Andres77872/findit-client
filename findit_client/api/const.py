EMBEDDING_SEARCH_API_PATH = 'pic2encoder'
EMBEDDING_GET_VECTOR_API_PATH = 'get_vector'

EMBEDDING_GET_VECTOR_CLIP_TEXT_API_PATH = 'pic2encoder_clip/text'
EMBEDDING_GET_VECTOR_CLIP_IMAGE_API_PATH = 'pic2encoder_clip/image'

SEARCH_BY_VECTOR_API_PATH = 'search_vector'
SEARCH_BY_ID_API_PATH = 'search_id'
SEARCH_SCROLL_API_PATH = 'search_scroll'
RANDOM_GENERATOR_API_PATH = 'https://img.arz.ai/random'

TAGGER_BY_FILE_API_PATH = 'anime_pic2tag/file'
TAGGER_BY_VECTOR_API_PATH = 'anime_pic2tag/vector'

URL_IMAGE_PROVIDER = [
    'https://img.arz.ai',
    # 'https://img-s4.arz.ai',
    # 'https://img-s5.arz.ai',
    # 'https://img-s6.arz.ai',
    # 'https://img-s7.arz.ai',
    # 'https://img-s8.arz.ai',
    # 'https://img-s9.arz.ai',
]

BOORUS_NAMES_STR = [
    'danbooru',
    'gelbooru',
    'zerochan',
    'anime-pictures',
    'yande.re',
    'e-shuushuu',
    'safebooru',
    'konachan',
    'tbib'
]

BOORU_TO_ID = {
    'danbooru': 1,
    'gelbooru': 2,
    'zerochan': 3,
    'anime-pictures': 4,
    'yande.re': 5,
    'e-shuushuu': 6,
    'safebooru': 7,
    'konachan': 8,
    'tbib': 9,
}

ID_TO_BOORU = {
    1: 'danbooru',
    2: 'gelbooru',
    3: 'zerochan',
    4: 'anime-pictures',
    5: 'yande.re',
    6: 'e-shuushuu',
    7: 'safebooru',
    8: 'konachan',
    9: 'tbib'
}

BOORU_SOURCE_URL = [
    'https://danbooru.donmai.us/posts/{0}',
    'https://gelbooru.com/index.php?page=post&s=view&id={0}',
    'https://www.zerochan.net/{0}',
    'https://anime-pictures.net/posts/{0}',
    'https://yande.re/post/show/{0}',
    'https://e-shuushuu.net/image/{0}',
    'https://safebooru.org/index.php?page=post&s=view&id={0}',
    'https://konachan.com/post/show/{0}',
    'https://tbib.org/index.php?page=post&s=view&id={0}',
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
    'check_sum': 18,
    'params_keys': [28, 2],  # file_id, image_size
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
                    ],
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
