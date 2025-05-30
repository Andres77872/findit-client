from findit_client.base import FindItBase

__version__ = '0.4.9'


class FindItClient(FindItBase):

    def __init__(self, url_api_embedding: str = 'https://nn.arz.ai/',
                 url_image_backend: str = 'https://img.arz.ai/',
                 url_api_back_search: str = 'https://search.arz.ai/',
                 pixiv_credentials: dict = {'username': '', 'pasword': ''},
                 **kwargs):
        super().__init__(url_api_embedding=url_api_embedding,
                         url_api_back_search=url_api_back_search,
                         url_image_backend=url_image_backend,
                         __version__=__version__,
                         pixiv_credentials=pixiv_credentials,
                         **kwargs)

    def download_embedding_search_model(self):
        pass

    def download_tagger_model(self):
        pass

    def download_tags_dictionary(self):
        pass
