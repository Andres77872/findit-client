from findit_client.base import FindItBase

__version__ = '0.2.1'


class FindItClient(FindItBase):

    def __init__(self, url_api_embedding: str = 'http://127.0.0.1:7999/',
                 url_api_back_search: str = 'https://search.arz.ai/',
                 private_key: str | None = None,
                 __ChatGPT_TOKEN__: str = None):
        super().__init__(url_api_embedding=url_api_embedding,
                         url_api_back_search=url_api_back_search,
                         private_key=private_key,
                         __version__=__version__,
                         __ChatGPT_TOKEN__=__ChatGPT_TOKEN__)

    def download_embedding_search_model(self):
        pass

    def download_tagger_model(self):
        pass

    def download_tags_dictionary(self):
        pass
