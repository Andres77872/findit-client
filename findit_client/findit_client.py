from findit_client.base import FindItBase

__version__ = '0.2.1'


class FindItClient(FindItBase):
    def __init__(self, url_api_embedding: str, url_api_back_search: str):
        super().__init__(url_api_embedding=url_api_embedding,
                         url_api_back_search=url_api_back_search,
                         __version__=__version__)

    def download_embedding_search_model(self):
        pass

    def download_tagger_model(self):
        pass

    def download_tags_dictionary(self):
        pass
