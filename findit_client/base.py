from findit_client.api.method_search import FindItMethodsSearch
from findit_client.api.method_tagger import FindItMethodsTagger
from findit_client.api.method_util import FindItMethodsUtil


class FindItBase:
    def __init__(self,
                 __version__: str,
                 **kwargs):
        self.__version__ = __version__
        self.search = FindItMethodsSearch(__version__=__version__, **kwargs)
        self.tagger = FindItMethodsTagger(__version__=__version__, **kwargs)
        self.util = FindItMethodsUtil(__version__=__version__, **kwargs)
