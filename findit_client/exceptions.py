from findit_client.api.const import BOORUS_NAMES_STR


class ImageNotLoadedException(Exception):
    def __init__(self, mode, origin, message="Image was not loaded"):
        self.message = message + ', mode: ' + mode + ', origin: ' + origin
        super().__init__(self.message)


class EmbeddingException(Exception):
    def __init__(self, origin, message="Embedding error"):
        self.message = message + ', origin: ' + origin
        super().__init__(self.message)


class RemoteRawSearchException(Exception):
    def __init__(self, origin, message="Error in remote search engine"):
        self.message = message + ', origin: ' + origin
        super().__init__(self.message)


class ImageNotFetchedException(Exception):
    def __init__(self, origin, message="Error to load the URL img"):
        self.message = message + ', origin: ' + origin
        super().__init__(self.message)


class ImageSizeTooBigException(Exception):
    def __init__(self, size, limit, origin, message="Error the remote image size is too big"):
        self.message = message + ', origin: ' + origin + ', size: ' + size + ', limit: ' + limit
        super().__init__(self.message)


class ImageRemoteNotAsImageContentTypeException(Exception):
    def __init__(self, origin, message="Error, remote image not recognized as an image content type"):
        self.message = message + ', origin: ' + origin
        super().__init__(self.message)


class ImageRemoteNoContentLengthFoundException(Exception):
    def __init__(self, origin, message="Error, remote image not have Content-Length"):
        self.message = message + ', origin: ' + origin
        super().__init__(self.message)


class SearchBooruNotFound(Exception):
    def __init__(self, booru, message="Error, booru not found"):
        self.message = message + ', booru: ' + booru + ', avaible boorus: ' + ', '.join(BOORUS_NAMES_STR)
        super().__init__(self.message)
