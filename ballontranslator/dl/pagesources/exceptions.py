#  BaseError structure stolen from dmMaze
class BaseError(Exception):
    """
    base error structure class
    """

    def __init__(self, val, message):
        """
        @param val: actual value
        @param message: message shown to the user
        """
        self.val = val
        self.message = message
        super().__init__()

    def __str__(self):
        return "{} --> {}".format(self.val, self.message)


class NotValidUrl(BaseError):
    """
    exception thrown if the user enters an invalid url
    """

    def __init__(self,
                 val,
                 message='text must be a valid url, it must also include https://'):
        super(NotValidUrl, self).__init__(val, message)


class ImagesNotFoundInRequest(BaseError):
    """
    exception thrown if the program fails to locate images on a website
    """

    def __init__(self, val,
                 message='the specified website is not currently supported, '
                         'you can always suggest implementation of any website on github'):
        super(ImagesNotFoundInRequest, self).__init__(val, message)

