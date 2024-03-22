class CustomError(Exception):
    """Base class for custom exceptions."""
    pass

class InvalidQuestionError(CustomError):
    """Exception raised for custom errors."""

    def __init__(self, message="An error occurred while validating Question."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.__class__.__name__}: {self.message}'


class InvalidTripleError(CustomError):
    """Exception raised for custom errors."""

    def __init__(self, message="An error occurred while validating Triple."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.__class__.__name__}: {self.message}'


class JsonParsingError(CustomError):
    """Exception raised for custom errors."""

    def __init__(self, message="An error occurred while parsing generated response."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.__class__.__name__}: {self.message}'


class ContextLengthError(CustomError):
    """Exception raised for custom errors."""

    def __init__(self, message="Invalid context length error."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.__class__.__name__}: {self.message}'