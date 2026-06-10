class AiwfError(Exception):
    """Base error for AIWF Studio."""


class ModelNotFoundError(AiwfError):
    pass


class GenerationCancelledError(AiwfError):
    pass


class ValidationError(AiwfError):
    pass