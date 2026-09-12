class IssueSenseError(Exception):
    """Base class for all IssueSense exceptions."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class AuthorizationError(IssueSenseError):
    """Raised when a request fails authorization checks."""
    pass

class EvidenceNotFoundError(IssueSenseError):
    """Raised when no grounding evidence is found for a claim."""
    pass

class ConnectorError(IssueSenseError):
    """Raised when an external source adapter fails."""
    pass

class GuardrailError(IssueSenseError):
    """Raised when input or output safety checks are violated."""
    pass
