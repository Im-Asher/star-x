from enum import Enum

class ResponseCode(Enum):
    Success = 1000
    
    Exception = 4000

    Fail = 9000