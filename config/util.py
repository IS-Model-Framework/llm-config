import getpass
from typing import Optional


def get_current_user(user: Optional[str]=None):
    if user:
        return user
    return getpass.getuser()
