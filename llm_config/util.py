import getpass


def get_current_user(user: str | None = None):
  if user:
    return user
  return getpass.getuser()
