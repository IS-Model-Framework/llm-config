from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column

from llm_config.util import get_current_user


class Base(DeclarativeBase, MappedAsDataclass):
  __abstract__ = True

  id: Mapped[int] = mapped_column(primary_key=True, autoincrement="auto", init=False)
  name: Mapped[str] = mapped_column(String(30), unique=True)
  user: Mapped[str] = mapped_column(String(30), default=get_current_user, init=False)

  def __repr__(self):
    attrs = []
    for k, v in self.__dict__.items():
      if k in ["_sa_instance_state", "model_config", "id", "user"]:
        continue
      if isinstance(v, Base):
        attrs.append(f"{k}={v.name}")
      else:
        attrs.append(f"{k}={v}")
    return f"{self.__class__.__name__}[{', '.join(attrs)}]"
