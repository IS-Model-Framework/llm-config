import os
from sqlalchemy import create_engine, String
from sqlalchemy.orm import DeclarativeBase, Session, Mapped, mapped_column

from config.util import get_current_user


ENGINE = None
_DEFAULT_DB_NAME = "configs.sqlite"
SQL_PATH = os.path.join(os.path.dirname(__file__), _DEFAULT_DB_NAME)


def set_db_name(db_name: str):
    """Set the database file name. Call this before any database operations."""
    global SQL_PATH, ENGINE
    SQL_PATH = os.path.join(os.path.dirname(__file__), db_name)
    # Reset engine so it uses the new path
    ENGINE = None


def get_db_name() -> str:
    """Get the current database file name."""
    return os.path.basename(SQL_PATH)


class Base(DeclarativeBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True,  autoincrement="auto")
    name: Mapped[str] = mapped_column(String(30), unique=True)
    user: Mapped[str] = mapped_column(String(30), default=get_current_user)

    def __repr__(self):
        attrs = []
        for k, v in self.__dict__.items():
            if k in ['_sa_instance_state', 'model_config', 'id', 'user']:
                continue
            if isinstance(v, Base):
                attrs.append(f"{k}={v.name}")
            else:
                attrs.append(f"{k}={v}")
        return f"{self.__class__.__name__}[{', '.join(attrs)}]"


def get_engine(debug=False):
    global ENGINE
    if ENGINE is None:
        ENGINE = create_engine(f"sqlite:///{SQL_PATH}", echo=debug)
    return ENGINE


def create_tables():
    engine = get_engine()
    Base.metadata.create_all(engine)


def create_session():
    engine = get_engine()
    return Session(engine)


def is_initialize():
    return os.path.exists(SQL_PATH)
