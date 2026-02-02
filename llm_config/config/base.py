import os
from sqlalchemy import create_engine, String
from sqlalchemy.orm import DeclarativeBase, Session, Mapped, mapped_column

from llm_config.config.util import get_current_user


ENGINE = None
_DEFAULT_DB_NAME = "configs.sqlite"
_SQL_PATH = os.environ.get('HLO_CONFIG_DB_PATH', os.path.dirname(__file__))
SQL_PATH = os.path.join(os.path.abspath(_SQL_PATH),
                        os.environ.get('HLO_CONFIG_DB_NAME', _DEFAULT_DB_NAME)
                        )

def get_db_path():
    return SQL_PATH


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
        ENGINE = create_engine(f"sqlite:///{get_db_path()}", echo=debug)
    return ENGINE


def create_tables():
    print(f"SQL PATH: {get_db_path()}")
    engine = get_engine()
    Base.metadata.create_all(engine)


def create_session():
    engine = get_engine()
    return Session(engine)


def is_initialize():
    return os.path.exists(get_db_path())
