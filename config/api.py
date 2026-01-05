import os
import configparser
from typing import Type, List, Optional
from sqlalchemy import select

from config.configs import *
from config.base import create_tables, create_session, Base, is_initialize


INDENTATION_STRIDE = 2
indentation_size = 0

CONFIG_MAP = {
    'model': ModelConfig,
    'mla': MLAConfig,
    'mha': MHAConfig,
    'rope': RopeConfig,
    'moe': MoEConfig,
    'mlp': MLPConfig,
    'norm': RMSNormConfig,
}
ATTRS_MAP = {
    'mla': 'mla_config',
    'mha': 'mha_config',
    'rope': 'rope_config',
    'moe': 'moe_config',
    'mlp': 'mlp_config',
    'norm': 'rmsnorm_config',
}
REVERSE_ATTRS_MAP = dict(zip(ATTRS_MAP.values(), ATTRS_MAP.keys()))
INIT_MODEL_FILE = [
    'deepseek-v3-671B.ini',
    'llama-3-8B.ini',
]


def comments_detail(config_type: Type[Base]) -> str:
    comments = [f"{config_type.__name__}:"]
    global indentation_size
    indentation_size += INDENTATION_STRIDE
    for col in config_type.__table__.columns:
        if col.name == 'id':
            continue
        comments.append(f"{' ' * indentation_size}{col.name}: {col.comment}")
    indentation_size -= INDENTATION_STRIDE
    return '\n'.join(comments)


def config_detail(config: Base) -> str:
    global indentation_size
    attrs = [f"{' ' * indentation_size}{config.__class__.__name__}:"]
    sub_attrs = []
    indentation_size += INDENTATION_STRIDE
    for k, v in config.__dict__.items():
        if k in ['_sa_instance_state', 'id', 'user']:
            continue
        if k == 'model_config':
            attrs.append(f"{' ' * indentation_size}{k}={', '.join(map(lambda x: x.name, v))}")
        elif k.endswith('_config_name') and v is not None:
            component_name = REVERSE_ATTRS_MAP.get("_".join(k.split('_')[:-1]))
            config_type = CONFIG_MAP.get(component_name)
            assert config_type is not None, f"`{component_name}` is not a valid config name"
            obj = query(v, config_type)
            assert obj is not None
            sub_attrs.append(config_detail(obj))
        elif isinstance(v, Base):
            sub_attrs.append(config_detail(v))
        else:
            attrs.append(f"{' ' * indentation_size}{k}={v}")
    indentation_size -= INDENTATION_STRIDE
    attrs.extend(sub_attrs)
    return '\n'.join(attrs)


def _get_init_model_files():
    cur_dir = os.path.dirname(__file__)
    templates_dir = os.path.join(cur_dir, 'templates')
    return [os.path.join(templates_dir, init_model_file) for init_model_file in INIT_MODEL_FILE]


def _parse_file(file: str, component_name: Optional[str]=None):
    config = configparser.ConfigParser()
    config.read(file, encoding='utf-8')
    components = {}
    for section in config.sections():
        config_obj = CONFIG_MAP.get(component_name or section.lower())
        assert config_obj is not None
        attrs = {k: getattr(config_obj, k).type.python_type(v) for k, v in config.items(section)}
        components[section.lower()] = config_obj(**attrs)
    return components


def _construct_config(components):
    if 'model' in components.keys():
        for component_name in [name for name in components.keys() if name != 'model']:
            component = components.pop(component_name)
            attr = ATTRS_MAP.get(component_name)
            assert attr is not None
            setattr(components['model'], attr, component)
    return list(components.values())


def initialize():
    if not is_initialize():
        create_tables()

        init_model_files = _get_init_model_files()
        cfgs = []
        for init_model_file in init_model_files:
            components = _parse_file(init_model_file)
            cfgs += _construct_config(components)
        add_all(cfgs)


def add_all(configs: List[Base]):
    assert all(map(lambda cfg: isinstance(cfg, Base), configs))
    with create_session() as session:
        session.add_all(configs)
        session.commit()


def query(name: str, config_type: Type[Base]) -> Base:
    assert issubclass(config_type, Base)
    with create_session() as session:
        stmt = select(config_type).where(config_type.name == name)
        obj = session.scalar(stmt)
    return obj


def query_all(config_type: Type[Base]) -> List[Base]:
    assert issubclass(config_type, Base)
    with create_session() as session:
        stmt = select(config_type)
        objs = session.scalars(stmt).all()
    return objs


def update(name: str, config_type: Type[Base], **new_attrs):
    obj = query(name, config_type)
    with create_session() as session:
        for k, v in new_attrs.items():
            assert k in new_attrs.__dict__.keys()
            setattr(obj, k, v)
        session.commit()


def delete(name: str, config_type: Type[Base]):
    obj = query(name, config_type)
    with create_session() as session:
        session.delete(obj)
        session.commit()


def show(name: str, component_name: str, attr_name: Optional[str]=None, is_detail: bool=False):
    config_type = CONFIG_MAP.get(component_name)
    assert config_type is not None, f"`{component_name}` is not a valid config name"
    obj = query(name, config_type)
    assert obj is not None, f"no config found named `{name}` from component `{component_name}`"
    if attr_name is not None:
        assert hasattr(obj, attr_name), f"object {obj.__class__} has no attribute name: {attr_name}"
        print(getattr(obj, attr_name))
    elif is_detail:
        print(config_detail(obj))
    else:
        print(obj)


def show_list(component_name: str):
    config_type = CONFIG_MAP.get(component_name)
    assert config_type is not None, f"`{component_name}` is not a valid config name"
    objs = query_all(config_type)
    for obj in objs:
        print(obj)


def show_attributes(component_name: str):
    config_type = CONFIG_MAP.get(component_name)
    assert config_type is not None, f"`{component_name}` is not a valid config name"
    attributes = comments_detail(config_type)
    print(attributes)


def all_tables():
    return CONFIG_MAP.keys()


def config_from_file(file: str, module: str):
    component_name = module if module != 'model' else None
    components = _parse_file(file, component_name)
    cfg = _construct_config(components)
    return cfg


def config_from_base(module: str, base: str, update: str):
    ...


# config_from_file('templates/moe.ini', '')
# show_list('model')
# show('Llama-3-8B', 'model', is_detail=True)
# show_attributes('model')
