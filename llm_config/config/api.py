import os
import re
import configparser
import ast
from typing import Type, List, Optional, Dict, Any, Union
from sqlalchemy import select

from llm_config.config.configs import *
from llm_config.config.base import create_tables, create_session, Base, is_initialize


INDENTATION_STRIDE = 2
indentation_size = 0

CONFIG_MAP = {
    'mesh': MeshConfig,
    'model': ModelConfig,
    'mla': MLAConfig,
    'mha': MHAConfig,
    'rope': RopeConfig,
    'embed': EmbeddingConfig,
    'moe': MoEConfig,
    'mlp': MLPConfig,
    'norm': RMSNormConfig,

}
ATTRS_MAP = {
    'mesh': 'mesh_config',
    'mla': 'mla_config',
    'mha': 'mha_config',
    'rope': 'rope_config',
    'embed': 'embed_config',
    'moe': 'moe_config',
    'mlp': 'mlp_config',
    'norm': 'rmsnorm_config',
}
REVERSE_ATTRS_MAP = dict(zip(ATTRS_MAP.values(), ATTRS_MAP.keys()))
ATTRS_NAME_MAP = {
    'mesh': 'mesh_config_name',
    'mla': 'mla_config_name',
    'mha': 'mha_config_name',
    'rope': 'rope_config_name',
    'embed': 'embed_config_name',
    'moe': 'moe_config_name',
    'mlp': 'mlp_config_name',
    'norm': 'rmsnorm_config_name',
}
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
    from sqlalchemy import Enum as SQLEnum
    
    config = configparser.ConfigParser()
    config.read(file, encoding='utf-8')
    components = {}
    for section in config.sections():
        # If component_name is provided, use it for all sections
        # Otherwise, try to infer from section name (for model configs)
        if component_name:
            config_obj = CONFIG_MAP.get(component_name)
        else:
            # For model configs, section names like [Model], [MHA], [MLP] etc.
            config_obj = CONFIG_MAP.get(section.lower())
        assert config_obj is not None, f"Unknown config type: {component_name or section.lower()}. Available types: {list(CONFIG_MAP.keys())}"
        attrs = {}
        for k, v in config.items(section):
            attr_column = getattr(config_obj, k)
            attr_type = attr_column.type
            # Handle enum types
            if isinstance(attr_type, SQLEnum):
                enum_class = attr_type.enum_class
                # Find enum by value
                enum_value = None
                for enum_item in enum_class:
                    if enum_item.value.lower() == v.lower():
                        enum_value = enum_item
                        break
                if enum_value is None:
                    raise ValueError(f"Invalid enum value '{v}' for {k}. Valid values: {[e.value for e in enum_class]}")
                attrs[k] = enum_value
            else:
                try:
                    v = attr_type.python_type(v)
                except:
                    try:
                        v = [ast.literal_eval(item) for item in v.strip("[]").split(",")]
                    except:
                        v = [item.strip() for item in v.strip("[]").split(",")]
                attrs[k] = v
        components[section.lower()] = config_obj(**attrs)
    return components


def _construct_config(components):
    if 'model' in components.keys():
        for component_name in [name for name in components.keys() if name != 'model']:
            component = components.pop(component_name)
            attr = ATTRS_MAP.get(component_name)
            assert attr is not None
            setattr(components['model'], attr, component)
        if not components['model'].embed_config:
            components['model'].embed_config = EmbeddingConfig(name=components['model'].name)
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


def query(name: str, config_type: Type[Base], eager_load: bool = False) -> Base:
    assert issubclass(config_type, Base)
    with create_session() as session:
        stmt = select(config_type).where(config_type.name == name)
        
        # Eager load relationships for ModelConfig
        if eager_load and config_type == ModelConfig:
            from sqlalchemy.orm import joinedload
            stmt = stmt.options(
                joinedload(ModelConfig.mesh_config),
                joinedload(ModelConfig.mla_config),
                joinedload(ModelConfig.mha_config),
                joinedload(ModelConfig.mlp_config),
                joinedload(ModelConfig.moe_config),
                joinedload(ModelConfig.rmsnorm_config),
                joinedload(ModelConfig.rope_config),
                joinedload(ModelConfig.embed_config),
            )
        
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
    assert obj is not None, f"no config found named `{name}` from component `{config_type.__name__}`"
    with create_session() as session:
        for k, v in new_attrs.items():
            assert hasattr(obj, k), f"object {obj.__class__} has no attribute: {k}"
            setattr(obj, k, v)
        session.commit()


def delete(name: str, config_type: Type[Base]):
    obj = query(name, config_type)
    assert obj is not None, f"no config found named `{name}` from component `{config_type.__name__}`"
    with create_session() as session:
        session.delete(obj)
        session.commit()


def show(name: str, component_name: str, attr_name: Optional[str]=None, is_detail: bool=False):
    config_type = CONFIG_MAP.get(component_name)
    assert config_type is not None, f"`{component_name}` is not a valid config name"
    
    # Enable eager loading for model component to avoid DetachedInstanceError
    eager_load = (component_name == 'model')
    obj = query(name, config_type, eager_load=eager_load)

    assert obj is not None, f"no config found named `{name}` from component `{component_name}`"
    if attr_name is not None:
        # Check if it's a nested attribute
        if '.' in attr_name:
            value = _get_nested_attribute(obj, attr_name)
            print(value)
        else:
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


def _parse_update_string(update_str: str) -> Dict[str, Any]:
    """Parse update string like 'name=new-name,kv_lora_rank=1024' into dict"""
    updates = {}
    if not update_str:
        return updates

    pairs = re.split(r',(?=[\w\d_]+=)', update_str)

    def recursive_parse(value: str, is_list_context: bool = False) -> Any:
        value = value.strip()

        if (value.startswith('[') and value.endswith(']')) or \
           (value.startswith('{') and value.endswith('}')) or \
           (value.startswith('(') and value.endswith(')')):
            inner = value[1:-1].strip()
            if not inner:
                return []

            elements = []
            bracket_level = 0
            current_token = []

            for char in inner:
                if char in '[{':
                    bracket_level += 1
                elif char in ']}':
                    bracket_level -= 1

                if char == ',' and bracket_level == 0:
                    elements.append("".join(current_token).strip())
                    current_token = []
                else:
                    current_token.append(char)
            elements.append("".join(current_token).strip())

            return [recursive_parse(e, is_list_context=True) for e in elements]

        if not is_list_context and ',' in value:
            items = value.split(',')
            return [recursive_parse(i.strip(), is_list_context=True) for i in items]

        if not value:
            return None if is_list_context else ""

        return value

    updates = {}
    for pair in pairs:
        if '=' not in pair:
            continue
        key, val_str = pair.split('=', 1)
        updates[key.strip()] = recursive_parse(val_str.strip())

    return updates

def _get_nested_attribute(obj: Base, attr_path: str):
    """Get nested attribute value like 'mlp.dim' from object"""
    if '.' not in attr_path:
        # Direct attribute
        return getattr(obj, attr_path)
    
    parts = attr_path.split('.')
    if len(parts) != 2:
        raise ValueError(f"Invalid nested attribute format: {attr_path}. Expected format: 'component.attribute'")
    
    component_name, attr_name = parts
    attr_name_full = ATTRS_MAP.get(component_name)
    if not attr_name_full:
        raise ValueError(f"Unknown component name: {component_name}. Available components: {list(ATTRS_MAP.keys())}")
    
    component_obj = getattr(obj, attr_name_full, None)
    if component_obj is None:
        raise ValueError(f"Component {component_name} not found in {obj.name}")
    
    if not hasattr(component_obj, attr_name):
        raise ValueError(f"Attribute {attr_name} not found in {component_name} component")
    
    return getattr(component_obj, attr_name)


def _set_attr(obj: Base, attr_name: str, value: Any):
    if getattr(obj, attr_name, None) != value:
        setattr(obj, attr_name, value)


def _set_nested_attribute(obj: Base, attr_path: str, value: Any):
    """Set nested attribute value like 'mlp.dim' on object"""
    if '.' not in attr_path:
        _set_attr(obj, attr_path, value)
        return
    
    parts = attr_path.split('.')
    if len(parts) != 2:
        raise ValueError(f"Invalid nested attribute format: {attr_path}. Expected format: 'component.attribute'")
    
    component_name, attr_name = parts
    attr_name_full = ATTRS_MAP.get(component_name)
    if not attr_name_full:
        raise ValueError(f"Unknown component name: {component_name}. Available components: {list(ATTRS_MAP.keys())}")
    
    component_obj = getattr(obj, attr_name_full, None)
    if component_obj is None and attr_name == "name":
        component_obj = query(value, CONFIG_MAP[component_name])
        assert component_obj is not None
        _set_attr(obj, ATTRS_NAME_MAP[component_name], value)
        _set_attr(obj, attr_name_full, component_obj)
        return
    if component_obj is None:
        raise ValueError(f"Component {component_name} not found in {obj.name}")
    
    # Get attribute type for conversion (only if value is string)
    if isinstance(value, str):
        attr_column = getattr(component_obj.__class__, attr_name)
        converted_value = _convert_value(value, attr_column.type)
    else:
        converted_value = value
    
    # Set the value
    _set_attr(component_obj, attr_name, converted_value)


def _convert_value(value: str, attr_type):
    """Convert string value to appropriate type, handling enums"""
    from sqlalchemy import Enum as SQLEnum
    
    python_type = attr_type.python_type
    
    # Handle Enum types
    if isinstance(attr_type, SQLEnum):
        enum_class = attr_type.enum_class
        # Try to find the enum value by string
        for enum_item in enum_class:
            if enum_item.value.lower() == value.lower():
                return enum_item
        raise ValueError(f"Invalid enum value '{value}' for {enum_class.__name__}. Valid values: {[e.value for e in enum_class]}")
    elif python_type == bool:
        return value.lower() in ['true', '1', 'yes']
    else:
        try:
            v = python_type(value)
        except:
            try:
                v = [ast.literal_eval(item) for item in value.strip("[]").split(",")]
            except:
                v = [item.strip() for item in value.strip("[]").split(",")]
        return v


def _apply_updates(obj: Base, updates: Dict[str, Any], config_type: Type[Base]):
    """Apply updates to an object, handling nested attributes like 'mlp.dim' or 'moe.routed_experts_dim'"""
    for key, value in updates.items():
        if '.' in key:
            # Handle nested attributes using helper function
            _set_nested_attribute(obj, key, value)
        else:
            # Direct attribute
            if key == 'name':
                setattr(obj, key, value)
            else:
                attr_column = getattr(obj.__class__, key)
                converted_value = _convert_value(value, attr_column.type) if isinstance(value, str) else value
                setattr(obj, key, converted_value)


def _find_dependent_models(component_name: str, component_type: Type[Base]) -> List[str]:
    """Find all models that depend on a component"""
    dependent_models = []
    with create_session() as session:
        all_models = session.scalars(select(ModelConfig)).all()
        for model in all_models:
            for attr_name, component_attr in ATTRS_MAP.items():
                model_component = getattr(model, component_attr, None)
                if model_component and model_component.name == component_name:
                    dependent_models.append(model.name)
                    break
    return dependent_models


def _confirm_action(message: str, warning_details: List[str]) -> bool:
    """Centralized function to ask for user confirmation, respecting test mode."""
    import os
    if os.environ.get('HLO_CONFIG_TEST_MODE', 'false').lower() in ['1', 'true']:
        # In test mode, automatically confirm
        return True

    print(message)
    for detail in warning_details:
        print(f"  - {detail}")
    response = input("Do you want to continue? (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def _check_dependencies_and_confirm(component_name: str, component_type: Type[Base], operation: str, exclude_model_name: Optional[str] = None) -> bool:
    """Check dependencies and ask for user confirmation if there are dependent models."""
    dependent_models = _find_dependent_models(component_name, component_type)
    
    if exclude_model_name:
        other_dependent_models = [m for m in dependent_models if m != exclude_model_name]
    else:
        other_dependent_models = dependent_models

    if other_dependent_models:
        message = f"Warning: {operation} '{component_name}' will affect the following models:"
        return _confirm_action(message, other_dependent_models)
    return True


def _clone_obj(obj: Base):
    """Creates a new instance of a SQLAlchemy object, copying attributes but not relationships or state."""
    if obj is None:
        return None
    cls = obj.__class__
    data = {c.name: getattr(obj, c.name) for c in cls.__table__.columns if c.name != 'id'}
    new_obj = cls(**data)
    return new_obj


def _generate_unique_clone_name(base_name: str, component_type: Type[Base]) -> str:
    """Generates a unique name for a cloned component."""
    clone_name = f"{base_name}-clone"
    if query(clone_name, component_type) is None:
        return clone_name
    
    i = 1
    while True:
        next_name = f"{clone_name}-{i}"
        if query(next_name, component_type) is None:
            return next_name
        i += 1


def config_from_base(module: str, base: str, update_str: str, base_layer: Optional[str] = None) -> List[Base]:
    """Create config from base config with updates"""
    config_type = CONFIG_MAP.get(module)
    assert config_type is not None, f"`{module}` is not a valid module name"
    
    with create_session() as session:
        stmt = select(config_type).where(config_type.name == base)
        if module == 'model':
            from sqlalchemy.orm import joinedload
            stmt = stmt.options(
                joinedload(ModelConfig.mesh_config),
                joinedload(ModelConfig.mla_config),
                joinedload(ModelConfig.mha_config),
                joinedload(ModelConfig.mlp_config),
                joinedload(ModelConfig.moe_config),
                joinedload(ModelConfig.rmsnorm_config),
                joinedload(ModelConfig.rope_config),
                joinedload(ModelConfig.embed_config),
            )
        base_obj = session.scalar(stmt)
        assert base_obj is not None, f"no config found named `{base}` from module `{module}`"

    new_model_config = _clone_obj(base_obj)
    
    raw_updates = _parse_update_string(update_str)
    model_direct_updates = {k: v for k, v in raw_updates.items() if '.' not in k}
    component_nested_updates = {}
    for k, v in raw_updates.items():
        if '.' in k:
            comp, attr = k.split('.', 1)
            component_nested_updates.setdefault(comp, {})[attr] = v

    if 'name' not in model_direct_updates:
        raise ValueError("'name' must be provided in --attributes parameter when creating from base")
    new_model_name = model_direct_updates['name']
    if query(new_model_name, config_type) is not None:
        raise ValueError(f"Config with name '{new_model_name}' already exists")
    
    _apply_updates(new_model_config, model_direct_updates, config_type)
    
    if module == 'model' and base_layer:
        _parse_base_layer(new_model_config, base_layer)
    
    all_configs_to_merge = []
    if module == 'model':
        for comp_name, attr_full in ATTRS_MAP.items():
            updates = component_nested_updates.get(comp_name, {})
            original_component = getattr(base_obj, attr_full)
            has_attr_changes = any(k != 'name' for k in updates)

            # Scenario 2 & 3: Attribute changes exist, must clone.
            if has_attr_changes:
                atom_name = updates.get('name', original_component.name if original_component else None)
                if not atom_name:
                    raise ValueError(f"Cannot determine base component for cloning '{comp_name}'")
                
                atom_component = query(atom_name, CONFIG_MAP[comp_name])
                if not atom_component:
                    raise ValueError(f"Component '{atom_name}' to clone from does not exist.")

                new_clone_name = _generate_unique_clone_name(atom_component.name, CONFIG_MAP[comp_name])
                cloned_component = _clone_obj(atom_component)
                cloned_component.name = new_clone_name
                
                updates_to_apply = {k: v for k, v in updates.items() if k != 'name'}
                _apply_updates(cloned_component, updates_to_apply, CONFIG_MAP[comp_name])
                
                setattr(new_model_config, attr_full, cloned_component)

            # Scenario 1: Only name is specified, no attribute changes.
            # TODO: add test case.
            elif 'name' in updates:
                target_comp_name = updates['name']
                old_comp_name = getattr(new_model_config, f"{ATTRS_MAP[comp_name]}_name")
                target_component = query(old_comp_name, CONFIG_MAP[comp_name])
                target_component = _clone_obj(target_component)
                if not target_component:
                    raise ValueError(f"Specified component '{comp_name}' with name '{target_comp_name}' not found.")
                target_component.name = target_comp_name
                setattr(new_model_config, attr_full, target_component)

            # Scenario 4: No updates for this component.
            else:
                if original_component:
                    setattr(new_model_config, attr_full, original_component)

    all_configs_to_merge.append(new_model_config)
    
    # Remove duplicates before returning
    unique_configs = {cfg.name: cfg for cfg in all_configs_to_merge}.values()
    return list(unique_configs)


def _parse_base_layer(model_obj: ModelConfig, base_layer_str: str):
    """Parse --base-layer string like 'mha.name=Llama-3-8B,rope.name=Llama-3-8B'"""
    for item in base_layer_str.split(','):
        if '=' not in item:
            continue
        key, component_value = item.split('=', 1)
        key = key.strip()
        component_value = component_value.strip()
        
        if key.endswith('.name'):
            component_name = key[:-5]  # Remove '.name'
            attr_name = ATTRS_MAP.get(component_name)
            if attr_name:
                component_type = CONFIG_MAP.get(component_name)
                component_obj = query(component_value, component_type)
                assert component_obj is not None, f"Component {component_name} with name {component_value} not found"
                setattr(model_obj, attr_name, component_obj)
            else:
                raise ValueError(f"Invalid component name: {component_name}")
        else:
            raise ValueError(f"Invalid base-layer format: {key}. Expected format: 'component.name=value'")
