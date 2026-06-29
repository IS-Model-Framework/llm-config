import ast
import configparser

from llm_config.base import Base
from llm_config.configs import (
  EmbeddingConfig,
  MeshConfig,
  MHAConfig,
  MLAConfig,
  MLPConfig,
  ModelConfig,
  MoEConfig,
  RMSNormConfig,
  RopeConfig,
  ParallelismConfig,
)

INDENTATION_STRIDE = 2
indentation_size = 0

CONFIG_MAP = {
  "mesh": MeshConfig,
  "model": ModelConfig,
  "mla": MLAConfig,
  "mha": MHAConfig,
  "rope": RopeConfig,
  "embed": EmbeddingConfig,
  "moe": MoEConfig,
  "mlp": MLPConfig,
  "norm": RMSNormConfig,
  "parallelism": ParallelismConfig
}
ATTRS_MAP = {
  "mesh": "mesh_config",
  "mla": "mla_config",
  "mha": "mha_config",
  "rope": "rope_config",
  "embed": "embed_config",
  "moe": "moe_config",
  "mlp": "mlp_config",
  "norm": "rmsnorm_config",
  "parallelism": "parallelism_config",
}
REVERSE_ATTRS_MAP = dict(zip(ATTRS_MAP.values(), ATTRS_MAP.keys(), strict=True))
ATTRS_NAME_MAP = {
  "mesh": "mesh_config_name",
  "mla": "mla_config_name",
  "mha": "mha_config_name",
  "rope": "rope_config_name",
  "embed": "embed_config_name",
  "moe": "moe_config_name",
  "mlp": "mlp_config_name",
  "norm": "rmsnorm_config_name",
  "parallelism": "parallelism_config_name",
}


def comments_detail(config_type: Base) -> str:
  comments = [f"{config_type.__name__}:"]
  global indentation_size
  indentation_size += INDENTATION_STRIDE
  for col in config_type.__table__.columns:
    if col.name == "id":
      continue
    comments.append(f"{' ' * indentation_size}{col.name}: {col.comment}")
  indentation_size -= INDENTATION_STRIDE
  return "\n".join(comments)


def parse_list_str(s):
  if s is None:
    return None

  s = s.strip()
  if s == "":
    return None

  i = 0
  n = len(s)

  def parse_value():
    nonlocal i
    while i < n and s[i].isspace():
      i += 1
    if i < n and s[i] == "[":
      return parse_array()
    start = i
    while i < n and s[i] not in ",]":
      i += 1
    token = s[start:i].strip()
    if token == "" or token.lower() in {"nil", "null", "none"}:
      return None
    try:
      return ast.literal_eval(token)
    except (ValueError, SyntaxError):
      return token

  def parse_array():
    nonlocal i
    result = []
    if s[i] != "[":
      raise ValueError(f"Expected '[' at position {i}")
    i += 1
    while i < n:
      while i < n and s[i].isspace():
        i += 1
      if i < n and s[i] == "]":
        i += 1
        return result

      value = parse_value()
      result.append(value)

      while i < n and s[i].isspace():
        i += 1
      if i < n and s[i] == ",":
        i += 1
        while i < n and s[i].isspace():
          i += 1
        if i < n and s[i] == "]":
          result.append(None)
          i += 1
          return result
      elif i < n and s[i] == "]":
        i += 1
        return result
      else:
        raise ValueError(f"Expected ',' or ']' at position {i}")
    raise ValueError("Unclosed '['")

  parsed = parse_value()
  while i < n and s[i].isspace():
    i += 1
  if i != n:
    raise ValueError(f"Unexpected content at position {i}")

  return parsed


def _parse_file(file: str, component_name: str | None = None):
  from sqlalchemy import Enum as SQLEnum

  config = configparser.ConfigParser()
  config.read(file, encoding="utf-8")
  components = {}
  for section in config.sections():
    # If component_name is provided, use it for all sections
    # Otherwise, try to infer from section name (for model configs)
    if component_name:
      config_obj = CONFIG_MAP.get(component_name)
    else:
      # For model configs, section names like [Model], [MHA], [MLP] etc.
      config_obj = CONFIG_MAP.get(section.lower())
    if config_obj is None:
      raise ValueError(
        f"Unknown config type: {component_name or section.lower()}. "
        f"Available types: {list(CONFIG_MAP.keys())}"
      )
    attrs = {}
    for k, v in config.items(section):
      attr_column = getattr(config_obj, k)
      attr_type = attr_column.type
      # Handle enum types
      if isinstance(attr_type, SQLEnum):
        enum_class = attr_type.enum_class
        # Find enum by value
        enum_value = None
        for enum_item in enum_class:  # type: ignore[union-attr]
          if enum_item.value.lower() == v.lower():
            enum_value = enum_item
            break
        if enum_value is None:
          raise ValueError(
            f"Invalid enum value '{v}' for {k}. Valid values: "
            f"{[e.value for e in enum_class]}"  # type: ignore[union-attr]
          )
        attrs[k] = enum_value
      else:
        try:
          if issubclass(attr_type.python_type, bool):
            v = str(v).lower() not in ["0", "false", "no"]  # type: ignore[assignment]
          else:
            v = attr_type.python_type(v)  # type: ignore[assignment]
        except (ValueError, TypeError):
          try:
            v = [ast.literal_eval(item) for item in v.strip("[]").split(",")]  # type: ignore[assignment]
          except (ValueError, SyntaxError, NotImplementedError):
            v = parse_list_str(v)
        attrs[k] = v
    components[section.lower()] = config_obj(**attrs)
  return components


def _construct_config(components):
  if "model" in components.keys():
    for component_name in [name for name in components.keys() if name != "model"]:
      component = components.pop(component_name)
      attr = ATTRS_MAP.get(component_name)
      assert attr is not None
      setattr(components["model"], attr, component)
    if not components["model"].embed_config:
      components["model"].embed_config = EmbeddingConfig(name=components["model"].name)
  return list(components.values())


def show_attributes(component_name: str):
  config_type = CONFIG_MAP.get(component_name)
  assert config_type is not None, f"`{component_name}` is not a valid config name"
  attributes = comments_detail(config_type)  # type: ignore[assignment,arg-type]
  print(attributes)


def all_tables():
  return CONFIG_MAP.keys()


def parse_file(file: str, module: str):
  component_name = module if module != "model" else None
  components = _parse_file(file, component_name)
  cfg = _construct_config(components)
  return cfg
