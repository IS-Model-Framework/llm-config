import ast
import configparser
from collections import OrderedDict
from typing import Any

from sqlalchemy import Enum as SQLEnum

from llm_config.base import Base
from llm_config.configs import (
  AttentionConfigBase,
  ComponentSlotSpec,
  DecoderConfig,
  DeepSeekModelConfig,
  EmbeddingConfig,
  FusionConfig,
  KDNConfig,
  MHAConfig,
  MLAConfig,
  MLPConfig,
  MeshConfig,
  ModelConfig,
  ModelConfigBase,
  MoEConfig,
  ProjectorConfig,
  RMSNormConfig,
  RopeConfig,
  T5TextEncoderConfig,
  TextEncoderConfig,
  TransformerLMConfig,
  VaeDecoderConfig,
  VaeEncoderConfig,
  VisionEncoderConfig,
  WanModelConfig,
)

INDENTATION_STRIDE = 2
indentation_size = 0

CONFIG_REGISTRY = {
  "mesh": MeshConfig,
  "model": ModelConfig,
  "transformer_lm": TransformerLMConfig,
  "deepseek": DeepSeekModelConfig,
  "wan": WanModelConfig,
  "mla": MLAConfig,
  "mha": MHAConfig,
  "kdn": KDNConfig,
  "rope": RopeConfig,
  "embed": EmbeddingConfig,
  "embedding": EmbeddingConfig,
  "moe": MoEConfig,
  "mlp": MLPConfig,
  "norm": RMSNormConfig,
  "rmsnorm": RMSNormConfig,
  "text_encoder": TextEncoderConfig,
  "t5": T5TextEncoderConfig,
  "t5_text_encoder": T5TextEncoderConfig,
  "vision_encoder": VisionEncoderConfig,
  "vae_encoder": VaeEncoderConfig,
  "vae_decoder": VaeDecoderConfig,
  "projector": ProjectorConfig,
  "fusion": FusionConfig,
  "decoder": DecoderConfig,
}

LEGACY_SECTION_SLOT_MAP = {
  "mesh": "mesh_config",
  "mla": "attention_config",
  "mha": "attention_config",
  "kdn": "attention_config",
  "rope": "rope_config",
  "embed": "embed_config",
  "embedding": "embed_config",
  "moe": "moe_config",
  "mlp": "mlp_config",
  "norm": "rmsnorm_config",
  "rmsnorm": "rmsnorm_config",
}


def comments_detail(config_type: type[Base]) -> str:
  comments = [f"{config_type.__name__}:"]
  global indentation_size
  indentation_size += INDENTATION_STRIDE
  for col in config_type.__table__.columns:
    if col.name == "id":
      continue
    comments.append(f"{' ' * indentation_size}{col.name}: {col.comment}")
  indentation_size -= INDENTATION_STRIDE
  return "\n".join(comments)


def _convert_value(attr_column, raw_value: str):
  attr_type = attr_column.type
  if isinstance(attr_type, SQLEnum):
    enum_class = attr_type.enum_class
    enum_value = None
    for enum_item in enum_class:  # type: ignore[union-attr]
      if enum_item.value.lower() == raw_value.lower():
        enum_value = enum_item
        break
    if enum_value is None:
      raise ValueError(
        f"Invalid enum value '{raw_value}' for {attr_column.key}. Valid values: "
        f"{[e.value for e in enum_class]}"  # type: ignore[union-attr]
      )
    return enum_value

  try:
    if issubclass(attr_type.python_type, bool):
      return str(raw_value).lower() not in ["0", "false", "no"]
    return attr_type.python_type(raw_value)
  except (ValueError, TypeError, NotImplementedError):
    normalized = raw_value.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
      items = [item.strip() for item in normalized.strip("[]").split(",") if item.strip()]
      try:
        return [ast.literal_eval(item) for item in items]
      except (ValueError, SyntaxError):
        return items
    return raw_value


def _normalize_attr_name(config_obj: type[Base], attr_name: str) -> str:
  if not issubclass(config_obj, ModelConfigBase):
    return attr_name
  slot = config_obj.slot_for_name(attr_name)
  if slot is None:
    return attr_name
  return slot.name_attr


def _resolve_config_class(
  section_name: str, attrs: dict[str, str], component_name: str | None
) -> type[Base]:
  if component_name:
    config_obj = CONFIG_REGISTRY.get(component_name)
  else:
    config_type = attrs.pop("type", None)
    config_obj = CONFIG_REGISTRY.get((config_type or section_name).lower())
  if config_obj is None:
    raise ValueError(
      f"Unknown config type: {component_name or section_name.lower()}. "
      f"Available types: {sorted(CONFIG_REGISTRY.keys())}"
    )
  return config_obj


def _parse_file(file: str, component_name: str | None = None):
  config = configparser.ConfigParser()
  config.read(file, encoding="utf-8")
  components: OrderedDict[str, Base] = OrderedDict()
  section_types: dict[str, str] = {}
  for section in config.sections():
    raw_attrs = dict(config.items(section))
    config_obj = _resolve_config_class(section, raw_attrs, component_name)
    attrs: dict[str, Any] = {}
    for k, v in raw_attrs.items():
      normalized_key = _normalize_attr_name(config_obj, k)
      attr_column = getattr(config_obj, normalized_key, None)
      if attr_column is None or not hasattr(attr_column, "type"):
        raise ValueError(f"Unknown attribute `{k}` for config type `{config_obj.__name__}`")
      attrs[normalized_key] = _convert_value(attr_column, v)
    instance = config_obj(**attrs)
    components[section] = instance
    section_types[section] = config_obj.__name__
  return components, section_types


def _bind_slot(model: ModelConfigBase, slot: ComponentSlotSpec, component: Base):
  if not isinstance(component, slot.component_types):
    allowed = ", ".join(cls.__name__ for cls in slot.component_types)
    raise ValueError(
      f"`{model.name}.{slot.attr}` expects one of ({allowed}), "
      f"but got {component.__class__.__name__}"
    )
  if hasattr(model, "bind_component"):
    model.bind_component(slot.attr, component)  # type: ignore[misc]
  else:
    setattr(model, slot.attr, component)
    setattr(model, slot.name_attr, component.name)


def _apply_reference_bindings(
  model: ModelConfigBase, configs_by_name: dict[str, Base]
) -> set[str]:
  bound_names: set[str] = set()
  for slot in model.component_slots():
    referenced_name = getattr(model, slot.name_attr, None)
    if not referenced_name:
      continue
    component = configs_by_name.get(referenced_name)
    if component is None:
      raise ValueError(
        f"`{model.name}.{slot.attr}` references unknown component `{referenced_name}`"
      )
    _bind_slot(model, slot, component)
    bound_names.add(component.name)
  return bound_names


def _apply_legacy_auto_bindings(
  models: list[ModelConfigBase], sections: OrderedDict[str, Base]
) -> set[str]:
  bound_names: set[str] = set()
  if len(models) != 1:
    return bound_names
  model = models[0]
  for section_name, component in sections.items():
    slot_attr = LEGACY_SECTION_SLOT_MAP.get(section_name.lower())
    if slot_attr is None:
      continue
    slot = model.slot_for_name(slot_attr)
    if slot is None or getattr(model, slot.name_attr, None):
      continue
    _bind_slot(model, slot, component)
    bound_names.add(component.name)
  return bound_names


def _apply_default_components(models: list[ModelConfigBase]) -> set[str]:
  synthetic_bound_names: set[str] = set()
  for model in models:
    for slot in model.component_slots():
      if getattr(model, slot.attr, None) is not None:
        continue
      if slot.default_factory is None:
        if slot.required:
          raise ValueError(
            f"`{model.name}` requires component slot `{slot.attr}` to be configured"
          )
        continue
      component = slot.default_factory(model)
      _bind_slot(model, slot, component)
      synthetic_bound_names.add(component.name)
  return synthetic_bound_names


def _construct_config(sections: OrderedDict[str, Base]):
  models = [cfg for cfg in sections.values() if isinstance(cfg, ModelConfigBase)]
  components = [cfg for cfg in sections.values() if not isinstance(cfg, ModelConfigBase)]
  if not models:
    return list(sections.values())

  configs_by_name = {config.name: config for config in sections.values()}
  bound_component_names: set[str] = set()
  for model in models:
    bound_component_names.update(_apply_reference_bindings(model, configs_by_name))
  bound_component_names.update(_apply_legacy_auto_bindings(models, sections))
  bound_component_names.update(_apply_default_components(models))

  unattached_components = [
    component
    for component in components
    if component.name not in bound_component_names
  ]
  return [*models, *unattached_components]


def show_attributes(component_name: str):
  config_type = CONFIG_REGISTRY.get(component_name)
  assert config_type is not None, f"`{component_name}` is not a valid config name"
  attributes = comments_detail(config_type)
  print(attributes)


def all_tables():
  return CONFIG_REGISTRY.keys()


def parse_file(file: str, module: str):
  component_name = module if module != "model" else None
  components, _ = _parse_file(file, component_name)
  return _construct_config(components)
