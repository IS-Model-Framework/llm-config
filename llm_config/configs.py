import enum
from dataclasses import dataclass, field
from typing import Callable, ClassVar

from sqlalchemy import JSON, Boolean, Enum, Float, Integer, String
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Mapped, mapped_column

from llm_config.base import Base


@dataclass(frozen=True)
class ComponentSlotSpec:
  attr: str
  name_attr: str
  component_types: tuple[type[Base], ...]
  required: bool = False
  aliases: tuple[str, ...] = ()
  default_factory: Callable[["ModelConfigBase"], Base] | None = None


__all__ = [
  "Activation",
  "AttentionConfigBase",
  "ComponentConfig",
  "ComponentSlotSpec",
  "DecoderConfig",
  "DeepSeekModelConfig",
  "Dtype",
  "EmbeddingConfig",
  "EncoderConfigBase",
  "FusionConfig",
  "Hardware",
  "KDNConfig",
  "LinearAttentionConfigBase",
  "MHAConfig",
  "MLAConfig",
  "MLPConfig",
  "MeshConfig",
  "ModelConfig",
  "ModelConfigBase",
  "MoEConfig",
  "ProjectorConfig",
  "RMSNormConfig",
  "RopeConfig",
  "RopeType",
  "T5TextEncoderConfig",
  "TextEncoderConfig",
  "TransformerLMConfig",
  "TransformerModelConfigBase",
  "VaeDecoderConfig",
  "VaeEncoderConfig",
  "VisionEncoderConfig",
  "WanModelConfig",
]


class Activation(enum.Enum):
  SiLU = "silu"
  GeLU = "gelu"
  Sigmoid = "sigmoid"


class RopeType(enum.Enum):
  ROPE = "rope"
  YARN = "yarn"


class Hardware(enum.Enum):
  CPU = "cpu"
  TPU = "tpu"
  GPU = "gpu"


class Dtype(enum.Enum):
  FLOAT16 = "float16"
  BFLOAT16 = "bfloat16"
  FLOAT32 = "float32"
  INT32 = "int32"


class ComponentConfig(Base):
  __abstract__ = True
  __allow_unmapped__ = True


class ModelConfigBase(Base):
  __abstract__ = True
  __allow_unmapped__ = True
  __component_slots__: ClassVar[tuple[ComponentSlotSpec, ...]] = ()

  emb_dim: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  max_seq_len: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  vocab_size: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  num_layers: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)
  n_dense_layers: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  batch_size: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)
  tie_word_embeddings: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  activation_dtype: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )
  assemble: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=list
  )
  data_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=list
  )
  output_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=list
  )

  def __hash__(self):
    return hash(self.__repr__())

  @classmethod
  def component_slots(cls) -> tuple[ComponentSlotSpec, ...]:
    slots: dict[str, ComponentSlotSpec] = {}
    for klass in reversed(cls.__mro__):
      for slot in getattr(klass, "__component_slots__", ()):
        slots[slot.attr] = slot
    return tuple(slots.values())

  @classmethod
  def slot_for_name(cls, attr_name: str) -> ComponentSlotSpec | None:
    for slot in cls.component_slots():
      if attr_name in {slot.attr, slot.name_attr, *slot.aliases}:
        return slot
    return None


class MeshConfig(ComponentConfig):
  __tablename__ = "MeshConfig"

  axes_name: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None
  )
  shape: Mapped[list[int]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None
  )
  hardware: Mapped[Hardware] = mapped_column(
    Enum(Hardware), default_factory=lambda: Hardware.CPU
  )


class AttentionConfigBase(ComponentConfig):
  __abstract__ = True

  num_query_heads: Mapped[int] = mapped_column(Integer)
  num_kv_heads: Mapped[int] = mapped_column(Integer)
  dropout: Mapped[float] = mapped_column(
    Float, nullable=True, default_factory=lambda: None
  )
  sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None
  )
  out_projection_use_bias: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: False
  )
  matmul_precision: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )
  weight_dtype: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )
  use_softmax: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  use_scale: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_qkv_projection: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  export_dot_attention: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  export_out_projection: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )


class MLAConfig(AttentionConfigBase):
  __tablename__ = "MLAConfig"

  q_lora_rank: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  kv_lora_rank: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  qk_nope_head_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  qk_rope_head_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  v_head_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  use_indexer: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
  scale_fmt: Mapped[str] = mapped_column(
    String(10), nullable=True, default_factory=lambda: None
  )
  index_n_heads: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  index_head_dim: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  index_topk: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )


class MHAConfig(AttentionConfigBase):
  __tablename__ = "MHAConfig"

  head_dim: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )


class LinearAttentionConfigBase(AttentionConfigBase):
  __abstract__ = True

  state_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  chunk_size: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  causal: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)


class KDNConfig(LinearAttentionConfigBase):
  __tablename__ = "KDNConfig"

  kernel_type: Mapped[str] = mapped_column(String(30), default_factory=lambda: "relu")
  low_rank_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  normalize_kernel: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )


class MLPConfig(ComponentConfig):
  __tablename__ = "MLPConfig"

  dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  activation: Mapped[Activation] = mapped_column(
    Enum(Activation), default_factory=lambda: Activation.SiLU
  )
  dropout: Mapped[float] = mapped_column(
    Float, nullable=True, default_factory=lambda: None
  )
  sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None
  )
  use_bias: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
  use_gate: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
  matmul_precision: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )
  weight_dtype: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )
  activations_in_float32: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  export_up_projection: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  export_down_projection: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )


class MoEConfig(ComponentConfig):
  __tablename__ = "MoEConfig"

  matmul_precision: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )
  weight_dtype: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )
  n_routed_experts: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  n_activated_experts: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  routed_experts_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  routed_experts_activation: Mapped[Activation] = mapped_column(
    Enum(Activation), default_factory=lambda: Activation.SiLU
  )
  routed_experts_use_bias: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: False
  )
  aux_loss_alpha: Mapped[float] = mapped_column(
    Float, nullable=True, default_factory=lambda: None
  )
  norm_topk_prob: Mapped[bool] = mapped_column(
    Boolean, nullable=True, default_factory=lambda: None
  )
  score_func: Mapped[Activation] = mapped_column(
    Enum(Activation), nullable=True, default_factory=lambda: None
  )
  route_scale: Mapped[float] = mapped_column(
    Float, nullable=True, default_factory=lambda: None
  )
  seq_aux: Mapped[bool] = mapped_column(
    Boolean, nullable=True, default_factory=lambda: None
  )
  shared_experts_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None
  )
  routed_experts_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None
  )
  gate_logit_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None
  )
  shared_experts_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  n_shared_experts: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  shared_experts_activation: Mapped[Activation] = mapped_column(
    Enum(Activation), default_factory=lambda: Activation.SiLU
  )
  shared_experts_use_bias: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: False
  )
  shared_experts_use_gate: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: False
  )
  shared_experts_activations_in_float32: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  shared_experts_dropout: Mapped[float] = mapped_column(
    Float, nullable=True, default_factory=lambda: None
  )
  tile_batch_seq: Mapped[int] = mapped_column(
    Integer, nullable=False, default_factory=lambda: 512
  )
  tile_activation_dim: Mapped[int] = mapped_column(
    Integer, nullable=False, default_factory=lambda: 1024
  )
  tile_weight_dim: Mapped[int] = mapped_column(
    Integer, nullable=False, default_factory=lambda: 1024
  )
  export_routed_block: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  export_shared_block: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  export_gate_logit: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_permute: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_unpermute: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_routed_mlp: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)

  def shared_experts_to_mlp(self) -> MLPConfig:
    return MLPConfig(
      name=self.name,
      dim=self.shared_experts_dim * self.n_shared_experts,
      activation=self.shared_experts_activation,
      use_bias=self.shared_experts_use_bias,
      sharding=self.shared_experts_sharding,
      use_gate=self.shared_experts_use_gate,
      matmul_precision=self.matmul_precision,
      weight_dtype=self.weight_dtype,
      activations_in_float32=self.shared_experts_activations_in_float32,
      dropout=self.shared_experts_dropout,
    )


class RopeConfig(ComponentConfig):
  __tablename__ = "RopeConfig"

  theta: Mapped[float] = mapped_column(Float, default_factory=lambda: 0.0)
  type: Mapped[RopeType] = mapped_column(
    Enum(RopeType), default_factory=lambda: RopeType.ROPE
  )
  beta_fast: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  beta_slow: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  factor: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  original_seq_len: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  mscale: Mapped[float] = mapped_column(
    Float, nullable=True, default_factory=lambda: None
  )
  mscale_all_dim: Mapped[float] = mapped_column(
    Float, nullable=True, default_factory=lambda: None
  )
  min_timescale: Mapped[int] = mapped_column(Float, default_factory=lambda: 1)
  max_timescale: Mapped[int] = mapped_column(Float, default_factory=lambda: 10000)


class EmbeddingConfig(ComponentConfig):
  __tablename__ = "EmbeddingConfig"

  sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None
  )
  use_iota_embed: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  attend_dtype: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )


class RMSNormConfig(ComponentConfig):
  __tablename__ = "RMSNormConfig"

  epsilon: Mapped[float] = mapped_column(Float, default_factory=lambda: 1.0e-05)
  sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None
  )
  weight_dtype: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )


class EncoderConfigBase(ComponentConfig):
  __abstract__ = True

  hidden_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  num_layers: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)
  max_seq_len: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )
  dropout: Mapped[float] = mapped_column(
    Float, nullable=True, default_factory=lambda: None
  )
  activation_dtype: Mapped[Dtype] = mapped_column(
    Enum(Dtype), default_factory=lambda: Dtype.FLOAT32
  )


class TextEncoderConfig(EncoderConfigBase):
  __tablename__ = "TextEncoderConfig"

  vocab_size: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)


class T5TextEncoderConfig(EncoderConfigBase):
  __tablename__ = "T5TextEncoderConfig"

  vocab_size: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  ffw_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  num_heads: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)


class VisionEncoderConfig(EncoderConfigBase):
  __tablename__ = "VisionEncoderConfig"

  image_size: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  patch_size: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  num_channels: Mapped[int] = mapped_column(Integer, default_factory=lambda: 3)


class VaeEncoderConfig(ComponentConfig):
  __tablename__ = "VaeEncoderConfig"

  latent_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  image_size: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  downsample_factor: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)
  use_quant_conv: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)


class VaeDecoderConfig(ComponentConfig):
  __tablename__ = "VaeDecoderConfig"

  latent_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  image_size: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  upsample_factor: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)


class ProjectorConfig(ComponentConfig):
  __tablename__ = "ProjectorConfig"

  input_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  output_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  num_layers: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)
  use_bias: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)


class FusionConfig(ComponentConfig):
  __tablename__ = "FusionConfig"

  fusion_type: Mapped[str] = mapped_column(String(30), default_factory=lambda: "cross_attention")
  hidden_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  num_layers: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)


class DecoderConfig(ComponentConfig):
  __tablename__ = "DecoderConfig"

  hidden_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
  num_layers: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)
  vocab_size: Mapped[int] = mapped_column(
    Integer, nullable=True, default_factory=lambda: None
  )


class TransformerModelConfigBase(ModelConfigBase):
  __abstract__ = True

  mesh_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  attention_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  mla_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  mha_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  mlp_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  moe_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  rmsnorm_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  rope_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  embed_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  export_mlp: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_moe: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_rmsnorm: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_embed: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_lm_head: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_transformer_body: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  export_model_computation: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  export_loss_computation: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )
  export_mla: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_mha: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
  export_whole_computation: Mapped[bool] = mapped_column(
    Boolean, default_factory=lambda: True
  )

  mesh_config: MeshConfig | None = field(default=None, init=False, repr=False)
  attention_config: AttentionConfigBase | None = field(
    default=None, init=False, repr=False
  )
  mla_config: MLAConfig | None = field(default=None, init=False, repr=False)
  mha_config: MHAConfig | None = field(default=None, init=False, repr=False)
  mlp_config: MLPConfig | None = field(default=None, init=False, repr=False)
  moe_config: MoEConfig | None = field(default=None, init=False, repr=False)
  rmsnorm_config: RMSNormConfig | None = field(default=None, init=False, repr=False)
  rope_config: RopeConfig | None = field(default=None, init=False, repr=False)
  embed_config: EmbeddingConfig | None = field(default=None, init=False, repr=False)

  __component_slots__ = (
    ComponentSlotSpec("mesh_config", "mesh_config_name", (MeshConfig,)),
    ComponentSlotSpec(
      "attention_config",
      "attention_config_name",
      (AttentionConfigBase,),
      aliases=("mla_config", "mha_config"),
    ),
    ComponentSlotSpec("mlp_config", "mlp_config_name", (MLPConfig,)),
    ComponentSlotSpec("moe_config", "moe_config_name", (MoEConfig,)),
    ComponentSlotSpec("rmsnorm_config", "rmsnorm_config_name", (RMSNormConfig,)),
    ComponentSlotSpec("rope_config", "rope_config_name", (RopeConfig,)),
    ComponentSlotSpec(
      "embed_config",
      "embed_config_name",
      (EmbeddingConfig,),
      default_factory=lambda model: EmbeddingConfig(name=model.name),
    ),
  )

  def bind_component(self, attr: str, component: Base):
    setattr(self, attr, component)
    name_attr = f"{attr}_name"
    if hasattr(self, name_attr):
      setattr(self, name_attr, component.name)
    if attr == "attention_config":
      self._sync_attention_aliases(component)

  def _sync_attention_aliases(self, component: Base):
    self.attention_config = component  # type: ignore[assignment]
    self.attention_config_name = component.name
    if isinstance(component, MLAConfig):
      self.mla_config = component
      self.mla_config_name = component.name
      self.mha_config = None
      self.mha_config_name = None
    elif isinstance(component, MHAConfig):
      self.mha_config = component
      self.mha_config_name = component.name
      self.mla_config = None
      self.mla_config_name = None
    else:
      self.mla_config = None
      self.mla_config_name = None
      self.mha_config = None
      self.mha_config_name = None


class TransformerLMConfig(TransformerModelConfigBase):
  __tablename__ = "TransformerLMConfig"


class DeepSeekModelConfig(TransformerModelConfigBase):
  __tablename__ = "DeepSeekModelConfig"

  moe_layer_freq: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)
  first_k_dense_replace: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)


class WanModelConfig(ModelConfigBase):
  __tablename__ = "WanModelConfig"

  text_encoder_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  image_encoder_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  video_encoder_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  audio_encoder_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  vae_encoder_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  vae_decoder_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  projector_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  fusion_config_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  decoder_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )
  backbone_name: Mapped[str] = mapped_column(
    String(30), nullable=True, default_factory=lambda: None
  )

  text_encoder: TextEncoderConfig | T5TextEncoderConfig | None = field(
    default=None, init=False, repr=False
  )
  image_encoder: VisionEncoderConfig | None = field(
    default=None, init=False, repr=False
  )
  video_encoder: VisionEncoderConfig | None = field(
    default=None, init=False, repr=False
  )
  audio_encoder: EncoderConfigBase | None = field(default=None, init=False, repr=False)
  vae_encoder: VaeEncoderConfig | None = field(default=None, init=False, repr=False)
  vae_decoder: VaeDecoderConfig | None = field(default=None, init=False, repr=False)
  projector: ProjectorConfig | None = field(default=None, init=False, repr=False)
  fusion_config: FusionConfig | None = field(default=None, init=False, repr=False)
  decoder: DecoderConfig | TransformerLMConfig | None = field(
    default=None, init=False, repr=False
  )
  backbone: TransformerLMConfig | DeepSeekModelConfig | None = field(
    default=None, init=False, repr=False
  )

  __component_slots__ = (
    ComponentSlotSpec(
      "text_encoder",
      "text_encoder_name",
      (TextEncoderConfig, T5TextEncoderConfig),
      required=True,
    ),
    ComponentSlotSpec(
      "image_encoder",
      "image_encoder_name",
      (VisionEncoderConfig,),
    ),
    ComponentSlotSpec(
      "video_encoder",
      "video_encoder_name",
      (VisionEncoderConfig,),
    ),
    ComponentSlotSpec("audio_encoder", "audio_encoder_name", (EncoderConfigBase,)),
    ComponentSlotSpec(
      "vae_encoder",
      "vae_encoder_name",
      (VaeEncoderConfig,),
      required=True,
    ),
    ComponentSlotSpec("vae_decoder", "vae_decoder_name", (VaeDecoderConfig,)),
    ComponentSlotSpec("projector", "projector_name", (ProjectorConfig,)),
    ComponentSlotSpec("fusion_config", "fusion_config_name", (FusionConfig,)),
    ComponentSlotSpec(
      "decoder",
      "decoder_name",
      (DecoderConfig, TransformerLMConfig, DeepSeekModelConfig),
    ),
    ComponentSlotSpec(
      "backbone",
      "backbone_name",
      (TransformerLMConfig, DeepSeekModelConfig),
    ),
  )

  def bind_component(self, attr: str, component: Base):
    setattr(self, attr, component)
    name_attr = f"{attr}_name"
    if hasattr(self, name_attr):
      setattr(self, name_attr, component.name)


ModelConfig = TransformerLMConfig
