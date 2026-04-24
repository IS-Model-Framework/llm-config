import enum

from sqlalchemy import JSON, Boolean, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship

from llm_config.base import Base

__all__ = [
  "EmbeddingConfig",
  "Hardware",
  "MHAConfig",
  "MLAConfig",
  "MLPConfig",
  "MeshConfig",
  "MoEConfig",
  "ModelConfig",
  "RMSNormConfig",
  "RopeConfig",
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


class MeshConfig(Base):
  __tablename__ = "MeshConfig"

  model_config: Mapped[list["ModelConfig"]] = relationship(
    back_populates="mesh_config",
    cascade="all, delete-orphan",
    init=False,
    lazy="immediate",
  )

  axes_name: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default=None
  )
  shape: Mapped[list[int]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default=None
  )
  hardware: Mapped[Hardware] = mapped_column(Enum(Hardware), default=Hardware.CPU)


# ==============================================================================
# model
# ==============================================================================
class ModelConfig(Base):
  __tablename__ = "ModelConfig"

  emb_dim: Mapped[int] = mapped_column(
    Integer, comment="The hidden size for the model."
  )
  max_seq_len: Mapped[int] = mapped_column(Integer)
  vocab_size: Mapped[int] = mapped_column(Integer)
  num_layers: Mapped[int] = mapped_column(Integer, default=1)
  n_dense_layers: Mapped[int] = mapped_column(Integer, default=0)
  batch_size: Mapped[int] = mapped_column(Integer, default=1)
  tie_word_embeddings: Mapped[bool] = mapped_column(Boolean, default=True)
  activation_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default=Dtype.FLOAT32)
  assemble: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=list
  )

  data_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=list
  )
  output_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default_factory=list
  )

  mesh_config_name: Mapped[str] = mapped_column(
    ForeignKey("MeshConfig.name"), nullable=True, default=None
  )
  mesh_config: Mapped["MeshConfig"] = relationship(
    back_populates="model_config", init=False, lazy="immediate"
  )
  mla_config_name: Mapped[str] = mapped_column(
    ForeignKey("MLAConfig.name"), nullable=True, default=None
  )
  mla_config: Mapped["MLAConfig"] = relationship(
    back_populates="model_config", init=False, lazy="immediate"
  )
  mha_config_name: Mapped[str] = mapped_column(
    ForeignKey("MHAConfig.name"), nullable=True, default=None
  )
  mha_config: Mapped["MHAConfig"] = relationship(
    back_populates="model_config", init=False, lazy="immediate"
  )
  mlp_config_name: Mapped[str] = mapped_column(
    ForeignKey("MLPConfig.name"), nullable=True, default=None
  )
  mlp_config: Mapped["MLPConfig"] = relationship(
    back_populates="model_config", init=False, lazy="immediate"
  )
  moe_config_name: Mapped[str] = mapped_column(
    ForeignKey("MoEConfig.name"), nullable=True, default=None
  )
  moe_config: Mapped["MoEConfig"] = relationship(
    back_populates="model_config", init=False, lazy="immediate"
  )
  rmsnorm_config_name: Mapped[str] = mapped_column(
    ForeignKey("RMSNormConfig.name"), nullable=True, default=None
  )
  rmsnorm_config: Mapped["RMSNormConfig"] = relationship(
    back_populates="model_config", init=False, lazy="immediate"
  )
  rope_config_name: Mapped[str] = mapped_column(
    ForeignKey("RopeConfig.name"), nullable=True, default=None
  )
  rope_config: Mapped["RopeConfig"] = relationship(
    back_populates="model_config", init=False, lazy="immediate"
  )
  embed_config_name: Mapped[str] = mapped_column(
    ForeignKey("EmbeddingConfig.name"), nullable=True, default=None
  )
  embed_config: Mapped["EmbeddingConfig"] = relationship(
    back_populates="model_config", init=False, lazy="immediate"
  )

  export_mlp: Mapped[bool] = mapped_column(Boolean, default=True)
  export_moe: Mapped[bool] = mapped_column(Boolean, default=True)
  export_rmsnorm: Mapped[bool] = mapped_column(Boolean, default=True)
  export_embed: Mapped[bool] = mapped_column(Boolean, default=True)
  export_lm_head: Mapped[bool] = mapped_column(Boolean, default=True)
  export_transformer_body: Mapped[bool] = mapped_column(Boolean, default=True)
  export_model_computation: Mapped[bool] = mapped_column(Boolean, default=True)
  export_loss_computation: Mapped[bool] = mapped_column(Boolean, default=True)
  export_mla: Mapped[bool] = mapped_column(Boolean, default=True)
  export_mha: Mapped[bool] = mapped_column(Boolean, default=True)
  export_whole_computation: Mapped[bool] = mapped_column(Boolean, default=True)

  def __hash__(self):
    return hash(self.__repr__())


# ==============================================================================
# attention
# ==============================================================================
class AttentionConfig(Base):
  __abstract__ = True

  num_query_heads: Mapped[int] = mapped_column(Integer)
  num_kv_heads: Mapped[int] = mapped_column(Integer)
  dropout: Mapped[float] = mapped_column(Float, nullable=True, default=None)
  sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default=None
  )
  out_projection_use_bias: Mapped[bool] = mapped_column(Boolean, default=False)
  matmul_precision: Mapped[Dtype] = mapped_column(Enum(Dtype), default=Dtype.FLOAT32)
  weight_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default=Dtype.FLOAT32)
  use_softmax: Mapped[bool] = mapped_column(Boolean, default=True)
  use_scale: Mapped[bool] = mapped_column(Boolean, default=True)

  export_qkv_projection: Mapped[bool] = mapped_column(Boolean, default=True)
  export_dot_attention: Mapped[bool] = mapped_column(Boolean, default=True)
  export_out_projection: Mapped[bool] = mapped_column(Boolean, default=True)


class MLAConfig(AttentionConfig):
  __tablename__ = "MLAConfig"

  model_config: Mapped[list["ModelConfig"] | None] = relationship(
    back_populates="mla_config",
    cascade="all, delete-orphan",
    init=False,
    lazy="immediate",
  )

  q_lora_rank: Mapped[int] = mapped_column(Integer, default=0)
  kv_lora_rank: Mapped[int] = mapped_column(Integer, default=0)
  qk_nope_head_dim: Mapped[int] = mapped_column(Integer, default=0)
  qk_rope_head_dim: Mapped[int] = mapped_column(Integer, default=0)
  v_head_dim: Mapped[int] = mapped_column(Integer, default=0)

  use_indexer: Mapped[bool] = mapped_column(Boolean, default=False)
  scale_fmt: Mapped[str] = mapped_column(String(10), nullable=True, default=None)
  index_n_heads: Mapped[int] = mapped_column(Integer, nullable=True, default=None)
  index_head_dim: Mapped[int] = mapped_column(Integer, nullable=True, default=None)
  index_topk: Mapped[int] = mapped_column(Integer, nullable=True, default=None)


class MHAConfig(AttentionConfig):
  __tablename__ = "MHAConfig"

  model_config: Mapped[list["ModelConfig"] | None] = relationship(
    back_populates="mha_config",
    cascade="all, delete-orphan",
    init=False,
    lazy="immediate",
  )

  head_dim: Mapped[int] = mapped_column(Integer, nullable=True, default=None)


# ==============================================================================
# mlp
# ==============================================================================
class MLPConfig(Base):
  __tablename__ = "MLPConfig"

  model_config: Mapped[list["ModelConfig"] | None] = relationship(
    back_populates="mlp_config",
    cascade="all, delete-orphan",
    init=False,
    lazy="immediate",
  )

  dim: Mapped[int] = mapped_column(Integer, default=0)
  activation: Mapped[Activation] = mapped_column(
    Enum(Activation), default=Activation.SiLU
  )
  dropout: Mapped[float] = mapped_column(Float, nullable=True, default=None)
  sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default=None
  )
  use_bias: Mapped[bool] = mapped_column(Boolean, default=False)
  use_gate: Mapped[bool] = mapped_column(Boolean, default=False)
  matmul_precision: Mapped[Dtype] = mapped_column(Enum(Dtype), default=Dtype.FLOAT32)
  weight_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default=Dtype.FLOAT32)
  activations_in_float32: Mapped[bool] = mapped_column(Boolean, default=True)

  export_up_projection: Mapped[bool] = mapped_column(Boolean, default=True)
  export_down_projection: Mapped[bool] = mapped_column(Boolean, default=True)


class MoEConfig(Base):
  __tablename__ = "MoEConfig"

  model_config: Mapped[list["ModelConfig"] | None] = relationship(
    back_populates="moe_config",
    cascade="all, delete-orphan",
    init=False,
    lazy="immediate",
  )
  # common attributes
  matmul_precision: Mapped[Dtype] = mapped_column(Enum(Dtype), default=Dtype.FLOAT32)
  weight_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default=Dtype.FLOAT32)

  # routed experts attributes
  n_routed_experts: Mapped[int] = mapped_column(Integer, default=0)
  n_activated_experts: Mapped[int] = mapped_column(Integer, default=0)
  routed_experts_dim: Mapped[int] = mapped_column(Integer, default=0)
  routed_experts_activation: Mapped[Activation] = mapped_column(
    Enum(Activation), default=Activation.SiLU
  )
  routed_experts_use_bias: Mapped[bool] = mapped_column(Boolean, default=False)
  aux_loss_alpha: Mapped[float] = mapped_column(Float, nullable=True, default=None)
  norm_topk_prob: Mapped[bool] = mapped_column(Boolean, nullable=True, default=None)
  score_func: Mapped[Activation] = mapped_column(
    Enum(Activation), nullable=True, default=None
  )
  route_scale: Mapped[float] = mapped_column(Float, nullable=True, default=None)
  seq_aux: Mapped[bool] = mapped_column(Boolean, nullable=True, default=None)
  n_expert_groups: Mapped[int] = mapped_column(Integer, default=1)
  n_limited_groups: Mapped[int] = mapped_column(Integer, default=1)

  # sharding attributes
  shared_experts_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default=None
  )
  routed_experts_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default=None
  )
  gate_logit_sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default=None
  )

  # shared experts attributes
  shared_experts_dim: Mapped[int] = mapped_column(Integer, default=0)
  n_shared_experts: Mapped[int] = mapped_column(Integer, default=0)
  shared_experts_activation: Mapped[Activation] = mapped_column(
    Enum(Activation), default=Activation.SiLU
  )
  shared_experts_use_bias: Mapped[bool] = mapped_column(Boolean, default=False)
  shared_experts_use_gate: Mapped[bool] = mapped_column(Boolean, default=False)
  shared_experts_activations_in_float32: Mapped[bool] = mapped_column(
    Boolean, default=True
  )
  shared_experts_dropout: Mapped[float] = mapped_column(
    Float, nullable=True, default=None
  )

  # gmm attributes
  tile_batch_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=512)
  tile_activation_dim: Mapped[int] = mapped_column(
    Integer, nullable=False, default=1024
  )
  tile_weight_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=1024)

  # export attributes
  export_routed_block: Mapped[bool] = mapped_column(Boolean, default=True)
  export_shared_block: Mapped[bool] = mapped_column(Boolean, default=True)
  export_gate_logit: Mapped[bool] = mapped_column(Boolean, default=True)
  export_permute: Mapped[bool] = mapped_column(Boolean, default=True)
  export_unpermute: Mapped[bool] = mapped_column(Boolean, default=True)
  export_routed_mlp: Mapped[bool] = mapped_column(Boolean, default=True)
  export_shared_block_up_projection: Mapped[bool] = mapped_column(Boolean, default=True)
  export_shared_block_down_projection: Mapped[bool] = mapped_column(Boolean, default=True)

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
      export_up_projection=self.export_shared_block_up_projection,
      export_down_projection=self.export_shared_block_down_projection,
    )


# ==============================================================================
# embedding
# ==============================================================================
class RopeConfig(Base):
  __tablename__ = "RopeConfig"

  model_config: Mapped[list["ModelConfig"] | None] = relationship(
    back_populates="rope_config",
    cascade="all, delete-orphan",
    init=False,
    lazy="immediate",
  )

  theta: Mapped[float] = mapped_column(Float, default=0.0)
  type: Mapped[RopeType] = mapped_column(Enum(RopeType), default=RopeType.ROPE)
  beta_fast: Mapped[int] = mapped_column(Integer, nullable=True, default=None)
  beta_slow: Mapped[int] = mapped_column(Integer, nullable=True, default=None)
  factor: Mapped[int] = mapped_column(Integer, nullable=True, default=None)
  original_seq_len: Mapped[int] = mapped_column(Integer, nullable=True, default=None)
  mscale: Mapped[float] = mapped_column(Float, nullable=True, default=None)
  mscale_all_dim: Mapped[float] = mapped_column(Float, nullable=True, default=None)

  min_timescale: Mapped[Integer] = mapped_column(Float, nullable=False, default=1)
  max_timescale: Mapped[Integer] = mapped_column(Float, nullable=False, default=10000)


class EmbeddingConfig(Base):
  __tablename__ = "EmbeddingConfig"

  model_config: Mapped[list["ModelConfig"] | None] = relationship(
    back_populates="embed_config",
    cascade="all, delete-orphan",
    init=False,
    lazy="immediate",
  )

  sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default=None
  )
  use_iota_embed: Mapped[bool] = mapped_column(Boolean, default=True)
  attend_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default=Dtype.FLOAT32)


# ==============================================================================
# norm
# ==============================================================================
class RMSNormConfig(Base):
  __tablename__ = "RMSNormConfig"

  model_config: Mapped[list["ModelConfig"] | None] = relationship(
    back_populates="rmsnorm_config",
    cascade="all, delete-orphan",
    init=False,
    lazy="immediate",
  )

  epsilon: Mapped[float] = mapped_column(Float, default=1.0e-05)
  sharding: Mapped[list[str]] = mapped_column(
    MutableList.as_mutable(JSON), nullable=True, default=None
  )
  weight_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default=Dtype.FLOAT32)
