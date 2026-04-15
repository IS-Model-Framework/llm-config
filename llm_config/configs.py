import enum
from typing import List, Optional
from sqlalchemy import Integer, ForeignKey, Boolean, Enum, Float, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.ext.mutable import MutableList

from llm_config.base import Base


__all__ = [
    "MeshConfig",
    "ModelConfig",
    "MLAConfig",
    "MHAConfig",
    "MLPConfig",
    "MoEConfig",
    "RopeConfig",
    "EmbeddingConfig",
    "RMSNormConfig",
    "Hardware",
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

    model_config: Mapped[List["ModelConfig"]] = relationship(
        back_populates="mesh_config",
        cascade="all, delete-orphan",
        init=False,
        lazy="immediate"
    )

    axes_name: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None)
    shape: Mapped[List[int]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None)
    hardware: Mapped[Hardware] = mapped_column(Enum(Hardware), default_factory=lambda: Hardware.CPU)


# ==============================================================================
# model
# ==============================================================================
class ModelConfig(Base):
    __tablename__ = "ModelConfig"

    emb_dim: Mapped[int] = mapped_column(Integer, comment="The hidden size for the model.")
    max_seq_len: Mapped[int] = mapped_column(Integer)
    vocab_size: Mapped[int] = mapped_column(Integer)
    num_layers: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)
    n_dense_layers: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    batch_size: Mapped[int] = mapped_column(Integer, default_factory=lambda: 1)
    tie_word_embeddings: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    activation_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default_factory=lambda: Dtype.FLOAT32)
    assemble: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=list)

    data_sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=list)
    output_sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=list)

    mesh_config_name: Mapped[str] = mapped_column(ForeignKey("MeshConfig.name"), nullable=True, default_factory=lambda: None)
    mesh_config: Mapped["MeshConfig"] = relationship(back_populates="model_config", init=False, lazy="immediate")
    mla_config_name: Mapped[str] = mapped_column(ForeignKey("MLAConfig.name"), nullable=True, default_factory=lambda: None)
    mla_config: Mapped["MLAConfig"] = relationship(back_populates="model_config", init=False, lazy="immediate")
    mha_config_name: Mapped[str] = mapped_column(ForeignKey("MHAConfig.name"), nullable=True, default_factory=lambda: None)
    mha_config: Mapped["MHAConfig"] = relationship(back_populates="model_config", init=False, lazy="immediate")
    mlp_config_name: Mapped[str] = mapped_column(ForeignKey("MLPConfig.name"), nullable=True, default_factory=lambda: None)
    mlp_config: Mapped["MLPConfig"] = relationship(back_populates="model_config", init=False, lazy="immediate")
    moe_config_name: Mapped[str] = mapped_column(ForeignKey("MoEConfig.name"), nullable=True, default_factory=lambda: None)
    moe_config: Mapped["MoEConfig"] = relationship(back_populates="model_config", init=False, lazy="immediate")
    rmsnorm_config_name: Mapped[str] = mapped_column(ForeignKey("RMSNormConfig.name"), nullable=True, default_factory=lambda: None)
    rmsnorm_config: Mapped["RMSNormConfig"] = relationship(back_populates="model_config", init=False, lazy="immediate")
    rope_config_name: Mapped[str] = mapped_column(ForeignKey("RopeConfig.name"), nullable=True, default_factory=lambda: None)
    rope_config: Mapped["RopeConfig"] = relationship(back_populates="model_config", init=False, lazy="immediate")
    embed_config_name: Mapped[str] = mapped_column(ForeignKey("EmbeddingConfig.name"), nullable=True, default_factory=lambda: None)
    embed_config: Mapped["EmbeddingConfig"] = relationship(back_populates="model_config", init=False, lazy="immediate")

    export_mlp: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_moe: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_rmsnorm: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_embed: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_transformer_body: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_model_computation: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_loss_computation: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_mla: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_mha: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_whole_computation: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)

    def __hash__(self):
        return hash(self.__repr__())


# ==============================================================================
# attention
# ==============================================================================
class AttentionConfig(Base):
    __abstract__ = True

    num_query_heads: Mapped[int] = mapped_column(Integer)
    num_kv_heads: Mapped[int] = mapped_column(Integer)
    dropout: Mapped[float] = mapped_column(Float, nullable=True, default_factory=lambda: None)
    sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None)
    out_projection_use_bias: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
    matmul_precision: Mapped[Dtype] = mapped_column(Enum(Dtype), default_factory=lambda: Dtype.FLOAT32)
    weight_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default_factory=lambda: Dtype.FLOAT32)
    use_softmax: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    use_scale: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)

    export_qkv_projection: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_dot_attention: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_out_projection: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)


class MLAConfig(AttentionConfig):
    __tablename__ = "MLAConfig"

    model_config: Mapped[Optional[List["ModelConfig"]]] = relationship(
        back_populates="mla_config",
        cascade="all, delete-orphan",
        init=False,
        lazy = "immediate"
    )

    q_lora_rank: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    kv_lora_rank: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    qk_nope_head_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    qk_rope_head_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    v_head_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)

    use_indexer: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
    scale_fmt: Mapped[str] = mapped_column(String(10), nullable=True, default_factory=lambda: None)
    index_n_heads: Mapped[int] = mapped_column(Integer, nullable=True, default_factory=lambda: None)
    index_head_dim: Mapped[int] = mapped_column(Integer, nullable=True, default_factory=lambda: None)
    index_topk: Mapped[int] = mapped_column(Integer, nullable=True, default_factory=lambda: None)


class MHAConfig(AttentionConfig):
    __tablename__ = "MHAConfig"

    model_config: Mapped[Optional[List["ModelConfig"]]] = relationship(
        back_populates="mha_config",
        cascade="all, delete-orphan",
        init=False,
        lazy="immediate"
    )

    head_dim: Mapped[int] = mapped_column(Integer, nullable=True, default_factory=lambda: None)


# ==============================================================================
# mlp
# ==============================================================================
class MLPConfig(Base):
    __tablename__ = "MLPConfig"

    model_config: Mapped[Optional[List["ModelConfig"]]] = relationship(
        back_populates="mlp_config",
        cascade="all, delete-orphan",
        init=False,
        lazy="immediate"
    )

    dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    activation: Mapped[Activation] = mapped_column(Enum(Activation), default_factory=lambda: Activation.SiLU)
    dropout: Mapped[float] = mapped_column(Float, nullable=True, default_factory=lambda: None)
    sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None)
    use_bias: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
    use_gate: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
    matmul_precision: Mapped[Dtype] = mapped_column(Enum(Dtype), default_factory=lambda: Dtype.FLOAT32)
    weight_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default_factory=lambda: Dtype.FLOAT32)
    activations_in_float32: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)

    export_up_projection: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_down_projection: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)


class MoEConfig(Base):
    __tablename__ = "MoEConfig"

    model_config: Mapped[Optional[List["ModelConfig"]]] = relationship(
        back_populates="moe_config",
        cascade="all, delete-orphan",
        init=False,
        lazy="immediate"
    )
    # common attributes
    matmul_precision: Mapped[Dtype] = mapped_column(Enum(Dtype), default_factory=lambda: Dtype.FLOAT32)
    weight_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default_factory=lambda: Dtype.FLOAT32)


    # routed experts attributes
    n_routed_experts: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    n_activated_experts: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    routed_experts_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    routed_experts_activation: Mapped[Activation] = mapped_column(Enum(Activation), default_factory=lambda: Activation.SiLU)
    routed_experts_use_bias: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
    aux_loss_alpha: Mapped[float] = mapped_column(Float, nullable=True, default_factory=lambda: None)
    norm_topk_prob: Mapped[bool] = mapped_column(Boolean, nullable=True, default_factory=lambda: None)
    score_func: Mapped[Activation] = mapped_column(Enum(Activation), nullable=True, default_factory=lambda: None)
    route_scale: Mapped[float] = mapped_column(Float, nullable=True, default_factory=lambda: None)
    seq_aux: Mapped[bool] = mapped_column(Boolean, nullable=True, default_factory=lambda: None)

    # sharding attributes
    shared_experts_sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None)
    routed_experts_sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None)
    gate_logit_sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None)

    # shared experts attributes
    shared_experts_dim: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    n_shared_experts: Mapped[int] = mapped_column(Integer, default_factory=lambda: 0)
    shared_experts_activation: Mapped[Activation] = mapped_column(Enum(Activation), default_factory=lambda: Activation.SiLU)
    shared_experts_use_bias: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
    shared_experts_use_gate: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: False)
    shared_experts_activations_in_float32: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    shared_experts_dropout: Mapped[float] = mapped_column(Float, default_factory=lambda: None)

    # gmm attributes
    tile_batch_seq: Mapped[int] = mapped_column(Integer, nullable=False, default_factory=lambda: 512)
    tile_activation_dim: Mapped[int] = mapped_column(Integer, nullable=False, default_factory=lambda: 1024)
    tile_weight_dim: Mapped[int] = mapped_column(Integer, nullable=False, default_factory=lambda: 1024)

    # export attributes
    export_routed_block: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_shared_block: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_gate_logit: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_permute: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_unpermute: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    export_routed_mlp: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)


    def shared_experts_to_mlp(self) -> MLPConfig:
        return MLPConfig(
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


# ==============================================================================
# embedding
# ==============================================================================
class RopeConfig(Base):
    __tablename__ = "RopeConfig"

    model_config: Mapped[Optional[List["ModelConfig"]]] = relationship(
        back_populates="rope_config",
        cascade="all, delete-orphan",
        init=False,
        lazy="immediate"
    )

    theta: Mapped[float] = mapped_column(Float, default_factory=lambda: 0.0)
    type: Mapped[RopeType] = mapped_column(Enum(RopeType), default_factory=lambda: RopeType.ROPE)
    beta_fast: Mapped[int] = mapped_column(Integer, nullable=True, default_factory=lambda: None)
    beta_slow: Mapped[int] = mapped_column(Integer, nullable=True, default_factory=lambda: None)
    factor: Mapped[int] = mapped_column(Integer, nullable=True, default_factory=lambda: None)
    original_seq_len: Mapped[int] = mapped_column(Integer, nullable=True, default_factory=lambda: None)
    mscale: Mapped[float] = mapped_column(Float, nullable=True, default_factory=lambda: None)
    mscale_all_dim: Mapped[float] = mapped_column(Float, nullable=True, default_factory=lambda: None)

    min_timescale: Mapped[Integer] = mapped_column(Float, nullable=False, default_factory=lambda: 1)
    max_timescale: Mapped[Integer] = mapped_column(Float, nullable=False, default_factory=lambda: 10000)


class EmbeddingConfig(Base):
    __tablename__ = "EmbeddingConfig"

    model_config: Mapped[Optional[List["ModelConfig"]]] = relationship(
        back_populates="embed_config",
        cascade="all, delete-orphan",
        init=False,
        lazy="immediate"
    )

    sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None)
    use_iota_embed: Mapped[bool] = mapped_column(Boolean, default_factory=lambda: True)
    attend_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default_factory=lambda: Dtype.FLOAT32)


# ==============================================================================
# norm
# ==============================================================================
class RMSNormConfig(Base):
    __tablename__ = "RMSNormConfig"

    model_config: Mapped[Optional[List["ModelConfig"]]] = relationship(
        back_populates="rmsnorm_config",
        cascade="all, delete-orphan",
        init=False,
        lazy="immediate"
    )

    epsilon: Mapped[float] = mapped_column(Float, default_factory=lambda: 1.e-05)
    sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True, default_factory=lambda: None)
    weight_dtype: Mapped[Dtype] = mapped_column(Enum(Dtype), default_factory=lambda: Dtype.FLOAT32)
