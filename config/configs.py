import enum
from typing import List
from sqlalchemy import Integer, ForeignKey, JSON, Boolean, Enum, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.ext.mutable import MutableList

from config.base import Base


__all__ = [
    "ModelConfig",
    "MLAConfig",
    "MHAConfig",
    "MLPConfig",
    "MoEConfig",
    "RopeConfig",
    "RMSNormConfig",
]


class Activation(enum.Enum):
    SiLU = "silu"
    GeLU = "gelu"
    Sigmoid = "sigmoid"


class RopeType(enum.Enum):
    ROPE = "rope"
    YARN = "yarn"


# ==============================================================================
# model
# ==============================================================================
class ModelConfig(Base):
    __tablename__ = "ModelConfig"

    emb_dim: Mapped[int] = mapped_column(Integer, comment="The hidden size for the model.")
    max_seq_len: Mapped[int] = mapped_column(Integer)
    vocab_size: Mapped[int] = mapped_column(Integer)

    num_layers: Mapped[int] = mapped_column(Integer, default=1)
    n_dense_layers: Mapped[int] = mapped_column(Integer, default=0)
    batch_size: Mapped[int] = mapped_column(Integer, default=1)
    tie_word_embeddings: Mapped[bool] = mapped_column(Boolean, default=True)
    assemble: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True)
    mesh: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True)

    mla_config_name: Mapped[str] = mapped_column(ForeignKey("MLAConfig.name"), nullable=True)
    mla_config: Mapped["MLAConfig"] = relationship(back_populates="model_config")
    mha_config_name: Mapped[str] = mapped_column(ForeignKey("MHAConfig.name"), nullable=True)
    mha_config: Mapped["MHAConfig"] = relationship(back_populates="model_config")
    mlp_config_name: Mapped[str] = mapped_column(ForeignKey("MLPConfig.name"), nullable=True)
    mlp_config: Mapped["MLPConfig"] = relationship(back_populates="model_config")
    moe_config_name: Mapped[str] = mapped_column(ForeignKey("MoEConfig.name"), nullable=True)
    moe_config: Mapped["MoEConfig"] = relationship(back_populates="model_config")
    rmsnorm_config_name: Mapped[str] = mapped_column(ForeignKey("RMSNormConfig.name"), nullable=True)
    rmsnorm_config: Mapped["RMSNormConfig"] = relationship(back_populates="model_config")
    rope_config_name: Mapped[str] = mapped_column(ForeignKey("RopeConfig.name"), nullable=True)
    rope_config: Mapped["RopeConfig"] = relationship(back_populates="model_config")


# ==============================================================================
# attention
# ==============================================================================
class AttentionConfig(Base):
    __abstract__ = True

    num_query_heads: Mapped[int] = mapped_column(Integer)
    num_kv_heads: Mapped[int] = mapped_column(Integer)
    dropout: Mapped[float] = mapped_column(Float, nullable=True)
    use_bias: Mapped[bool] = mapped_column(Boolean, default=False)
    sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True)


class MLAConfig(AttentionConfig):
    __tablename__ = "MLAConfig"

    q_lora_rank: Mapped[int] = mapped_column(Integer)
    kv_lora_rank: Mapped[int] = mapped_column(Integer)
    qk_nope_head_dim: Mapped[int] = mapped_column(Integer)
    qk_rope_head_dim: Mapped[int] = mapped_column(Integer)
    v_head_dim: Mapped[int] = mapped_column(Integer)

    model_config: Mapped[List["ModelConfig"]] = relationship(
        back_populates="mla_config",
        cascade="all, delete-orphan"
    )


class MHAConfig(AttentionConfig):
    __tablename__ = "MHAConfig"

    head_dim: Mapped[int] = mapped_column(Integer, nullable=True)

    model_config: Mapped[List["ModelConfig"]] = relationship(
        back_populates="mha_config",
        cascade="all, delete-orphan"
    )


# ==============================================================================
# mlp
# ==============================================================================
class MLPConfig(Base):
    __tablename__ = "MLPConfig"

    dim: Mapped[int] = mapped_column(Integer)
    activation: Mapped[Activation] = mapped_column(Enum(Activation))
    sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True)

    model_config: Mapped[List["ModelConfig"]] = relationship(
        back_populates="mlp_config",
        cascade="all, delete-orphan"
    )


class MoEConfig(Base):
    __tablename__ = "MoEConfig"

    n_routed_experts: Mapped[int] = mapped_column(Integer)
    n_shared_experts: Mapped[int] = mapped_column(Integer, default=0)
    n_activated_experts: Mapped[int] = mapped_column(Integer)
    aux_loss_alpha: Mapped[float] = mapped_column(Float, nullable=True)
    norm_topk_prob: Mapped[bool] = mapped_column(Boolean, nullable=True)
    score_func: Mapped[Activation] = mapped_column(Enum(Activation), nullable=True)
    route_scale: Mapped[float] = mapped_column(Float, nullable=True)
    seq_aux: Mapped[bool] = mapped_column(Boolean, nullable=True)

    shared_experts_dim: Mapped[int] = mapped_column(Integer)
    shared_experts_activation: Mapped[Activation] = mapped_column(Enum(Activation))

    routed_experts_dim: Mapped[int] = mapped_column(Integer)
    routed_experts_activation: Mapped[Activation] = mapped_column(Enum(Activation))

    shared_experts_sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True)
    routed_experts_sharding: Mapped[List[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=True)

    model_config: Mapped[List["ModelConfig"]] = relationship(
        back_populates="moe_config",
        cascade="all, delete-orphan"
    )


# ==============================================================================
# embedding
# ==============================================================================
class RopeConfig(Base):
    __tablename__ = "RopeConfig"

    theta: Mapped[float] = mapped_column(Float)
    type: Mapped[RopeType] = mapped_column(Enum(RopeType))
    beta_fast: Mapped[int] = mapped_column(Integer, nullable=True)
    beta_slow: Mapped[int] = mapped_column(Integer, nullable=True)
    factor: Mapped[int] = mapped_column(Integer, nullable=True)
    original_seq_len: Mapped[int] = mapped_column(Integer, nullable=True)
    mscale: Mapped[float] = mapped_column(Float, nullable=True)
    mscale_all_dim: Mapped[float] = mapped_column(Float, nullable=True)

    model_config: Mapped[List["ModelConfig"]] = relationship(
        back_populates="rope_config",
        cascade="all, delete-orphan"
    )


# ==============================================================================
# norm
# ==============================================================================
class RMSNormConfig(Base):
    __tablename__ = "RMSNormConfig"

    epsilon: Mapped[float] = mapped_column(Float, default=1.e-05)

    model_config: Mapped[List["ModelConfig"]] = relationship(
        back_populates="rmsnorm_config",
        cascade="all, delete-orphan"
    )
