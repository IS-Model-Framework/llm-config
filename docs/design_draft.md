## 思路
- 配置参数和hlo export解耦
- 引入sqlite轻量级数据库，使用hlo_config工具管理数据库模型参数：包括create, update, delete，query
- hlo_export工具运行时，指定模型名称，查询相关配置，生成指定hlo

## layer层级参数管理
### 创建layer参数：

#### 基于配置文件生成
组件可以使用配置文件批量创建（MoE依赖其它组件，只能单个配置），如下配置文件创建多个mla组件：
```python
[Custom1]  
name=Custom1  
use_bias=false  
dropout=0.0  
kv_lora_rank=512  
q_lora_rank=1536  
qk_nope_head_dim=128  
qk_rope_head_dim=64  
v_head_dim=128  
num_query_heads=128  
num_kv_heads=128  
  
[Custom2]  
name=Custom2  
use_bias=false  
dropout=0.0  
kv_lora_rank=512  
q_lora_rank=1536  
qk_nope_head_dim=128  
qk_rope_head_dim=64  
v_head_dim=128  
num_query_heads=128  
num_kv_heads=128  
  
[Custom3]  
name=Custom3  
use_bias=false  
dropout=0.0  
kv_lora_rank=512  
q_lora_rank=1536  
qk_nope_head_dim=128  
qk_rope_head_dim=64  
v_head_dim=128  
num_query_heads=128  
num_kv_heads=128
```
使用指令如下：
```shell
hlo_config add --module mla --from-file ${FILE_PATH} 
```

MoE组件创建模板如下：
```python
[CUSTOM]  
name=CUSTOM  
aux_loss_alpha=0.001  
n_routed_experts=256  
n_shared_experts=1  
n_activated_experts=8  
norm_topk_prob=true  
score_func=sigmoid  
route_scale=2.5  
seq_aux=true  
shared_experts_activation=silu  
shared_experts_dim=2048  
routed_experts_activation=silu  
routed_experts_dim=2048
```
使用指令如下：
```shell
hlo_config add --module moe --from-file ${FILE_PATH}
```
#### 基于base layer创建
也可以基于现有的layer配置进行创建，比如基于上述mla组件`Custom1`创建新的mla，使用指令如下：
```shell
hlo_config add --module mla --from-base Custom1 --update name=new-Custom1,kv_lora_rank=1024
```
需要注意的是--from-file和--from-base不能同时使用；另外，name是区别组件的唯一键，并且该参数为用户指定，在创建新的layer必须创建name。
如MoE这种多组件构成的组件的属性配置如下：
```shell
hlo_config add --module moe --from-base CUSTOM --update name=new-CUSTOM,n_routed_experts=128```
如果嵌套组件未设置新的名称，则会与base组件共用相同的组件，在此情况下若有该组件的属性修改，则会影响其它组件或模型的属性，工具会提示用户影响范围并要求用户确认。
### 更新layer参数：
更新layer参数逻辑与**基于base layer创建**逻辑类似，指令如下：
```shell
hlo_config update --module mla --name new-Custom1 --update kv_lora_rank=1024

hlo_config update --module moe --name new-CUSTOM  --update moe.RoutedExperts.dim=1024
```
若修改组件被多次引用，工具会报出影响范围并要求用户确认。另外，组件创立之后是不能修改的，如果需要使用新名称，只能新创建一个新名称的组件操作。
### 删除layer参数：
```shell
hlo_config delete --module mla --name new-Custom1
```
在删除过程中，有其它组件或模型依赖该组件，则会报出影响范围并要求用户确认风险。

## 模型/subgraph参数管理
### 创建模型参数：

#### 基于配置文件生成
例如如下llama-3-8B配置文件：
```python
[Model]  
name=Llama-3-8B  
num_layers=32  
emb_dim=4096  
max_seq_len=8192  
vocab_size=128256  
tie_word_embeddings=false  
  
[MHA]  
name=Llama-3-8B  
use_bias=false  
dropout=0.0  
num_query_heads=32  
num_kv_heads=8  
  
[Rope]  
name=Llama-3-8B  
theta=500000.0  
type=rope  
  
[MLP]  
name=Llama-3-8B  
activation=silu  
dim=14336  
  
[Norm]  
name=Llama-3-8B  
epsilon=1e-05
```
使用如下指令配置文件，`hlo_config`工具就会自动将model信息和依赖组件信息加入到表中：
```shell
hlo_config add --module model --from-file ${FILE_PATH}
```
#### 基于base模型创建
逻辑上与**基于base layer创建**中创建MoE组件方式相似，指令如下：
```shell
hlo_config add --module model --from-base DeepSeek-V3-671B --update name=new-Llama-3-8B,num_layers=8,moe.name=new-CUSTOM-MOE,moe.shared_experts_dim=1024,moe.routed_experts_dim=1024,mla.kv_lora_rank=256
```
#### 基于已有layer创建
该场景下，只需关注model固有属性，组件通过--base-layer进行传递，model属性如下所示：
```python
[Model]  
name=Custom-1 
num_layers=32  
emb_dim=4096  
max_seq_len=8192  
vocab_size=128256  
tie_word_embeddings=false  
```
假设我们需要使用llama-3-8B的组件，可以使用如下指令：
```shell
hlo_config add --module model --from-file ${FILE_PATH} --base-layer mha.name=Llama-3-8B,rope.name=Llama-3-8B,mlp.name=Llama-3-8B,norm.name=Llama-3-8B
```
### 更新模型参数：
参数更新逻辑与**更新layer参数**逻辑相似，指令如下：
```shell
hlo_config update --module model --name Custom-1  --update num_layers=24,mlp.activation=gelu
```
另外不建议更新name参数，若更新了组件的信息，并且组件涉及其它模型引用，则会报出影响范围并要求用户确认。
### 删除模型参数：
类似**删除layer参数**中删除MoE组件逻辑，指令如下：
```shell
hlo_config delete --module model --name Custom-1 --all
```
若不使用--all，则只删除model信息，--all会删除所有组件，若删除组件被其它模型引用，则报出影响范围并要求用户确认
## 信息展示
### 模型/layer展示
展示所有模型，指令如下：
```shell
hlo_config show --module model --list

ModelConfig[mesh=None, name=DeepSeek-V3-671B, mla_config_name=DeepSeek-V3-671B, vocab_size=129280, mha_config_name=None, num_layers=61, mlp_config_name=DeepSeek-V3-671B-Dense, n_dense_layers=3, moe_config_name=DeepSeek-V3-671B, emb_dim=7168, batch_size=1, rmsnorm_config_name=DeepSeek-V3-671B, max_seq_len=163840, tie_word_embeddings=True, rope_config_name=DeepSeek-V3-671B, assemble=None]
ModelConfig[mesh=None, name=Llama-3-8B, mla_config_name=None, vocab_size=128256, mha_config_name=Llama-3-8B, num_layers=32, mlp_config_name=Llama-3-8B, n_dense_layers=0, moe_config_name=None, emb_dim=4096, batch_size=1, rmsnorm_config_name=Llama-3-8B, max_seq_len=8192, tie_word_embeddings=True, rope_config_name=Llama-3-8B, assemble=None]
```
展示模型信息：
```shell
hlo_config show --module model --name Llama-3-8B

ModelConfig[mesh=None, name=Llama-3-8B, mla_config_name=None, vocab_size=128256, mha_config_name=Llama-3-8B, num_layers=32, mlp_config_name=Llama-3-8B, n_dense_layers=0, moe_config_name=None, emb_dim=4096, batch_size=1, rmsnorm_config_name=Llama-3-8B, max_seq_len=8192, tie_word_embeddings=True, rope_config_name=Llama-3-8B, assemble=None]
```
展示模型细节信息
```shell
hlo_config show --module model --name Llama-3-8B --detailed

ModelConfig:
  mesh=None
  name=Llama-3-8B
  mla_config_name=None
  vocab_size=128256
  num_layers=32
  n_dense_layers=0
  moe_config_name=None
  emb_dim=4096
  batch_size=1
  max_seq_len=8192
  tie_word_embeddings=True
  assemble=None
  MHAConfig:
    num_query_heads=32
    head_dim=None
    dropout=0.0
    sharding=None
    name=Llama-3-8B
    num_kv_heads=8
    use_bias=True
  MLPConfig:
    sharding=None
    name=Llama-3-8B
    dim=14336
    activation=Activation.SiLU
  RMSNormConfig:
    name=Llama-3-8B
    epsilon=1e-05
  RopeConfig:
    beta_fast=None
    factor=None
    mscale=None
    type=RopeType.ROPE
    beta_slow=None
    theta=500000.0
    original_seq_len=None
    mscale_all_dim=None
    name=Llama-3-8B
```
展示属性值
```shell
hlo_config show --module model --name Llama-3-8B --attribute vocab_size

hlo_config show --module model --name Llama-3-8B --attribute mlp.dim
```
### 各个属性含义查询
当然用户对于每个字段属性含义可能不是很了解，我们贴心的加入了--list-attributes参数，用于获取每个字段的含义。
```shell
hlo_config show --module model --list-attributes

ModelConfig:
  emb_dim: The hidden size for the model.
  max_seq_len: ...
  vocab_size: ...
  ...
```

## 运行hlo_export
对于子层，model只需要设置一个组件即可，对于多个layer，model设置num_layers属性即可，因此hlo_export只需要围绕model展开即可，比如跑Llama-3-8B模型，执行指令如下：
```shell
hlo_export --model_name Llama-3-8B --output_dir ${CUSTOM_DIR}
```