# LLM Config

## Overview
This is a repo that abstract all LLM into components config for convenient manual.

## Features

- Manual config as components for LLM.
- Add components by base or file by command.
- Update/Delete/Show components by command.
- Decouple configure and construct the LLM.
- Unify the attributes name for all components.

## Requirements

- sqlalchemy

## Installation

```bash
pip install https://github.com/IS-Model-Framework/llm-config.git@gemini_code
```

## Usage

#### add components
```bash
usage: llm_config add [-h] [--module {model,mla,mha,rope,embed,moe,mlp,norm}] [--from-file FROM_FILE]
                      [--from-base FROM_BASE] [--name NAME] [--attributes ATTRIBUTES] [--base-layer BASE_LAYER]

options:
  -h, --help            show this help message and exit
  --module {model,mla,mha,rope,embed,moe,mlp,norm}
  --from-file FROM_FILE
  --from-base FROM_BASE
  --name NAME
  --attributes ATTRIBUTES
  --base-layer BASE_LAYER
```

#### update components
```bash
usage: llm_config update [-h] [--module {model,mla,mha,rope,embed,moe,mlp,norm}] [--from-file FROM_FILE]
                         [--from-base FROM_BASE] [--name NAME] [--attributes ATTRIBUTES]

options:
  -h, --help            show this help message and exit
  --module {model,mla,mha,rope,embed,moe,mlp,norm}
  --from-file FROM_FILE
  --from-base FROM_BASE
  --name NAME
  --attributes ATTRIBUTES
```

#### delete components
```bash
usage: llm_config delete [-h] [--module {model,mla,mha,rope,embed,moe,mlp,norm}] [--name NAME] [--all]

options:
  -h, --help            show this help message and exit
  --module {model,mla,mha,rope,embed,moe,mlp,norm}
  --name NAME
  --all
```

#### show components
```bash
usage: llm_config show [-h] [--module {model,mla,mha,rope,embed,moe,mlp,norm}] [--name NAME] [--detailed]
                       [--attribute ATTRIBUTE] [--list-attributes] [--list]

options:
  -h, --help            show this help message and exit
  --module {model,mla,mha,rope,embed,moe,mlp,norm}
  --name NAME
  --detailed
  --attribute ATTRIBUTE
  --list-attributes
  --list
```

### Example

#### add model from file
- We can config the file refer to the template([deepseek](./llm_config/config/templates/deepseek-v3-671B.ini) and [llama](./llm_config/config/templates/llama-3-8B.ini))
```shell
llm_config add --module model --from-file ${FILE_PATH}
```

#### add model from base
```shell
llm_config add --module model --from-base ${base_name} --attibutes name=${new_model_name},num_layers=8,moe.shared_experts_dim=1024
```

#### add components from file
- Components configuration file refer to the template([MLA](./llm_config/config/templates/mla.ini) and [MoE](./llm_config/config/templates/moe.ini))
```shell
llm_config add --module ${component} --from-file ${FILE_PATH}
```

#### add components from base
```shell
llm_config add --module model --from-base ${base_name} --attibutes name=${new_component_name},...
```

#### update components attributes directly
```bash
llm_config update --module mlp --name ${your_component_name}  --update activation=gelu
```

#### update components attributes base on model
```bash
llm_config update --module model --name ${your_model_name}  --update mlp.activation=gelu
```

#### delete components
```bash
llm_config delete --module ${component} --name ${your_component_name}
```

#### delete model and its depended components
```bash
llm_config delete --module model --name ${your_model_name} --all
```

#### show model list
```bash
llm_config show --module model --list
```

#### show single model
```bash
llm_config show --module model --name ${your_model_name}
```

#### show single model and its components detailed
```bash
llm_config show --module model --name ${your_model_name} --detailed
```

#### show single components
```bash
llm_config show --module ${component} --name ${your_component_name}
```

#### show single attribute of components
```bash
llm_config show --module ${component} --name ${your_component_name} --attribute ${attribute_name}
```

#### show the meaning of attributes for components
```bash
llm_config show --module ${component} --list-attributes
```


## License

llm-config is licensed under Shanghai Infiscale Proprietary License.


## Questions

For issues and feature requests, please open an issue on the [GitHub repository](https://github.com/IS-Model-Framework/llm-config/issues).
