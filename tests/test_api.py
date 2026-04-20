import unittest
from pathlib import Path

from llm_config.api import parse_file
from llm_config.configs import (
  DeepSeekModelConfig,
  EmbeddingConfig,
  KDNConfig,
  MLAConfig,
  ModelConfig,
  T5TextEncoderConfig,
  WanModelConfig,
)


FIXTURES = Path(__file__).with_name("fixtures")


class ParseFileTests(unittest.TestCase):
  def test_legacy_model_sections_still_parse(self):
    parsed = parse_file(str(FIXTURES / "legacy_model.ini"), "model")

    self.assertEqual(len(parsed), 1)
    model = parsed[0]
    self.assertIsInstance(model, ModelConfig)
    self.assertIsInstance(model.attention_config, MLAConfig)
    self.assertIs(model.attention_config, model.mla_config)
    self.assertEqual(model.mla_config_name, "legacy_mla")
    self.assertEqual(model.mlp_config_name, "legacy_mlp")
    self.assertIsInstance(model.embed_config, EmbeddingConfig)
    self.assertEqual(model.embed_config.name, "legacy")

  def test_new_deepseek_model_parses_kdn_attention(self):
    parsed = parse_file(str(FIXTURES / "deepseek_kdn.ini"), "model")

    self.assertEqual(len(parsed), 1)
    model = parsed[0]
    self.assertIsInstance(model, DeepSeekModelConfig)
    self.assertIsInstance(model.attention_config, KDNConfig)
    self.assertEqual(model.attention_config_name, "ds_kdn")
    self.assertIsNone(model.mla_config)
    self.assertIsNone(model.mha_config)
    self.assertEqual(model.moe_config_name, "ds_moe")

  def test_multimodal_wan_model_binds_nested_components_and_backbone(self):
    parsed = parse_file(str(FIXTURES / "wan_multimodal.ini"), "model")

    self.assertEqual(len(parsed), 2)
    wan_model = next(item for item in parsed if isinstance(item, WanModelConfig))
    backbone = next(
      item for item in parsed if isinstance(item, DeepSeekModelConfig)
    )
    self.assertIsInstance(wan_model.text_encoder, T5TextEncoderConfig)
    self.assertEqual(wan_model.text_encoder_name, "t5_encoder")
    self.assertIs(wan_model.backbone, backbone)
    self.assertIsInstance(backbone.attention_config, MLAConfig)
    self.assertEqual(backbone.attention_config_name, "backbone_mla")

  def test_invalid_component_reference_raises(self):
    with self.assertRaisesRegex(ValueError, "unknown component"):
      parse_file(str(FIXTURES / "broken_wan.ini"), "model")


if __name__ == "__main__":
  unittest.main()
