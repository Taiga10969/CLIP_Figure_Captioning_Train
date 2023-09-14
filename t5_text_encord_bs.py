# 必要ライブラリのインポート
## T5のトークナイザーはtransformersライブラリを使用
## T5のモデル自体はlavisに記述してあるファイルを使用する
from transformers import T5TokenizerFast
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration

# モデル等の定義
t5_model="google/flan-t5-xl"
t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
t5_config = T5Config.from_pretrained(t5_model)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)
text_encoder = t5_model.get_encoder()


sample_texts = ['Figure 1: Training loss over train tokens for the 7B, 13B, 33B, and 65 models. LLaMA-33B and LLaMA- 65B were trained on 1.4T tokens. The smaller models were trained on 1.0T tokens. All models are trained with a batch size of 4M tokens.',
                'Figure 4. Prompt engineering and ensembling improve zeroshot performance. Compared to the baseline of using contextless class names, prompt engineering and ensembling boost zero-shot classification performance by almost 5 points on average across 36 datasets. This improvement is similar to the gain from using 4 times more compute with the baseline zero-shot method but is “free” when amortized over many predictions.'
                ]

# sample_textをID化
input_token = t5_tokenizer(text=sample_texts,
                     padding = 'longest', 
                     truncation = True, 
                     max_length = 74,
                     return_tensors = 'pt')

print('input_token.input_ids : ', input_token.input_ids)
print('input_token.input_ids.shape : ', input_token.input_ids.shape)
'''
input_token.input_ids :  tensor([[ 7996,   209,    10,  4017,  1453,   147,  2412, 14145,     7,    21,
             8,   489,   279,     6,  1179,   279,     6,  5400,   279,     6,
            11,  7123,  2250,     5,     3, 10376,     9,  4148,    18,  4201,
           279,    11,     3, 10376,     9,  4148,    18,  7123,   279,   130,
          4252,    30,     3, 14912,   382, 14145,     7,     5,    37,  2755,
          2250,   130,  4252,    30,     3, 12734,   382, 14145,     7,     5,
           432,  2250,    33,  4252,    28,     3,     9, 11587,   812,    13,
           314,   329, 14145,     1],
        [ 7996,  2853,   749,  1167,    17,  3867,    11,     3,    35,     7,
          8312,    53,  1172,  5733, 11159,   821,     5,     3, 25236,    12,
             8, 20726,    13,   338,  2625,   924,   853,  3056,     6,  9005,
          3867,    11,     3,    35,     7,  8312,    53,  4888,  5733,    18,
         11159, 13774,   821,    57,   966,   305,   979,    30,  1348,   640,
          4475, 17953,     7,     5,   100,  4179,    19,  1126,    12,     8,
          2485,    45,   338,   314,   648,    72, 29216,    28,     8, 20726,
          5733,    18, 11159,     1]])
input_token.input_ids.shape :  torch.Size([2, 74])
'''

tokens = t5_tokenizer.convert_ids_to_tokens(input_token.input_ids[0].cpu().numpy())
print('input_tokens :', tokens)
'''
input_tokens : ['▁Figure', '▁1', ':', '▁Training', '▁loss', '▁over', '▁train', '▁token', 's', '▁for', '▁the', '▁7', 'B', ',', '▁13', 'B', ',', '▁33', 'B', ',', '▁and', '▁65', '▁models', '.', '▁', 'LL', 'a', 'MA', '-', '33', 'B', '▁and', '▁', 'LL', 'a', 'MA', '-', '▁65', 'B', '▁were', '▁trained', '▁on', '▁', '1.4', 'T', '▁token', 's', '.', '▁The', '▁smaller', '▁models', '▁were', '▁trained', '▁on', '▁', '1.0', 'T', '▁token', 's', '.', '▁All', '▁models', '▁are', '▁trained', '▁with', '▁', 'a', '▁batch', '▁size', '▁of', '▁4', 'M', '▁token', '</s>']
'''

## ***********
#embed = t5_model.encoder.embed_tokens(input_token.input_ids)
#print('embed : ', embed.shape)
## ***********



# input_idを入力してテキストエンコーダの特徴量を取得
text_encoder_output = text_encoder(input_ids=input_token.input_ids)
#print('text_encoder_output : ', text_encoder_output)
print('text_encoder_output.last_hidden_state : ', text_encoder_output.last_hidden_state)
print('text_encoder_output.last_hidden_state.shape : ', text_encoder_output.last_hidden_state.shape)
print('</s>token_feature : ', text_encoder_output.last_hidden_state[:, -1, :].shape)
'''
text_encoder_output.last_hidden_state :  tensor([[[ 1.2709e-01, -4.2119e-02,  3.7257e-02,  ..., -8.5750e-02,
           1.9233e-01,  1.0170e-01],
         [ 9.2493e-02,  2.9511e-02, -1.0588e-01,  ..., -1.1965e-01,
           2.7691e-02,  6.9907e-02],
         [ 1.6742e-01, -7.9956e-02, -1.7301e-02,  ..., -5.4499e-02,
           8.4012e-02, -6.4990e-03],
         ...,
         [ 1.5538e-01, -2.8798e-02,  1.1639e-01,  ...,  1.5481e-01,
           1.0606e-01, -1.1275e-01],
         [ 1.7010e-01,  4.8919e-02,  2.4157e-01,  ..., -1.5599e-02,
          -4.5981e-02,  4.7309e-02],
         [-8.3551e-03,  1.0572e-03,  1.8490e-03,  ..., -7.4398e-05,
          -1.6891e-03,  1.2818e-02]],

        [[ 5.7057e-02, -6.1782e-02,  5.8979e-02,  ..., -1.1779e-01,
           1.7063e-01,  1.8660e-01],
         [ 6.8282e-02,  3.8917e-02, -1.2811e-01,  ...,  5.2718e-02,
           1.0435e-01,  7.1700e-02],
         [ 9.9613e-02,  8.7166e-02,  8.2053e-02,  ...,  1.6900e-02,
           8.5330e-02,  7.3580e-02],
         ...,
         [ 3.1613e-02, -9.1391e-02,  1.1930e-01,  ..., -4.0691e-02,
           1.7914e-02,  6.6370e-02],
         [-2.9842e-02,  6.6166e-02, -2.8555e-02,  ..., -1.5849e-01,
          -9.4729e-02,  9.2183e-03],
         [-7.4291e-03,  1.2939e-03,  5.1283e-04,  ...,  5.5802e-04,
          -3.1155e-03,  1.2578e-02]]], grad_fn=<MulBackward0>)
text_encoder_output.last_hidden_state.shape :  torch.Size([2, 74, 2048])
</s>token_feature :  torch.Size([2, 2048])
'''






'''
T5TokenizerFast(vocab_file: Any | None = None,
                tokenizer_file: Any | None = None,
                eos_token: str = "</s>",
                unk_token: str = "<unk>",
                pad_token: str = "<pad>",
                extra_ids: int = 100,
                additional_special_tokens: Any | None = None,
                **kwargs: Any
                )
'''
