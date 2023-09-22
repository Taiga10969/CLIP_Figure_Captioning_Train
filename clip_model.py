import numpy as np
import torch
import torch.nn as nn
from transformers import T5TokenizerFast
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration



class CLIP_model(nn.Module):
    def __init__(self, text_encoder, vision_encoder, vision_encoder_dim, text_encoder_dim, max_length=74, target_value=1, t=0.1):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.vision_linear = nn.Linear(vision_encoder_dim, text_encoder_dim)
        
        self.max_length = max_length
        self.target_value = target_value # インデックスを取得するidの値

        self.t = t
    
    def forward(self, input_ids, attention_mask, images):
        
        index = torch.nonzero(input_ids == self.target_value, as_tuple=False).tolist()
        index = [item[1] for item in index]

        text_encoder_output = self.text_encoder(input_ids=input_ids,
                                                attention_mask = attention_mask)
        
        #_text_encoderで埋め込まれた特徴量
        
        texts_output = text_encoder_output.last_hidden_state[list(range(text_encoder_output.last_hidden_state.shape[0])), index]
        #print('texts_output.shape : ', texts_output.shape)
        
        texts_output = texts_output.to(torch.float)

        # 画像データをGPUに転送
        #images = images.to(device)
        image_output = self.vision_encoder(images)
        image_output = self.vision_linear(image_output)
        image_output = image_output[:, 0, :]
        #print('image_output.shape : ', image_output.shape)

        texts_output = nn.functional.normalize(texts_output, p=2, dim=1)
        image_output = nn.functional.normalize(image_output, p=2, dim=1)

        return texts_output, image_output
        
        #データパラレル時に計算ができなくなるので，モデル外で行う
        similarity = torch.mm(texts_output, image_output.T) * np.exp(self.t)
        #print('similarity.shape : ', similarity.shape)
        #print('similarity : ', similarity)
        
        #return similarity



if __name__=='__main__':
    from tqdm import tqdm
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.transforms import transforms

    from scicap_dataset import SciCapDataset

    from lavis.models import load_model_and_preprocess

    from torchinfo import summary

    def torch_fix_seed(seed=42):
        # Python random
        random.seed(seed)
        # Numpy
        np.random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

    torch_fix_seed()

    # check GPU usage
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count()==0: print('Use 1 GPU')
    else: print(f'Use {torch.cuda.device_count()} GPUs')

    sicap_data_path = '/taiga/Datasets/scicap_data'

    transform=transforms.Compose([transforms.Resize((224, 224)), 
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                ])
    _, vis_processors, _ = load_model_and_preprocess(name='blip2_t5',
                                                     model_type='pretrain_flant5xl', #pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
                                                     is_eval=True,
                                                     #device=device
                                                    )

    dataset = SciCapDataset(dataset_path = sicap_data_path, 
                            transform = transform,
                            train = 'ALL',               # 学習用データなのか
                            train_include_val = True,         # データにvalデータを含めるか
                            include_subfig = True,     # データにsubfigデータを含めるか
                            use_remove = True,          # lowercase-and-token-and-remove（figure:などを消去した）データを使用するか
                            retrun_caption = True,
                            vis_processors=None,
                            )
    

    print("Use_Dataset_len : ", len(dataset))

    batch_size = 5
    
    # DataLoaderを使用してデータをバッチで読み込む
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    Iter = iter(dataloader)
    image, caption, img_path = next(Iter)
    
    image = image.to(device, dtype=torch.float)  # 画像をGPUに転送
    #caption = caption.to(device)  # キャプションをGPUに転送
    
    vision_encoder_dim = 1408
    text_encoder_dim = 2048

    t5_model="google/flan-t5-xl"
    t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
    t5_config = T5Config.from_pretrained(t5_model)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)
    text_encoder = t5_model.get_encoder()

    for param in text_encoder.parameters():
        param.requires_grad = False

    vision_encoder = create_eva_vit_g(img_size = 224, 
                                           drop_path_rate = 0, 
                                           use_checkpoint = False, 
                                           #precision = "fp16",
                                           )
    
    #summary(text_encoder)
    #summary(vision_encoder)

    t5_tokenizer
    text_encoder.to(torch.float)
    vision_encoder.to(torch.float)

    model = CLIP_model(t5_tokenizer, text_encoder, vision_encoder, vision_encoder_dim, text_encoder_dim)
    model = model.to(device)  # モデルをGPUに転送
    mode = model.to(torch.float)

    #summary(model)

    outputs = model(caption, image)


    label = torch.eye(batch_size).to(device)
    #label = torch.tensor([1, 2, 3, 4, 5]).to(device)
    print(label)

    criterion = nn.CrossEntropyLoss()

    # クロスエントロピーロスを計算
    # 交差エントロピー損失関数を定義
    #def cross_entropy_loss(logits, target):
    #    loss = F.cross_entropy(logits, torch.argmax(target, dim=1))
    #    return loss
    
    loss = criterion(outputs, torch.argmax(label, dim=1))
    
    # バッチ内の各データポイントの平均ロスを求める
    batch_size = outputs.shape[0]
    average_loss = loss / batch_size
    
    print("クロスエントロピーロス:", loss.item())
    print("平均ロス:", average_loss.item())
