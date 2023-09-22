import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from torchinfo import summary
from tqdm import tqdm

from transformers import T5TokenizerFast
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration

import utils
from scicap_dataset import SciCapDataset
from clip_model import CLIP_model
import config

config = config.Config()

if config.wandb == True:
     import wandb
     wandb.init(project = config.project_name)

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

### Create Model =================================================================================
t5_tokenizer = T5TokenizerFast.from_pretrained(config.t5_model)
t5_config = T5Config.from_pretrained(config.t5_model)
t5_model = T5ForConditionalGeneration.from_pretrained(config.t5_model, config=t5_config)
text_encoder = t5_model.get_encoder()

vision_encoder = create_eva_vit_g(img_size = 224, 
                                  drop_path_rate = 0, 
                                  use_checkpoint = False, 
                                  #precision = "fp16",
                                  )

# text_encoderについては，パラメータを凍結する
for param in text_encoder.parameters():
        param.requires_grad = False

text_encoder.to(torch.float)
vision_encoder.to(torch.float)

model = CLIP_model(t5_tokenizer, text_encoder, vision_encoder, config.vision_encoder_dim, config.text_encoder_dim)
model = nn.DataParallel(model)
model = model.to(device)  # モデルをGPUに転送
mode = model.to(torch.float)
print('==== CLIP_Model_summary')
summary(model)

### Create Dataset and DataLoader==================================================================
transform=transforms.Compose([transforms.Resize((224, 224)), 
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])

dataset = SciCapDataset(dataset_path = config.scicap_data_path, 
                        transform = transform,
                        train = 'ALL',                    # 学習用データなのか
                        train_include_val = True,         # データにvalデータを含めるか
                        include_subfig = True,            # データにsubfigデータを含めるか
                        use_remove = True,                # lowercase-and-token-and-remove（figure:などを消去した）データを使用するか
                        retrun_caption = True,
                        vis_processors=None,
                        )

print("==== Use_Dataset_len : ", len(dataset))

# DataLoaderを使用してデータをバッチで読み込む
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

### train =========================================================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(params = model.parameters(),
                        lr=config.lr, 
                        betas=config.betas, 
                        eps=config.eps, 
                        weight_decay=config.weight_decay)

lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                              T_0=config.t_0, 
                                                              T_mult=config.t_mult)

### one epoch train ================================================================================
def train(model, train_loader, criterion, optimizer, device):
    # ネットワークモデルを学習モードに設定
    model.train()
    sum_loss = 0.0
    count = 0
    
    # tqdmを使って進捗バーを表示
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    # tqdmを使って進捗バーを表示
    for image, caption, _ in progress_bar:
        count += config.batch_size

        image = image.to(device, dtype=torch.float)
        label = torch.eye(image.shape[0]).to(device)

        optimizer.zero_grad()

        texts_output, image_output = model(caption, image)

        print(texts_output.shape)
        print(image_output.shape)

        outputs = torch.mm(texts_output, image_output.T) * np.exp(config.t)
        print('outputs.shape : ', outputs.shape)
        print('outputs : ', outputs)
        
        outputs = outputs.reshape(config.batch_size, config.batch_size)
        
        print(outputs.shape)
        print(torch.argmax(label, dim=0).shape)

        loss = (criterion(outputs, torch.argmax(label, dim=1)) + criterion(outputs, torch.argmax(label, dim=0))) / 2
        
        loss.backward()
        
        optimizer.step()
        sum_loss += loss.item()

        # lossを進捗バーに表示
        progress_bar.set_postfix(loss=loss.item())

        # wandb
        if config.wandb == True:
            wandb.log({"iter_loss": loss.item(), 
                       "lr": optimizer.param_groups[0]["lr"], 
                       "wd": optimizer.param_groups[0]["weight_decay"]})

    return sum_loss / count


### TRAIN_MAIN =========================================================================================
train_loss_list = []
for epoch in range(1, config.num_epoch+1):
    train_loss = train(model, dataloader, criterion, optimizer, device)
    train_loss_list.append(train_loss)

    # エポックごとのlossを表示
    print(f"==== Epoch {epoch}/{config.num_epoch}, Loss: {train_loss:.4f}")

    # wandb
    if config.wandb == True:
        wandb.log({"epoch_loss": train_loss()})

