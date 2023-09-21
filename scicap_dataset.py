import os
from PIL import Image
import json
from torch.utils.data import Dataset


class SciCapDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 transform = None,
                 train = True,               # 学習用データなのか
                 train_include_val = True,         # データにvalデータを含めるか
                 include_subfig = False,     # データにsubfigデータを含めるか
                 use_remove = True,          # lowercase-and-token-and-remove（figure:などを消去した）データを使用するか
                 retrun_caption = True,
                 vis_processors=None,
                 ):
        
        self.path = dataset_path
        self.transform = transform
        self.train = train
        self.train_include_val = train_include_val
        self.include_subfig = include_subfig
        self.use_remove = use_remove
        self.return_caption = retrun_caption
        self.vis_processors = vis_processors

        self.image_filenames = self._load_image_filenames()

    def _load_image_filenames(self):
        image_filenames = []

        if self.train == True:
            file_path = os.path.join(self.path, "SciCap-No-Subfig-Img", "train")
            filenames = os.listdir(file_path)
            image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

            if self.include_subfig == True:
                file_path = os.path.join(self.path, "SciCap-Yes-Subfig-Img", "train")
                filenames = os.listdir(file_path)
                image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

            if self.train_include_val == True:
                file_path = os.path.join(self.path, "SciCap-No-Subfig-Img", "val")
                filenames = os.listdir(file_path)
                image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

                if self.include_subfig == True:
                    file_path = os.path.join(self.path, "SciCap-Yes-Subfig-Img", "val")
                    filenames = os.listdir(file_path)
                    image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])
        elif self.train == False:
            file_path = os.path.join(self.path, "SciCap-No-Subfig-Img", "test")
            filenames = os.listdir(file_path)
            image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

            if self.include_subfig == True:
                file_path = os.path.join(self.path, "SciCap-Yes-Subfig-Img", "test")
                filenames = os.listdir(file_path)
                image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

            if self.train_include_val == False:
                file_path = os.path.join(self.path, "SciCap-No-Subfig-Img", "val")
                filenames = os.listdir(file_path)
                image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

                if self.include_subfig == False:
                    file_path = os.path.join(self.path, "SciCap-Yes-Subfig-Img", "val")
                    filenames = os.listdir(file_path)
                    image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

        elif self.train == 'ALL':
            data_paths = [os.path.join(self.path, 'SciCap-No-Subfig-Img/test'),
                          os.path.join(self.path, 'SciCap-No-Subfig-Img/train'),
                          os.path.join(self.path, 'SciCap-No-Subfig-Img/val'),
                          os.path.join(self.path, 'SciCap-Yes-Subfig-Img/test'),
                          os.path.join(self.path, 'SciCap-Yes-Subfig-Img/train'),
                          os.path.join(self.path, 'SciCap-Yes-Subfig-Img/val'),
                          ]
            for data_path in data_paths:
                filenames = os.listdir(data_path)
                image_filenames.extend([os.path.join(data_path, filename) for filename in filenames])
            
        return image_filenames

    def _change_extension(self, filename, new_extension):
        base_name, _ = os.path.splitext(filename)
        return f"{base_name}.{new_extension}"


    def __len__(self):
        return len(self.image_filenames)
    

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.vis_processors:
            image = self.vis_processors['eval'](image).unsqueeze(0)

        if self.return_caption == True:
            file_name_with_extension = os.path.basename(img_path)
            file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
            dir_path = os.path.dirname(img_path)  # ファイルパスからディレクトリパスを取得
            directory_name = os.path.basename(dir_path)
            cap_path = os.path.join(self.path, 'SciCap-Caption-All', directory_name, file_name_without_extension+'.json')

            if self.use_remove == True:
                try:
                    with open(cap_path, 'r') as json_file:
                        data = json.load(json_file)
                        caption = data.get("1-lowercase-and-token-and-remove-figure-index")["caption"]  # キーが"1-lowercase-and-token-and-remove-figure-index""の値を取得
                except FileNotFoundError:
                    caption = None
                    print('except FileNotFoundError')
            
            else:
                try:
                    with open(cap_path, 'r') as json_file:
                        data = json.load(json_file)
                        caption = data.get("0-originally-extracted")  # キーが"0-originally-extracted"の値を取得
                except FileNotFoundError:
                    caption = None
                    print('except FileNotFoundError')
            
            return image, caption, img_path
        
        else:
            return _


                



if __name__ == '__main__':
    from tqdm import tqdm
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import transforms

    sicap_data_path = '/taiga/Datasets/scicap_data'

    transform=transforms.Compose([transforms.Resize((224, 224)), 
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                ])
    

    dataset = SciCapDataset(dataset_path = sicap_data_path, 
                            transform = transform,
                            train = 'ALL',               # 学習用データなのか
                            train_include_val = True,         # データにvalデータを含めるか
                            include_subfig = True,     # データにsubfigデータを含めるか
                            use_remove = True,          # lowercase-and-token-and-remove（figure:などを消去した）データを使用するか
                            retrun_caption = True,
                            vis_processors=None,
                            )
    
    print(len(dataset))
    # DataLoaderを使用してデータをバッチで読み込む
    #dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)  # バッチサイズやnum_workersは適切に設定してください
    #
    ## GPUを使用する場合
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    #for batch in tqdm(dataloader):
    #    _, _ = batch
