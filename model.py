import io
import pandas as pd
from pathlib import Path
import pickle
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# Класс для работы с изображениями
class ImagesDataset(Dataset):
    def __init__(self, df, images_dir, transform):
        self.df = df
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['Id']
        label = self.df.iloc[idx]['Category']
        image = Image.open(self.images_dir / path).convert('RGB')
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.df)


# Класс для работы с моделью в формате .pkl без доступа к GPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


# Класс модели
class Model:
    def __init__(self):
        self.model = None
        self.test_data = None
        self.test_dataset = None
        self.test_loader = None

    # метод принимает на вход полный путь до модели и загружает её
    def load_model(self, path_to_model):
        with open(path_to_model, 'rb') as file:
            if torch.cuda.is_available():
                self.model = pickle.load(file)
            else:
                self.model = CPU_Unpickler(file).load()
        self.model.eval()

    # метод принимает на вход полный путь до изображения, преобразует его до нужного размера, а также добавляет в датасет
    def get_image(self, path_to_image):
        self.test_data = pd.DataFrame({'Id': path_to_image, 'Category': [0]})
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((229, 229)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        reversed_path_to_image = path_to_image[:: -1]
        if path_to_image.find('/') != -1:
            index = reversed_path_to_image.find('/')
        else:
            index = reversed_path_to_image.find('\\')
        project_dir = path_to_image[: len(path_to_image) - index]
        self.test_dataset = ImagesDataset(df=self.test_data, images_dir=project_dir, transform=test_transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # метод принимает на вход полный путь до изображения пролежня, а возвращает число от 1 до 4 - его стадию
    def predict(self, path_to_image):
        self.get_image(path_to_image)
        for inputs, _ in self.test_loader:
            inputs = inputs.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                outputs = self.model(inputs)
                label = outputs.cpu().numpy().argmax(axis=1)
        return label[0] + 1
