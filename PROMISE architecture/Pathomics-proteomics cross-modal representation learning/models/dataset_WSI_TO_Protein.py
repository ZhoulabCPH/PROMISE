import os
import torchvision
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import h5py
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageFilter
is_amp = True
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from augmentation import *
import random
from torch.utils.data import random_split

random.seed(42)


def adjust_matrix(matrix, target_rows=3000):
    # 获取当前矩阵的行数
    current_rows = matrix.shape[0]

    if current_rows < target_rows:
        # 行数不足，生成随机数据并补充
        additional_rows = target_rows - current_rows
        # 生成 additional_rows 个随机的 1x1024 向量
        random_data = torch.randint(0, 2, (additional_rows, matrix.shape[1])).float()
        # 将原矩阵与生成的随机矩阵拼接
        matrix = torch.cat((matrix, random_data), dim=0)

    elif current_rows > target_rows:
        # 行数超过，随机抽取 target_rows 行
        indices = torch.randint(0, current_rows, (target_rows,))
        matrix = matrix[indices]

    # 返回处理后的矩阵
    return matrix

def make_big_model_feature_val(arg):

    CHCAMS_External = torch.load(arg.CHCAMS_External_Image)
    try:
        CHCAMS_External_Patients = CHCAMS_External['Patient_name']
        if CHCAMS_External_Patients[0] == 'BLOCKS':
            CHCAMS_External['Patient_name'] = [name.split('_')[0] for name in CHCAMS_External['Patch_name']]
            CHCAMS_External_Patients = CHCAMS_External['Patient_name']
    except KeyError:
        CHCAMS_External['Patient_name']=[name.split('_')[0] for name in CHCAMS_External['patch_file_path']]
        CHCAMS_External['Patch_name'] = CHCAMS_External['patch_file_path']
        CHCAMS_External_Patients=CHCAMS_External['Patient_name']
    CHCAMS_External_Clincial = pd.read_csv(arg.CHCAMS_External_Clincial)
    CHCAMS_External_patients=np.unique(CHCAMS_External_Patients)
    CHCAMS_External_indices = {x: [i for i, val in enumerate(CHCAMS_External_Patients) if val == x] for x in CHCAMS_External_patients}


    HMUCH = torch.load(arg.HMUCH_External_Image)
    try:
        HMUCH_Patients = HMUCH['Patient_name']
        if HMUCH_Patients[0] == 'BLOCKS':
            HMUCH['Patient_name'] = [name.split('_')[0] for name in HMUCH['Patch_name']]
            HMUCH_Patients = HMUCH['Patient_name']
    except KeyError:
        HMUCH['Patient_name']=[name.split('_')[0] for name in HMUCH['patch_file_path']]
        HMUCH['Patch_name'] = HMUCH['patch_file_path']
        HMUCH_Patients=HMUCH['Patient_name']
    HMUCH_Clincial = pd.read_csv(arg.HMUCH_External_Clincial)
    HMUCH_patients=np.unique(HMUCH_Patients)
    HMUCH_indices = {x: [i for i, val in enumerate(HMUCH_Patients) if val == x] for x in HMUCH_patients}


    #
    TMUGH = torch.load(arg.TMUGH_External_Image)
    try:
        TMUGH_Patients = TMUGH['Patient_name']
        if TMUGH_Patients[0] == 'BLOCKS':
            TMUGH['Patient_name'] = [name.split('_')[0] for name in TMUGH['Patch_name']]
            TMUGH_Patients = TMUGH['Patient_name']
    except KeyError:
        TMUGH['Patient_name']=[name.split('_')[0] for name in TMUGH['patch_file_path']]
        TMUGH['Patch_name'] = TMUGH['patch_file_path']
        TMUGH_Patients = TMUGH['Patient_name']
    TMUGH_Clincial = pd.read_csv(arg.TMUGH_External_Clincial)
    TMUGH_patients=np.unique(TMUGH_Patients)
    TMUGH_indices = {x: [i for i, val in enumerate(TMUGH_Patients) if val == x] for x in TMUGH_patients}



    # CHCAMS_PRE_Protein = pd.read_csv(arg.CHCAMS_PreProtein).drop(columns=['Unnamed: 0', 'OS', 'OSState', 'DFS', 'DFSState'])
    # TMUGH_PRE_Protein = pd.read_csv(arg.TMUGH_PreProtein).drop(
    #     columns=['Unnamed: 0', 'OS', 'OSState', 'DFS', 'DFSState'])
    # HMUCH_PRE_Protein = pd.read_csv(arg.HMUCH_PreProtein).drop(
    #     columns=['Unnamed: 0', 'OS', 'OSState', 'DFS', 'DFSState'])



    Cohorts={
        'HMUCH_indices':HMUCH_indices,
        'HMUCH_feature': HMUCH['feature'],
        'HMUCH_patch_name': HMUCH['Patch_name'],
        'HMUCH_clincial': HMUCH_Clincial,
        'HMUCH_dim': HMUCH['feature'][0].shape[0],
        # 'HMUCH_PRE_Protein': HMUCH_PRE_Protein,

        'TMUGH_indices':TMUGH_indices,
        'TMUGH_feature': TMUGH['feature'],
        'TMUGH_patch_name': TMUGH['Patch_name'],
        'TMUGH_dim': TMUGH['feature'][0].shape[0],
        'TMUGH_clincial': TMUGH_Clincial,
        # 'TMUGH_PRE_Protein': TMUGH_PRE_Protein,

        'CHCAMS_External_indices': CHCAMS_External_indices,
        'CHCAMS_External_feature': CHCAMS_External['feature'],
        'CHCAMS_External_patch_name': CHCAMS_External['Patch_name'],
        'CHCAMS_External_clincial': CHCAMS_External_Clincial,
        'CHCAMS_External_dim': CHCAMS_External['feature'][0].shape[0],
        # 'CHCAMS_PRE_Protein': CHCAMS_PRE_Protein,

        }
    return Cohorts

def make_big_model_feature_Fundation(arg):
    #CHCAMS_Discovery_CONCH_5X
    #CHCAMS_Discovery_Virchow_10X
    #CHCAMS_Discovery_CTransPath_5X
    #CHCAMS_Discovery_MUSK_5X
    #fundation_path_feature
    Discovery=torch.load(arg.fundation_path_feature)
    try:
        Patients = Discovery['Patient_name']
        if Patients[0]=='BLOCKS':
            Discovery['Patient_name']=[name.split('_')[0] for name in Discovery['Patch_name']]
            Patients = Discovery['Patient_name']
    except KeyError:
        Discovery['Patient_name']=[name.split('_')[0] for name in Discovery['patch_file_path']]
        Discovery['Patch_name'] = Discovery['patch_file_path']
        Patients = Discovery['Patient_name']

    #划分患者训练集测试集
    # 读取蛋白质数据集
    CHCAMS_Protein=pd.read_csv(arg.CHCAMS_Protein)
    CHCAMS_Protein['Unnamed: 0']=np.array(CHCAMS_Protein['Unnamed: 0'].values,dtype=np.str_)

    train_patients=np.array(pd.read_csv(arg.CHCAMS_Train_Clincial)['PatientID'].values,dtype=np.str_)
    test_patients = np.array(pd.read_csv(arg.CHCAMS_Test_Clincial)['PatientID'].values,dtype=np.str_)

    CHCAMS_Train_Protein = CHCAMS_Protein[CHCAMS_Protein['Unnamed: 0'].isin(train_patients)]
    CHCAMS_Test_Protein = CHCAMS_Protein[CHCAMS_Protein['Unnamed: 0'].isin(test_patients)]

    CHCAMS_Train_PRE_Protein = pd.read_csv(arg.Train_PreProtein).drop(columns=['Unnamed: 0', 'OS', 'OSState', 'DFS', 'DFSState'])
    CHCAMS_Test_PRE_Protein = pd.read_csv(arg.Test_PreProtein).drop(
        columns=['Unnamed: 0', 'OS', 'OSState', 'DFS', 'DFSState'])


    train_indices = {x: [i for i, val in enumerate(Patients) if val == x] for x in train_patients}
    test_indices = {x: [i for i, val in enumerate(Patients) if val == x] for x in test_patients}

    #添加一些临床信息
    CHCAMS_Discovery_Clincial=pd.read_csv(arg.CHCAMS_Discovery_Clincial)
    #
    Cohorts={'train_indices':train_indices,
             'test_indices':test_indices,
             'feature':torch.tensor(Discovery['feature']),
             'patch_name':Discovery['Patch_name'],
             'clincial':CHCAMS_Discovery_Clincial,
             'dim':Discovery['feature'][0].shape[0],
             "train_Protein":CHCAMS_Train_Protein,
             "test_Protein": CHCAMS_Test_Protein,
             "Train_PreProtein":CHCAMS_Train_PRE_Protein,
            "Test_PreProtein":CHCAMS_Test_PRE_Protein
             }


    return Cohorts



def make_big_model_feature_Train(arg):

    Discovery=torch.load(arg.CHCAMS_Discovery_Image)
    try:
        Patients = Discovery['Patient_name']
        if Patients[0]=='BLOCKS':
            Discovery['Patient_name']=[name.split('_')[0] for name in Discovery['Patch_name']]
            Patients = Discovery['Patient_name']
    except KeyError:
        Discovery['Patient_name']=[name.split('_')[0] for name in Discovery['patch_file_path']]
        Discovery['Patch_name'] = Discovery['patch_file_path']
        Patients = Discovery['Patient_name']

    #划分患者训练集测试集
    # 读取蛋白质数据集
    CHCAMS_Protein=pd.read_csv(arg.CHCAMS_Protein)
    CHCAMS_Protein['Unnamed: 0']=np.array(CHCAMS_Protein['Unnamed: 0'].values,dtype=np.str_)

    train_patients=np.array(pd.read_csv(arg.CHCAMS_Train_Clincial)['PatientID'].values,dtype=np.str_)
    test_patients = np.array(pd.read_csv(arg.CHCAMS_Test_Clincial)['PatientID'].values,dtype=np.str_)

    CHCAMS_Train_Protein = CHCAMS_Protein[CHCAMS_Protein['Unnamed: 0'].isin(train_patients)]
    CHCAMS_Test_Protein = CHCAMS_Protein[CHCAMS_Protein['Unnamed: 0'].isin(test_patients)]

    train_indices = {x: [i for i, val in enumerate(Patients) if val == x] for x in train_patients}
    test_indices = {x: [i for i, val in enumerate(Patients) if val == x] for x in test_patients}

    #添加一些临床信息
    CHCAMS_Discovery_Clincial=pd.read_csv(arg.CHCAMS_Discovery_Clincial)
    #
    Cohorts={'train_indices':train_indices,
             'test_indices':test_indices,
             'feature':Discovery['feature'],
             'patch_name':Discovery['Patch_name'],
             'clincial':CHCAMS_Discovery_Clincial,
             'dim':Discovery['feature'][0].shape[0],
             "train_Protein":CHCAMS_Train_Protein,
             "test_Protein": CHCAMS_Test_Protein
             }


    return Cohorts


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomRotation(
                degrees=90,
                resample=False,
                expand=False,
                center=None,
                fill=255,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.4),
            Solarization(p=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            # transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomRotation(
                degrees=90,
                resample=False,
                expand=False,
                center=None,
                fill=255,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

class Transform_:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        return y1,y1

def get_percent_subset(dataset,ratio):
    DF_Length = len(list(dataset.items()))
    Sizes = int(DF_Length * ratio)
    random_numbers = random.sample(range(0, DF_Length), Sizes)
    selected_elements = list([list(dataset.items())[i] for i in random_numbers])
    return selected_elements


class SCLCDataset(Dataset):
    def __init__(self,df,feature,name,clincials,Protein,args,ratio=1):
        self.df=get_percent_subset(df,ratio)
        self.args=args
        self.name=np.array(name)
        self.clincial=clincials
        self.Protein=Protein
        self.Data=feature
        self.length = len(self.df)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        # patient_feature_id=self.df[index]
        patient_name, patient_patch_id = self.df[index]
        # datasets=np.array(self.Data[patient_patch_id])
        #病理特征
        mydataframes_ = adjust_matrix(self.Data[patient_patch_id], self.args.Patch_number)
        #临床数据
        index_clincial=self.clincial[self.clincial['PatientID']==np.int64(patient_name)]
        OS=index_clincial['OS'].values[0]
        DFS=index_clincial['DFS'].values[0]
        OSState=index_clincial['OSState'].values[0]
        DFSState=index_clincial['DFSState'].values[0]
        #蛋白质数据

        index_Protein=torch.tensor(np.array(self.Protein[self.Protein['Unnamed: 0']==np.str_(patient_name)].iloc[:,1:].values,dtype=np.float32))


        r = {}
        r['index'] = index
        r['feature'] = mydataframes_
        r['patch_name']=self.name[patient_patch_id]
        r['patient_name'] = patient_name

        r['OS']=torch.tensor(OS)
        r['DFS'] = torch.tensor(DFS)
        r['OSState'] = torch.tensor(OSState)
        r['DFSState'] = torch.tensor(DFSState)

        r['Protein']=index_Protein
        return r

class SCLCDataset_Val(Dataset):
    def __init__(self,df,feature,name,clincials,Protein,args):
        self.df = list(df.items())
        self.args=args
        self.name=np.array(name)
        self.clincial=clincials
        self.Protein=Protein
        # Xs=np.array(X)
        # indexs=df['Index'].values
        self.Data=feature
        self.length = len(self.df)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        # patient_feature_id=self.df[index]
        patient_name, patient_patch_id = self.df[index]
        # datasets=np.array(self.Data[patient_patch_id])
        #病理特征
        mydataframes_=self.Data[patient_patch_id]
        # mydataframes_ = adjust_matrix(self.Data[patient_patch_id], self.args.Patch_number)
        #临床数据
        index_clincial=self.clincial[self.clincial['PatientID']==np.int64(patient_name)]
        OS=index_clincial['OS'].values[0]
        DFS=index_clincial['DFS'].values[0]
        OSState=index_clincial['OSState'].values[0]
        DFSState=index_clincial['DFSState'].values[0]
        #蛋白质数据
        try:
            index_Protein=torch.tensor(np.array(self.Protein[self.Protein['Unnamed: 0']==np.str_(patient_name)].iloc[:,1:].values,dtype=np.float32))
        except TypeError:
            index_Protein=None
        r = {}
        r['index'] = index
        r['feature'] = mydataframes_
        r['patch_name']=self.name[patient_patch_id]
        r['patient_name'] = patient_name

        r['OS']=torch.tensor(OS)
        r['DFS'] = torch.tensor(DFS)
        r['OSState'] = torch.tensor(OSState)
        r['DFSState'] = torch.tensor(DFSState)
        r['Protein']=index_Protein

        return r

tensor_list = [
     'name1','feature','OS','DFS','OSState','DFSState','Protein'
]





def image_to_tensor(image, mode='bgr'):  # image mode
    if mode == 'bgr':
        image = image[:, :, ::-1]
    x = image
    x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x


def tensor_to_image(x, mode='bgr'):
    image = x.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    if mode == 'bgr':
        image = image[:, :, ::-1]
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    return image

tensor_list = ['name1','feature','OS','DFS','OSState','DFSState','Protein']


def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_list:
            try:
                v = torch.stack(v)
            except TypeError:
                v=None
        d[k] = v
    # d['organ'] = d['organ'].reshape(-1)
    return d

