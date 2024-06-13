import os

from models.basenet import *
import torch
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def get_model_mme(net, num_class=13, unit_size=2048, temp=0.05):
    model_g = ResBase(net, unit_size=unit_size)
    model_c = ResClassifier_MME(num_classes=num_class, input_size=unit_size, temp=temp)
    return model_g, model_c


def save_model(model_g, model_c, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c_state_dict': model_c.state_dict(),
    }
    torch.save(save_dic, save_path)


def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c


def visualization(sourcebank, targetbank,step):
    feature = torch.cat((sourcebank.feature_bank, targetbank.feature_bank), dim=0)
    label = torch.cat((sourcebank.label_bank, targetbank.label_bank), dim=0)
    # label = torch.cat((torch.zeros_like(sourcebank.label_bank), torch.ones_like(targetbank.label_bank)), dim=0)
    label = label.cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(feature.cpu().detach().numpy())

    df_tsne = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
    df_tsne['Label'] = label
    label_mapping = {0: 'back_pack', 1: 'bike', 2: 'calculator', 3: 'headphones', 4: 'keyboard', 5: 'laptop_computer',
                     6: 'monitor', 7: 'mouse', 8: 'mug', 9: 'projector', 10: 'unknown'}
    # label_mapping = {0: 'source sample', 1: 'target sample'}
    df_tsne['Label'] = df_tsne['Label'].map(label_mapping)

    # 自定义颜色映射
    custom_palette = {
        'back_pack': 'red',
        'bike': 'blue',
        'calculator': 'green',
        'headphones': 'orange',
        'keyboard': 'purple',
        'laptop_computer': 'brown',
        'monitor': 'pink',
        'mouse': 'gray',
        'mug': 'cyan',
        'projector': 'yellow',
        'unknown': 'black'
    }
    # custom_palette = {
    #     'source sample': 'orange',
    #     'target sample': 'purple'
    # }

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Label', palette=custom_palette, data=df_tsne, legend='full',
                    marker='o', s=100)
    legend = plt.legend(title='Classes', bbox_to_anchor=(1, 1), loc='upper left', ncol=1, fontsize='12', frameon=None)

    legend.get_title().set_fontsize('20')
    for text in legend.get_texts():
        text.set_fontsize('15')
    plt.title('t-SNE Visualization of category')
    picturepath = os.path.join("record", "picture_category_2",
                               str(step) + ".jpg")

    plt.tight_layout()  # 自动调整子图参数，以使子图适应图形区域
    plt.savefig(picturepath, bbox_inches='tight')
    plt.show()

