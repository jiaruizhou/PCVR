import numpy as np
import torch
import os
from sklearn import manifold
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
# from tsnecuda import TSNE
from pathlib import Path
import argparse
import pdb


supk_dict={'Archaea': 0, 'Bacteria': 1, 'Eukaryota': 2, 'Viruses': 3, 'unknown': 4}
phyl_dict={'Aquificae': 0, 'Tenericutes': 1, 'Actinobacteria': 2, 'Chlorophyta': 3, 'Deinococcus-Thermus': 4, 'Nematoda': 5, 'Chordata': 6, 'Basidiomycota': 7, 'Crenarchaeota': 8, 'Proteobacteria': 9, 'Verrucomicrobia': 10, 'Ascomycota': 11, 'Bacillariophyta': 12, 'Negarnaviricota': 13, 'Rhodophyta': 14, 'Gemmatimonadetes': 15, 'Peploviricota': 16, 'Uroviricota': 17, 'Kitrinoviricota': 18, 'Nitrospirae': 19, 'Lentisphaerae': 20, 'Platyhelminthes': 21, 'Arthropoda': 22, 'Streptophyta': 23, 'Thermotogae': 24, 'Pisuviricota': 25, 'Euryarchaeota': 26, 'Mollusca': 27, 'Euglenozoa': 28, 'Planctomycetes': 29, 'Evosea': 30, 'Artverviricota': 31, 'Chlorobi': 32, 'Firmicutes': 33, 'Chloroflexi': 34, 'Candidatus Thermoplasmatota': 35, 'Chlamydiae': 36, 'Cyanobacteria': 37, 'Bacteroidetes': 38, 'Thaumarchaeota': 39, 'Apicomplexa': 40, 'Fusobacteria': 41, 'unknown': 42, 'Spirochaetes': 43}
genus_dict={'Microcaecilia': 0, 'Roseiflexus': 1, 'Schistosoma': 2, 'Euzebya': 3, 'Colletotrichum': 4, 'Gallus': 5, 'Strigops': 6, 'Methanosarcina': 7, 'Nitrospira': 8, 'Botrytis': 9, 'Asparagus': 10, 'Sparus': 11, 'Fervidicoccus': 12, 'Dictyostelium': 13, 'Bdellovibrio': 14, 'Oryctolagus': 15, 'Takifugu': 16, 'Punica': 17, 'Cynara': 18, 'Aspergillus': 19, 'Olea': 20, 'Rhopalosiphum': 21, 'Esox': 22, 'Ostreococcus': 23, 'Brassica': 24, 'Echeneis': 25, 'Aythya': 26, 'Egicoccus': 27, 'Astyanax': 28, 'Nitrosopumilus': 29, 'Pomacea': 30, 'Daucus': 31, 'Pyricularia': 32, 'Vitis': 33, 'Trichoplusia': 34, 'Elaeis': 35, 'Plasmodium': 36, 'Mus': 37, 'Actinopolyspora': 38, 'Ciona': 39, 'Theileria': 40, 'Lepisosteus': 41, 'Methanopyrus': 42, 'Helianthus': 43, 'Cyanidioschyzon': 44, 'Sarcophilus': 45, 'Legionella': 46, 'Gadus': 47, 'unknown': 48, 'Archaeoglobus': 49, 'Drosophila': 50, 'Rubrobacter': 51, 'Fusarium': 52, 'Leptospira': 53, 'Spodoptera': 54, 'Chanos': 55, 'Limnochorda': 56, 'Methanobacterium': 57, 'Candidatus Promineofilum': 58, 'Gopherus': 59, 'Stanieria': 60, 'Solanum': 61, 'Calypte': 62, 'Thermus': 63, 'Beta': 64, 'Sedimentisphaera': 65, 'Rudivirus': 66, 'Phyllostomus': 67, 'Quercus': 68, 'Neisseria': 69, 'Akkermansia': 70, 'Sesamum': 71, 'Cucumis': 72, 'Ictalurus': 73, 'Ktedonosporobacter': 74, 'Anas': 75, 'Citrus': 76, 'Cynoglossus': 77, 'Aquila': 78, 'Oryzias': 79, 'Brachyspira': 80, 'Papaver': 81, 'Apis': 82, 'Methanocella': 83, 'Leishmania': 84, 'Cercospora': 85, 'Egibacter': 86, 'Cryptococcus': 87, 'Equus': 88, 'Salinisphaera': 89, 'Streptomyces': 90, 'Gemmata': 91, 'Octopus': 92, 'Sporisorium': 93, 'Pseudomonas': 94, 'Deinococcus': 95, 'Thermococcus': 96, 'Gossypium': 97, 'Betta': 98, 'Aeromonas': 99, 'Thermogutta': 100, 'Frankia': 101, 'Thalassiosira': 102, 'Crassostrea': 103, 'Acyrthosiphon': 104, 'Denticeps': 105, 'Chlamydia': 106, 'Cottoperca': 107, 'Acidithiobacillus': 108, 'Ornithorhynchus': 109, 'Cucurbita': 110, 'Podarcis': 111, 'Malassezia': 112, 'Xiphophorus': 113, 'Perca': 114, 'Actinomyces': 115, 'Modestobacter': 116, 'Synechococcus': 117, 'Musa': 118, 'Oncorhynchus': 119, 'Methanobrevibacter': 120, 'Pyrobaculum': 121, 'Vibrio': 122, 'Tribolium': 123, 'Desulfovibrio': 124, 'Scleropages': 125, 'Ooceraea': 126, 'Sphaeramia': 127, 'Nymphaea': 128, 'Zymoseptoria': 129, 'Acidilobus': 130, 'Candidatus Kuenenia': 131, 'Chrysemys': 132, 'Phaeodactylum': 133, 'Salarias': 134, 'Nitrososphaera': 135, 'Coffea': 136, 'Clupea': 137, 'Bifidobacterium': 138, 'Ustilago': 139, 'Physcomitrium': 140, 'Populus': 141, 'Gouania': 142, 'Carassius': 143, 'Rhinatrema': 144, 'Mariprofundus': 145, 'Monodelphis': 146, 'Candidatus Nitrosocaldus': 147, 'Prosthecochloris': 148, 'Xenopus': 149, 'Erpetoichthys': 150, 'Methanocaldococcus': 151, 'Bradymonas': 152, 'Caenorhabditis': 153, 'Manihot': 154, 'Myripristis': 155}

new_supk_dict = {v : k for k, v in supk_dict.items()}
new_phyl_dict = {v : k for k, v in phyl_dict.items()}
new_genus_dict = {v : k for k, v in genus_dict.items()}


def get_args_parser():
    parser = argparse.ArgumentParser('plot_tsne', add_help=False)
    parser.add_argument('--model', default='bertax', type=str)
    parser.add_argument('--data', default='final', type=str)
    parser.add_argument('--rank', default='phyl', type=str)
    
    parser.add_argument('--pretrained', default=0, type=int)
    parser.add_argument("--title",default="",type=str)
    return parser


def main(args):
    rank_dict = {"supk": 0, "phyl": 1, "genus": 2}
    print("Load the data......")
    if args.pretrained:
        if args.model == "bertax":
            data = torch.load('/data5/zhoujr/bertax_training/outputs/pretrained/{}/q_fine.npy'.format(args.data))
            labels = torch.load('/data5/zhoujr/mae/outputs/output_dir_retrieval/pretrained/{}/q_y_glb.npy'.format(args.data))
        else:
            data = torch.load('/data5/zhoujr/mae/outputs/output_dir_retrieval/pretrained/{}/q_glb.npy'.format(args.data))

            labels = torch.load('/data5/zhoujr/mae/outputs/output_dir_retrieval/pretrained/{}/q_y_glb.npy'.format(args.data))

    
    else:
        if args.model == "bertax":
            data = torch.load('/data5/zhoujr/bertax_training/outputs/{}/q.npy'.format(args.data))
            labels = torch.load('/data5/zhoujr/mae/outputs/output_dir_retrieval/{}/q_y_glb.npy'.format(args.data))
        else:
            data = torch.load('/data5/zhoujr/mae/outputs/output_dir_retrieval/{}/q_glb.npy'.format(args.data))

            labels = torch.load('/data5/zhoujr/mae/outputs/output_dir_retrieval/{}/q_y_glb.npy'.format(args.data))
    data = torch.tensor(data).squeeze(1)
    data = F.normalize(data, dim=-1)
    print("Data successfully loaded")
    torch.manual_seed(42)
    if args.rank != "supk":
        classes = list(set(list(labels[:, rank_dict[args.rank]].numpy())))[::]
        print(classes)
        lb = labels[:, rank_dict[args.rank]]
        sample_ind = []
        for class_id in classes:
            sample_ind.extend(np.where(lb == class_id)[0])

    else:
        sample_ind = torch.randperm(len(data))[:100]
    print(len(sample_ind))
    sample_data = data[sample_ind]  #
    sample_y = labels[:, rank_dict[args.rank]][sample_ind]
    print("Inferencing......")
    plt.figure(figsize=(8, 8))

    print("Get the tsne......")
    t_tsne = manifold.TSNE(n_components=2, random_state=0)
    X_tsne = t_tsne.fit_transform(sample_data.cpu().detach().numpy())

    print("Ploting the scatter")
    train_x_min, train_x_max = X_tsne.min(0), X_tsne.max(0)
    train_X_norm = (X_tsne - train_x_min) / (train_x_max - train_x_min)
    import seaborn as sns
    

    classes = list(set(list(labels[:, rank_dict[args.rank]].numpy())))[:2]
    print(classes)
    lb = labels[:, rank_dict["phyl"]]
    sample_ind = []
    i=0
    for class_id in classes:
        l=len(np.where(lb == class_id)[0])

        plt.scatter(train_X_norm[i:i+l, 0], train_X_norm[i:i+l, 1], c=sample_y[i:i+l], s=2, cmap='Paired',marker="+",label=new_phyl_dict[class_id])
        i+=l
    plt.legend(loc='best')    
    # for i in range(int(data_bertax.shape[0])):
    #     plt.text(train_X_norm[i, 0], train_X_norm[i, 1], '.', color="blue", fontdict={'weight': 'bold', 'size': 9})
    # custom_palette = sns.color_palette("husl", 44)
    # plt.scatter(train_X_norm[:, 0], train_X_norm[:, 1], c=sample_y, s=2, cmap='husl',marker="+")
    plt.xticks(fontsize=18)  # 设置字体大小为12
    plt.title("{}".format(args.title),fontsize=22)
    # 调整 y 轴刻度字体
    plt.yticks(fontsize=18)  # 设置字体大小为12
    outFile = os.path.join("tsne/tsne_{}_{}_{}_{}xxx.pdf".format(args.rank, args.data, args.model,args.pretrained))
    # plt.title("{} TSNE on {} with {} Pre-trained Model".format(args.rank, args.data, args.model))
    plt.show()
    plt.savefig(outFile, bbox_inches='tight')
    plt.close()
    print("Program finished!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser('plot_tsne', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)
