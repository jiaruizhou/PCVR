import os
import PIL
import torch
from torch.utils.data import Dataset
from Bio import SeqIO  # please make sure Biopython is installed
from util.FCGR import fcgr
from util.tax_entry import TaxidLineage, supk_dict, phyl_dict, genus_dict, new_supk_dict, new_phyl_dict, new_genus_dict


class pretrain_Dataset(Dataset):

    def __init__(self, root_dir,kmer, save_tmp=False,save_path="./data/pretrain_data.npy"):
        self.kmer = kmer
        self.root_dir = root_dir
        self.save_path=save_path
        self.save_tmp=save_tmp

        self.classes = [d.split("_db.fa")[0] for d in os.listdir(root_dir)]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        
        for class_name in self.classes[:]:
            class_dir = os.path.join(self.root_dir, class_name + "_db.fa")
            class_idx = self.class_to_idx[class_name]

            sequences = [str(record.seq) for record in SeqIO.parse(class_dir, "fasta")]
            samples.extend([(torch.tensor(fcgr(sequence, sequence, self.kmer)).unsqueeze(0), class_idx) for sequence in sequences])
        if self.save_tmp:
            torch.save(samples, self.save_path)
        return samples

    def __len__(self):
        print("classes:{}\nsamples:{}\n".format(self.classes, len(self.samples)))
        return len(self.samples)

    def __getitem__(self, idx):
        seq_fcgr = self.samples[idx]

        return seq_fcgr


class Finetune_Dataset_All(Dataset):

    def __init__(self, args, files, kmer, phase="train",save_tmp=False,save_path="./data/pretrain_data.npy"):
        self.kmer = kmer
        self.tlineage = TaxidLineage()
        self.fcgr_features = []
        self.samples = []
        
        self.save_path=save_path
        self.save_tmp=save_tmp
        
        for phase0 in ["test", "train"]:
            file = files
            

            file = os.path.join(file, 'train.fa' if phase0 == "train" else 'test.fa')
            
            records = list(SeqIO.parse(file, "fasta"))
            ids = [str(record.id) for record in records]
            classes_names_phylum = [self.tlineage.get_ranks(id, ranks=["phylum"]) \
                                ["phylum"][1] for id in ids]
            
            classes_names_supk = [self.tlineage.get_ranks(id, ranks=["superkingdom"]) \
                                ["superkingdom"][1] for id in ids]
            classes_names_genus = [self.tlineage.get_ranks(id, ranks=["genus"]) \
                                ["genus"][1] for id in ids]
            if self.save_tmp:
                torch.save(classes_names_genus, os.join(self.save_path,"genus.npy"))
                torch.save(classes_names_supk, os.join(self.save_path,"supk.npy"))
                torch.save(classes_names_phylum, os.join(self.save_path,"phyl.npy"))


        file = os.path.join(files, 'train.fa' if phase == "train" else 'test.fa')


        records = list(SeqIO.parse(file, "fasta"))
        self.fcgr_features = self._load_fcgr(records)
        # # load the class list if exists
        
        genus_set = set(genus_dict.keys())
        genus_names = [genus_dict['unknown'] if element not in genus_set else genus_dict[element] for element in classes_names_genus]

        supk_set = set(supk_dict.keys())
        supk_names = [supk_dict['unknown'] if element not in supk_set else supk_dict[element] for element in classes_names_supk]

        phyl_set = set(phyl_dict.keys())
        phyl_names = [phyl_dict['unknown'] if element not in phyl_set else phyl_dict[element] for element in classes_names_phylum]

        self.samples.extend(\
            [(feature,(cls_name1,cls_name2,cls_name3)) \
            for _,(feature, cls_name1,cls_name2,cls_name3) in \
                enumerate(zip(self.fcgr_features, torch.tensor(supk_names)\
                    ,torch.tensor(phyl_names),torch.tensor(genus_names)))])
        


    def _load_fcgr(self, records):

        samples = []
        self.seqs = [str(record.seq) for record in records]
        
        samples.extend([torch.tensor(fcgr(sequence, sequence, self.kmer)).unsqueeze(0) for sequence in self.seqs])
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_fcgr, taxid = self.samples[idx]
        return seq_fcgr, taxid


class Pred_Dataset(Dataset):

    def __init__(self, args, files, kmer):
        self.kmer = kmer
        self.fcgr_features = []
        self.samples = []
        fcgr_path = files.split(".fa")[0] + '_{}mer.npy'.format(self.kmer)

        records = list(SeqIO.parse(files, "fasta"))
        if not os.path.exists(fcgr_path):
            self.seqs = [str(record.seq) for record in records]
            self.samples.extend([torch.tensor(fcgr(sequence, sequence, self.kmer)).unsqueeze(0) for sequence in self.seqs])
            torch.save(self.samples, fcgr_path)
        else:
            self.samples = torch.load(fcgr_path)
        print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_fcgr = self.samples[idx]
        return seq_fcgr, idx
