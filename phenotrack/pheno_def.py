#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2022 Zachary Sethna
"""

from __future__ import print_function, division
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tcr_seq_and_clone_definitions import TcellClone, CloneDefinition
from tcr_utils import ClonesAndData, ClonesAndCounts, TCRClonePvalue

#%
class CategoryIndexer(object):
    def __init__(self, x = []):
        self.x = list(x) #account for varying input types
        self.x_inds = {x_i: i for i, x_i in enumerate(self.x)}
        
    def __repr__(self):
        return "CategoryIndexer(%s)" %(', '.join(self.x))
    
    def __len__(self):
        return len(self.x)
    
    def __add__(self, x):
        out_indexer = type(self)()
        out_indexer.x = self.x.copy()
        out_indexer.x_inds = self.x_inds.copy()
        if type(x) == str:
            x = [x]
        for x_i in [x_i for x_i in x if x_i not in self.x_inds]:
            out_indexer.x.append(x_i)
            out_indexer.x_inds[x_i] = len(out_indexer.x) - 1
        return out_indexer
        
    def __iadd__(self, x):
        #Faster version that does not copy class attributes
        if type(x) == str:
            x = [x]
        for x_i in [x_i for x_i in x if x_i not in self.x_inds]:
            self.x.append(x_i)
            self.x_inds[x_i] = len(self.x) - 1
        return self

    def __getitem__(self, x_i):
        try:
            return self.x_inds[x_i]
        except KeyError:
            try:
                self.x[x_i]
                return x_i
            except:
                raise KeyError

    def __delitem__(self, x):
        if type(x) == str:
            x_ind = self[x]
            self.x.__delitem__(x_ind)
            self.x_inds.__delitem__(x)
            for x_i in self.x[x_ind:]:
                self.x_inds[x_i] -= 1
        elif type(x) == int:
            self.__delitem__(self[x])
        elif type(x) == slice:
            self.__delitem__(self.x[x])
        else:
            for x_i in x:
                self.__delitem__(self[x_i])
    
    def intersection(self, y):
        return set(self).intersection(y)
    
    def union(self, y):
        return set(self).union(y)
        
    def __iter__(self):
        return iter(self.x)


class Phenotypes(CategoryIndexer):

    def __init__(self, phenotypes = []):
        CategoryIndexer.__init__(self, x = phenotypes)
        self.phenotypes = self.x 
        self.phenotypes_inds = self.x_inds
        
    def __repr__(self):
        return "Phenotypes(%s)" %(', '.join(self.phenotypes))
    

class Genes(CategoryIndexer):
    def __init__(self, genes = []):
        CategoryIndexer.__init__(self, x = genes)
        self.genes = self.x #account for varying input types
        self.gene_inds = self.x_inds
        
    def __repr__(self):
        return "Genes(%s)" %(', '.join(self.genes))
    
class GeneExpr(object):
    def __init__(self, genes = Genes([]), gene_expr_vec = None, **kwargs):
        self.genes = genes
        if gene_expr_vec is not None:
            self.gene_expr_vec = np.array(gene_expr_vec)
        else:
            self.gene_expr_vec = np.zeros(len(self.genes))

    def __repr__(self):
        return "GenesExpr(%i total genes, %i expressed genes)" %(len(self.gene_expr_vec), np.sum(self.gene_expr_vec > 0))
    
    def __len__(self):
        return len(self.genes)

    def __getitem__(self, gene):
        return self.gene_expr_vec[self.genes[gene]]
            
    def __iter__(self):
        return iter(self.genes)
    
    def items(self):
        return zip(self.genes.genes, self.gene_expr_vec)

    def keys(self):
        return list(self.genes)

    def values(self):
        return list(self.gene_expr_vec)

    def get(self, gene, default = 0):
        try:
            return self[gene]
        except:
            return default
        
#%%
class PhenotypesAndCounts(object):

    def __init__(self, phenotypes = Phenotypes([]), counts = [], **kwargs):
        if 'phenotypes_and_counts' in kwargs:
            phenotypes_and_counts = kwargs['phenotypes_and_counts']
            if len(phenotypes) == 0:
                phenotypes = list(kwargs.get('phenotypes', phenotypes_and_counts.keys()))
            counts = [phenotypes_and_counts.get(p, 0) for p in phenotypes]
        self.phenotypes = phenotypes
        if len(counts) == 0 and len(phenotypes) > 0:
            counts = [0 for _ in phenotypes]
        self.counts = counts
            
    def __repr__(self):
        return "PhenotypesAndCounts(%s)"%(', '.join(['%s: %s'%(pheno, count) for pheno, count in zip(self.phenotypes, self.counts)]))
    
    def __len__(self):
        return len(self.phenotypes)
    
    def __add__(self, phenotypes_and_counts):
        if type(phenotypes_and_counts) == Tcell:
            return PhenotypesAndCounts(phenotypes = self.phenotypes, counts = [self[phenotype] + phenotypes_and_counts.phenotypes_and_counts.get(phenotype, 0) for phenotype in self.phenotypes])
        else:
            return PhenotypesAndCounts(phenotypes = self.phenotypes, counts = [self[phenotype] + phenotypes_and_counts.get(phenotype, 0) for phenotype in self.phenotypes])
    
    def __iadd__(self, phenotypes_and_counts):
        for phenotype in self.phenotypes.intersection(phenotypes_and_counts):
            self[phenotype] += phenotypes_and_counts[phenotype]
        return self

    def __getitem__(self, phenotype):
        return self.counts[self.phenotypes[phenotype]]

    def __setitem__(self, phenotype, count):
        self.counts[self.phenotypes[phenotype]] = count

    def __iter__(self):
        return iter(self.phenotypes)

    def items(self):
        return zip(self.phenotypes, self.counts)

    def keys(self):
        return list(self.phenotypes)

    def values(self):
        return self.counts

    def get(self, phenotype, default = 0):
        try:
            return self[phenotype]
        except:
            return default
        #return self.phenotypes_and_counts.get(phenotype, default)

#%%
class Tcell(TcellClone):
    def __init__(self, **kwargs):
        TcellClone.__init__(self, **kwargs)
        self.barcode = kwargs.get('barcode', None)
        self.phenotypes_and_counts = PhenotypesAndCounts(**kwargs)
        self.gene_expr = GeneExpr(**kwargs)
        if len(self.phenotypes_and_counts) == 0:
            self.count = kwargs.get('count', 1)
            self.phenotype = None
        else:
            self.phenotype = sorted(self.phenotypes_and_counts.phenotypes, key = self.phenotypes_and_counts.__getitem__, reverse = True)[0]
            self.count = kwargs.get('count', sum(self.phenotypes_and_counts.values()))

    def __repr__(self):
        return 'Tcell(clone: %s, phenotypes: %s)'%(self.clone_rep(), str(self.phenotypes_and_counts))

#%
class ClonesAndPhenotypes(object):
    """Base class for joint distribution between clones and phenotypes.
    """

    def __init__(self, clones_and_phenos = {}, phenotypes = Phenotypes([])):
        
        if len(phenotypes) == 0 and len(clones_and_phenos) > 0: 
            phenotypes = Phenotypes((clones_and_phenos.values())[0].keys())
        self.phenotypes = phenotypes
        self.clones_and_phenos = {clone: PhenotypesAndCounts(phenotypes_and_counts = phenotypes_and_counts, phenotypes = self.phenotypes) for clone, phenotypes_and_counts in clones_and_phenos.items()}
        self.norm = sum(sum(self.clones_and_phenos.values(), PhenotypesAndCounts(phenotypes = phenotypes)).values())
    
    def __repr__(self):
        if len(self) > 100:
            return 'ClonesAndPhenotypes({%s%s%s})'%(', '.join(['%s: %s'%(clone,str(self.clones_and_phenos[clone])) for clone in self.clones[:20]]), '\n...\n', ', '.join(['%s: %s'%(clone,str(self.clones_and_phenos[clone])) for clone in self.clones[-20:]]))
        else:
            return 'ClonesAndPhenotypes(%s)'%(self.clones_and_phenos.__repr__())
    
    def __len__(self):
        return len(self.clones_and_phenos)

    def __getitem__(self, clone_or_pheno):
        if clone_or_pheno in self.phenotypes:
            return ClonesAndCounts({clone: pheno_counts[clone_or_pheno] for clone, pheno_counts in self.clones_and_phenos.items()})
        else:
            return self.clones_and_phenos.get(clone_or_pheno, PhenotypesAndCounts(phenotypes = self.phenotypes))

    def __setitem__(self, clone, pheno_counts):
        self.norm += sum(pheno_counts.values()) - sum(self.clones_and_phenos.get(clone, {'': 0}).values())
        self.clones_and_phenos[clone] = pheno_counts

    def __delitem__(self, clone):
        self.norm -= sum(self.clones_and_phenos.get(clone, {'': 0}).values())
        del self.clones_and_phenos[clone]

    def __iter__(self):
        return iter(self.clones_and_phenos)

    def items(self):
        return self.clones_and_phenos.items()

    def keys(self):
        return self.clones_and_phenos.keys()

    def values(self):
        return self.clones_and_phenos.values()

    def get(self, clone_or_pheno, default = None):
        if clone_or_pheno in self.phenotypes:
            return ClonesAndCounts({clone: pheno_counts[self.phenotypes_dict[clone_or_pheno]] for clone, pheno_counts in self.clones_and_phenos.items()})
        else:
            return self.clones_and_phenos.get(clone_or_pheno, default)

    def get_phenotype_counts(self, clone, phenotypes):
        c_counts = self.clones_and_phenos[clone]
        return np.array([c_counts[phenotype] for phenotype in phenotypes])
    
    def clones(self):
        return list(self.clones_and_phenos.keys())

    def clone_intersection(self, clones):
        try:
            clones = set(clones.keys())
        except AttributeError:
            clones = set(clones)

        return list(clones.intersection(self.clones_and_phenos))

    def clone_union(self, clones):
        try:
            clones = set(clones.keys())
        except AttributeError:
            clones = set(clones)

        return list(clones.union(self.clones_and_phenos))
#%
class TcellPhenotypeRepertoire(CloneDefinition, ClonesAndPhenotypes, TCRClonePvalue):

    def __init__(self, **kwargs):

        CloneDefinition.__init__(self, **kwargs)
        ClonesAndPhenotypes.__init__(self, **{kw: kw_val for kw, kw_val in kwargs.items() if kw in ['clones_and_phenos', 'phenotypes']})
        TCRClonePvalue.__init__(self)

        self.name = ''
        
        self.filenames = []
        self.cell_list = []
        self.pvalues = {}
        
        for kw, kw_val in kwargs.items():
            if kw in self.__dict__:
                self.__dict__[kw] = kw_val
            elif 'filename' in kw:
                if 'adaptive' in kw:
                    self.load_adaptive_file(kw_val)
                elif '10x_clonotype' in kw:
                    self.load_10x_clonotypes(kw_val)
        
        if len(self.cell_list) > 0:
            self._set_consistency()
    
    def __repr__(self):
        return 'TcellPhenotypeRepertoire'
    
    def _set_consistency(self):
        self.clones_and_phenos = self.get_clones_and_phenos()
        self.clones = self.clones_and_phenos.clones()
        self.joint_distribution = self.get_clones_and_phenos_joint_distribution()
        self.norm = len(self.cell_list)
    
    def get_clones_and_phenos(self, **kwargs):
        c_clone_def = self.get_clone_def()
        for kw, kw_val in kwargs.items():
            if kw in c_clone_def: c_clone_def[kw] = kw_val
        clones_and_phenos = ClonesAndPhenotypes(phenotypes = self.phenotypes)
        for cell in self.cell_list:
            clones_and_phenos[cell.clone_rep(**c_clone_def)] += cell.phenotypes_and_counts
        return clones_and_phenos
        
    def get_clones_and_phenos_joint_distribution(self, **kwargs):
        clones_and_phenos = self.get_clones_and_phenos(**kwargs)
        joint_dist = np.array([[clones_and_phenos[clone][pheno] for pheno in self.phenotypes] for clone in self.clones])
        return joint_dist/np.sum(joint_dist)
        
    def get_phenotypic_flux(self, CellRepertoireB, phenotypes = None, min_cell_count = 0):
        
        phenotypic_flux = ClonesAndData()
        if phenotypes is None:
            phenotypes = self.phenotypes
        for clone in self.clones:
            c_phenosA = np.array([self.clones_and_phenos[clone][pheno] for pheno in phenotypes])
            c_phenosB = np.array([CellRepertoireB[clone][pheno] for pheno in phenotypes])
            if sum(c_phenosA) > min_cell_count and sum(c_phenosB) > min_cell_count:
                phenotypic_flux[clone] = np.sum(np.abs(c_phenosA/np.sum(c_phenosA) - c_phenosB/np.sum(c_phenosB)))
        
        return phenotypic_flux

    def get_significant_clones(self, name, pval_thresh):
#        if name not in self.pvalues:
#            InputError('name', name)
        return self.pvalues[name].get_significant_clones(pval_thresh)