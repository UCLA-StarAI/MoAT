import torch
from torch import nn
import networkx as nx
import math
import utils
import random

from tqdm import tqdm

EPS=1e-7

class MoAT(nn.Module):

    ########################################
    ###     Parameter Initialization     ###
    ########################################

    def __init__(self, n, x, device='cpu'):
        super().__init__()

        self.n = n

        print('initializing params ...')
        with torch.no_grad():
            m = x.shape[0]

            # estimate marginals from data
            x = x.to(device)

            # pairwise marginals
            E = torch.zeros(n, n, 2, 2).to(device)
            block_size = (2 ** 30) // (n * n * 2 * 2)
            for block_idx in tqdm(range(0, m, block_size)):
                block_size_ = min(block_size, m - block_idx)
                x_block = x[block_idx:block_idx+block_size_]
                x_1, x_2 = x_block.unsqueeze(2), x_block.unsqueeze(1)
                x_1, x_2 = x_1.type(torch.float32), x_2.type(torch.float32)
                x_2d = torch.zeros(block_size_, n, n, 2, 2).to(device)
                x_2d[:,:,:,0,0] = torch.matmul(1.0 - x_1, 1.0 - x_2)
                x_2d[:,:,:,0,1] = torch.matmul(1.0 - x_1, x_2)
                x_2d[:,:,:,1,0] = torch.matmul(x_1, 1.0 - x_2)
                x_2d[:,:,:,1,1] = torch.matmul(x_1, x_2)
                E += torch.sum(x_2d, dim=0)
            E = (E+1.0) / float(m+2)
            E = E.to('cpu')
            E_compress=E[:,:,1,1].clone()

            # univariate marginals
            V = torch.zeros(n, 2)
            V[:, 1] = (torch.sum(x, dim=0)+1) / (float(m)+2)
            V[:, 0] = 1 - V[:, 1]
            V_compress=V[:,1].clone()

            # scale pairwise marginals to [0,1]
            upper_bound = torch.minimum(V_compress.unsqueeze(0), V_compress.unsqueeze(-1))
            lower_bound = torch.maximum(V_compress.unsqueeze(-1) + V_compress.unsqueeze(0) - 1.0,
                            torch.zeros(E_compress.shape).to(V_compress.device)+EPS)
            E_compress = torch.div((E_compress-lower_bound),(upper_bound - lower_bound+EPS))
            E_compress=torch.abs(E_compress-EPS)
            E_compress = torch.tril(E_compress, diagonal=-1)
            E_compress = torch.transpose(E_compress, 0, 1) + E_compress

            # logit for unconstrained paramter learning
            V_compress=torch.special.logit(V_compress)
            E_compress=torch.special.logit(E_compress)

            print('computing MI ...')
            E_new = torch.maximum(E, torch.ones(1) * EPS).to(device)
            V_new = torch.maximum(V.unsqueeze(1).unsqueeze(-1) * V.unsqueeze(1).unsqueeze(0), torch.ones(1) * EPS).to(device)
            MI = torch.sum(torch.sum(E_new * torch.log(E_new / V_new), dim=-1), dim=-1)
            MI += EPS

            MI=torch.special.logit(MI)

        # W stores the edge weights
        self.W = nn.Parameter(MI, requires_grad=True)
        self.E_compress=nn.Parameter(E_compress, requires_grad=True)
        self.V_compress=nn.Parameter(V_compress, requires_grad=True)



    ########################################
    ###             Inference            ###
    ########################################

    def forward(self, x):
        batch_size, d = x.shape
        n, W, V_compress, E_compress = self.n, self.W, torch.sigmoid(self.V_compress),torch.sigmoid(self.E_compress)
        upper_bound = torch.minimum(V_compress.unsqueeze(0), V_compress.unsqueeze(-1))
        lower_bound = torch.maximum(V_compress.unsqueeze(-1) + V_compress.unsqueeze(0) - 1.0,
                        torch.zeros(E_compress.shape).to(V_compress.device)+EPS)
        E_compress = E_compress * ((upper_bound - lower_bound)+EPS) + lower_bound


        V1 = V_compress # n
        V0 = 1 - V_compress
        V = torch.stack((V0, V1), dim=1) # n * 2

        E11 = E_compress # n * n
        E01 = V1.unsqueeze(0) - E11
        E10 = V1.unsqueeze(-1) - E11
        E00 = 1 - E01 - E10 - E11
        E0 = torch.stack((E00, E01), dim=-1) # n * n * 2
        E1 = torch.stack((E10, E11), dim=-1) # n * n * 2
        E = torch.stack((E0, E1), dim=2)

        E_mask = (1.0 - torch.diag(torch.ones(n)).unsqueeze(-1).unsqueeze(-1)).to(E.device)
        E = E * E_mask
        E=torch.clamp(E,0,1)

        W = torch.sigmoid(W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W

        # det(principal minor of L_0) gives the normalizing factor over the spanning trees
        L_0 = -W + torch.diag_embed(torch.sum(W, dim=1))

        Pr = V[torch.arange(n).unsqueeze(0), x]

        P = E[torch.arange(n).unsqueeze(0).unsqueeze(-1),
                torch.arange(n).unsqueeze(0).unsqueeze(0),
                x.unsqueeze(-1),
                x.unsqueeze(1)] # E[i, j, x[idx, i], x[idx, j]]
        P = P / torch.matmul(Pr.unsqueeze(2), Pr.unsqueeze(1)) # P: bath_size * n * n


        W = W.unsqueeze(0) # W: 1 * n * n; W * P: batch_size * n * n
        L = -W * P + torch.diag_embed(torch.sum(W * P, dim=2))  # L: batch_size * n * n

        y = torch.sum(torch.log(Pr), dim=1) + torch.logdet(L[:, 1:, 1:]) - torch.logdet(L_0[1:, 1:])

        if y[y != y].shape[0] != 0:
            print("NaN!")
            exit(0)

        return y

    ########################################
    ### Methods for Sampling Experiments ###
    ########################################

    # can be repurposed to just return samples!

    # spanning tree distribution normalization constant
    def log_Z(self):
        W = torch.sigmoid(self.W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W
        L_0 = -W + torch.diag_embed(torch.sum(W, dim=1))
        return torch.logdet(L_0[1:, 1:]).item()

    def sample_spanning_tree(self,G,W):
        st=nx.algorithms.random_spanning_tree(G,weight='weight').edges
        parents=utils.get_parents(st,self.n)

        # unnormalized weight of sampled spanning tree
        # log_w=1
        # for (i,j) in st:
        #     log_w+=math.log(W[i][j])
        # normalized weight of spanning tree
        # log_wst=log_w - Z

        return st, parents #, log_wst

    # returns V,E,W after projecting to right space
    def get_processed_parameters(self):
        n, V_compress, E_compress = self.n, torch.sigmoid(self.V_compress),torch.sigmoid(self.E_compress)
        upper_bound = torch.minimum(V_compress.unsqueeze(0), V_compress.unsqueeze(-1))
        lower_bound = torch.maximum(V_compress.unsqueeze(-1) + V_compress.unsqueeze(0) - 1.0,
                        torch.zeros(E_compress.shape).to(V_compress.device)+EPS)

        E_compress = E_compress * ((upper_bound - lower_bound)+EPS) + lower_bound

        V1 = V_compress # n
        V0 = 1 - V_compress
        V = torch.stack((V0, V1), dim=1) # n * 2
        V=V.cpu().detach().numpy()

        E11 = E_compress # n * n
        E01 = V1.unsqueeze(0) - E11
        E10 = V1.unsqueeze(-1) - E11
        E00 = 1 - E01 - E10 - E11
        E0 = torch.stack((E00, E01), dim=-1) # n * n * 2
        E1 = torch.stack((E10, E11), dim=-1) # n * n * 2
        E = torch.stack((E0, E1), dim=2)
        E_mask = (1.0 - torch.diag(torch.ones(n)).unsqueeze(-1).unsqueeze(-1)).to(E.device)
        E = E * E_mask
        E=E.cpu().detach().numpy()

        W = torch.sigmoid(self.W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W

        return V,E,W


    def get_true_marginals(self,evidence):
        n=self.n
        true_marginals=[1 for i in range(n)]
        for i in range(n):
            if evidence[i]!=-1:
                true_marginals[i]=evidence[i]
                continue
            evidence[i]=0
            data=utils.generate_marginal_terms(n,evidence)
            p_0=torch.sum(torch.exp(self.forward(data))).item()
            evidence[i]=1
            data=utils.generate_marginal_terms(n,evidence)
            p_1=torch.sum(torch.exp(self.forward(data))).item()

            true_marginals[i]=p_1/(p_0+p_1)
            evidence[i]=-1
        return true_marginals

    def get_importance_samples(self,evidence,num_samples=1):
        num_missing=evidence.count(-1)
        n=self.n
        V,E,W=self.get_processed_parameters()

        true_marginals=self.get_true_marginals(evidence)

        marginals=[0 for i in range(n)]
        norm=0

        klds=[]
        wts=[]
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i+1,self.n):
                G.add_edge(i,j,weight=W[i][j].item())

        data=utils.generate_marginal_terms(n,evidence)
        m_e=torch.sum(torch.exp(self.forward(data))).item()

        for it in range(num_samples):
            st, parents = self.sample_spanning_tree(G,W)
            res,wt=utils.sample_from_tree_factor_autoregressive(n,parents,E,V,evidence)
            for i in range(n):
                if res[i]:
                    marginals[i]+=wt
            norm+=wt
            wts.append(wt/m_e)
            approximate_marginals=[marginals[i]/norm for i in range(n)]
            kld=utils.kld(true_marginals,approximate_marginals)/num_missing
            klds.append(kld)

        return klds,wts

    def get_collapsed_importance_samples(self,evidence,num_samples=1):
        num_missing=evidence.count(-1)
        n=self.n
        V,E,W=self.get_processed_parameters()

        true_marginals=self.get_true_marginals(evidence)

        marginals=[0 for i in range(n)]
        norm=0

        klds=[]
        wts=[]
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i+1,self.n):
                G.add_edge(i,j,weight=W[i][j].item())

        data=utils.generate_marginal_terms(n,evidence)
        m_e=torch.sum(torch.exp(self.forward(data))).item()

        for it in range(num_samples):
            st, parents = self.sample_spanning_tree(G,W)
            wt,marginal_vector=utils.sample_from_tree_parallel(n,parents,E,V,evidence)
            for i in range(n):
                marginals[i]+=wt*max(0,min(1,marginal_vector[i]))
            norm+=wt
            wts.append(wt/m_e)
            approximate_marginals=[marginals[i]/norm for i in range(n)]
            kld=utils.kld(true_marginals,approximate_marginals)/num_missing
            klds.append(kld)


        return klds,wts

    def get_gibbs_samples(self,evidence,burn_in=10,num_samples=1):
        num_missing=evidence.count(-1)
        n=self.n
        V,E,W=self.get_processed_parameters()

        true_marginals=self.get_true_marginals(evidence)

        marginals=[0 for i in range(n)]
        norm=0

        cur=evidence.copy()
        for i in range(n):
            if cur[i]==-1:
                cur[i]=random.randint(0, 1)

        idx=-1
        klds=[]
        for it in range(burn_in+num_samples):
            evi=cur.copy()
            for idx in range(n):
                if evidence[idx]!=-1:
                    continue
                evi[idx]=-1
                # note that this generates the 0 term first followed by the 1 term
                data=utils.generate_marginal_terms(n,evi)
                p=torch.exp(self.forward(data))
                cur[idx]=0 if random.uniform(0, 1)<=p[0].item()/(p[0].item()+p[1].item()) else 1
                evi[idx]=cur[idx]

            if it<burn_in:
                continue

            for i in range(n):
                if cur[i]:
                    marginals[i]+=1
            norm+=1

            if it >= burn_in:
                approximate_marginals=[marginals[i]/norm for i in range(n)]
                kld=utils.kld(true_marginals,approximate_marginals)/num_missing
                klds.append(kld)

        return klds

