from pgmpy.factors.discrete import State
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pyro.distributions.spanning_tree import *

import torch
import math
import random

def sample_spanning_tree_approx(N,W):
    w=[]
    for i in range(N):
        for j in range(i+1,N):
            w.append(math.log(W[i][j]))
    st=sample_tree(torch.tensor(w), mcmc_steps=1)
    return [(x.item(),y.item()) for [x,y] in st]

# samples from a tree distribution given evidence
def sample_from_tree(N, par, E, V, evidence):
    bn_edges=[(str(par[i]),str(i)) for i in range(1,N)]
    tree_network=BayesianNetwork(bn_edges)
    tree_network.add_cpds(TabularCPD(str(0), 2, [[V[0][0]], [V[0][1]]]))
    for i in range(1,N):
        tree_network.add_cpds(TabularCPD(str(i), 2, [[E[i][par[i]][0][0]/V[par[i]][0],E[i][par[i]][0][1]/V[par[i]][1]], [E[i][par[i]][1][0]/V[par[i]][0],E[i][par[i]][1][1]/V[par[i]][1]]],evidence=[str(par[i])], evidence_card=[2]))
    inference = BayesianModelSampling(tree_network)
    evi = []
    query_args={}
    query_vars=[]
    for i in range(N):
        if evidence[i]!=-1:
            evi.append(State(str(i),evidence[i]))
            query_args[str(i)]=evidence[i]
            query_vars.append(str(i))
    sample = inference.likelihood_weighted_sample(evidence=evi, size=1, show_progress=False)
    res=[int(sample.iloc[0][str(i)]) for i in range(N)]
    marginals = VariableElimination(tree_network)
    q = marginals.query(variables=query_vars, evidence={})
    marginal= q.get_value(**query_args)
    # print(q,marginal)
    return res,marginal#*sample.iloc[0]["_weight"]

# samples from a tree distribution given evidence
def sample_from_tree_autoregressive(N, par, E, V, evidence):
    bn_edges=[(str(par[i]),str(i)) for i in range(1,N)]
    tree_network=BayesianNetwork(bn_edges)
    tree_network.add_cpds(TabularCPD(str(0), 2, [[V[0][0]], [V[0][1]]]))
    for i in range(1,N):
        tree_network.add_cpds(TabularCPD(str(i), 2, [[E[i][par[i]][0][0]/V[par[i]][0],E[i][par[i]][0][1]/V[par[i]][1]], [E[i][par[i]][1][0]/V[par[i]][0],E[i][par[i]][1][1]/V[par[i]][1]]],evidence=[str(par[i])], evidence_card=[2]))

    marginals = VariableElimination(tree_network)

    evidence_args={}
    evidence_vars=[]
    for i in range(N):
        if evidence[i]!=-1:
            evidence_args[str(i)]=evidence[i]
            evidence_vars.append(str(i))

    q = marginals.query(variables=evidence_vars, evidence={})
    marginal= q.get_value(**evidence_args)

    sample=[-1 for i in range(N)]
    for i in range(N):
        if evidence[i]!=-1:
            sample[i]=evidence[i]
        else:
            q = marginals.query(variables=[str(i)], evidence=evidence_args, joint=False)
            sample[i]=0 if random.uniform(0, 1)<= q[str(i)].get_value(**{str(i):0}) else 1
            evidence_args[str(i)]=sample[i]
    return sample,marginal

# samples from a tree distribution given evidence
def sample_from_tree_factor_autoregressive(N, par, E, V, evidence):
    G = FactorGraph()
    G.add_nodes_from([str(i) for i in range(N)])
    f=[]
    f.append(DiscreteFactor(['0'], [2],[V[0][0], V[0][1]]))
    G.add_factors(f[0])
    G.add_edges_from([('0', f[0])])
    for i in range(1,N):
        f.append(DiscreteFactor([str(i),str(par[i])], [2,2],[E[i][par[i]][0][0]/V[par[i]][0],E[i][par[i]][0][1]/V[par[i]][1], E[i][par[i]][1][0]/V[par[i]][0],E[i][par[i]][1][1]/V[par[i]][1]]))
        G.add_factors(f[i])
        G.add_edges_from([(str(i), f[i]), (str(par[i]), f[i])])

    indicators=[None for i in range(N)]
    for i in range(N):
        if evidence[i]!=-1:
            indicators[i]=DiscreteFactor([str(i)], [2],[1-evidence[i], evidence[i]])
            G.add_factors(indicators[i])
            G.add_edges_from([(str(i), indicators[i])])

    marginal=G.get_partition_function()

    bn_edges=[(str(par[i]),str(i)) for i in range(1,N)]
    tree_network=BayesianNetwork(bn_edges)
    tree_network.add_cpds(TabularCPD(str(0), 2, [[V[0][0]], [V[0][1]]]))
    for i in range(1,N):
        tree_network.add_cpds(TabularCPD(str(i), 2, [[E[i][par[i]][0][0]/V[par[i]][0],E[i][par[i]][0][1]/V[par[i]][1]], [E[i][par[i]][1][0]/V[par[i]][0],E[i][par[i]][1][1]/V[par[i]][1]]],evidence=[str(par[i])], evidence_card=[2]))

    marginals = VariableElimination(tree_network)

    evidence_args={}
    evidence_vars=[]
    for i in range(N):
        if evidence[i]!=-1:
            evidence_args[str(i)]=evidence[i]
            evidence_vars.append(str(i))

    # if marginal<0:
    #     q = marginals.query(variables=evidence_vars, evidence={})
    #     q_marginal= q.get_value(**evidence_args)
    #     print(evidence, marginal,q_marginal)

    sample=[-1 for i in range(N)]
    for i in range(N):
        if evidence[i]!=-1:
            sample[i]=evidence[i]
        else:
            q = marginals.query(variables=[str(i)], evidence=evidence_args, joint=False)
            sample[i]=0 if random.uniform(0, 1)<= q[str(i)].get_value(**{str(i):0}) else 1
            evidence_args[str(i)]=sample[i]



    return sample,marginal

# samples from a tree distribution given evidence
def sample_from_tree_parallel(N, par, E, V, evidence):
    bn_edges=[(str(par[i]),str(i)) for i in range(1,N)]
    tree_network=BayesianNetwork(bn_edges)
    tree_network.add_cpds(TabularCPD(str(0), 2, [[V[0][0]], [V[0][1]]]))
    for i in range(1,N):
        tree_network.add_cpds(TabularCPD(str(i), 2, [[E[i][par[i]][0][0]/V[par[i]][0],E[i][par[i]][0][1]/V[par[i]][1]], [E[i][par[i]][1][0]/V[par[i]][0],E[i][par[i]][1][1]/V[par[i]][1]]],evidence=[str(par[i])], evidence_card=[2]))
    evidence_args={}
    evidence_vars=[]
    query_vars=[]

    for i in range(N):
        if evidence[i]!=-1:
            evidence_args[str(i)]=evidence[i]
            evidence_vars.append(str(i))
        else:
            query_vars.append(str(i))
    marginals = VariableElimination(tree_network)
    q = marginals.query(variables=query_vars, evidence=evidence_args, joint=False)
    marginal_vector=[-1 for i in range(N)]
    for i in range(N):
        if evidence[i]!=-1:
            marginal_vector[i]=evidence[i]
        else:
            marginal_vector[i]=q[str(i)].get_value(**{str(i):1})

    G = FactorGraph()
    G.add_nodes_from([str(i) for i in range(N)])
    f=[]
    f.append(DiscreteFactor(['0'], [2],[V[0][0], V[0][1]]))
    G.add_factors(f[0])
    G.add_edges_from([('0', f[0])])
    for i in range(1,N):
        f.append(DiscreteFactor([str(i),str(par[i])], [2,2],[E[i][par[i]][0][0]/V[par[i]][0],E[i][par[i]][0][1]/V[par[i]][1], E[i][par[i]][1][0]/V[par[i]][0],E[i][par[i]][1][1]/V[par[i]][1]]))
        G.add_factors(f[i])
        G.add_edges_from([(str(i), f[i]), (str(par[i]), f[i])])

    indicators=[None for i in range(N)]
    for i in range(N):
        if evidence[i]!=-1:
            indicators[i]=DiscreteFactor([str(i)], [2],[1-evidence[i], evidence[i]])
            G.add_factors(indicators[i])
            G.add_edges_from([(str(i), indicators[i])])

    marginal=G.get_partition_function()

    return marginal,marginal_vector

# assumes p and q are lists with equal size
def kld(p,q):
    d=0
    assert(len(p)==len(q))
    for i in range(len(p)):
        if p[i]<0 or p[i]>1:
            print(p,q)
        if q[i]<0 or q[i]>1:
            print(p,q)
        if p[i]==0 or q[i]==0: continue
        if p[i]==1 or q[i]==1: continue
        d+=p[i]*(math.log(p[i])-math.log(q[i]))
        d+=(1-p[i])*(math.log(1-p[i])-math.log(1-q[i]))
    return d


def dfs(node, par, parents, adj_list):
    parents[node]=par
    for ch in adj_list[node]:
        if ch!=par:
            dfs(ch,node,parents,adj_list)

# assumes vertices of tree are 0...N-1
def get_parents(edges, N):
    adj_list=[[] for i in range(N)]
    for (i,j) in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)
    parents=[-1 for i in range(N)]
    dfs(0,-1,parents,adj_list)
    if parents.count(-1)!=1:
        print(edges)
        print(parents)
    assert(parents.count(-1)==1)
    return parents

def generate_marginal_terms_helper(current,n,idx,data,evidence):
    if idx==n-1:
        if evidence[idx]!=-1:
            current[idx]=evidence[idx]
            data.append(current.copy())
        else:
            current[idx]=0
            data.append(current.copy())
            current[idx]=1
            data.append(current.copy())
    else:
        if evidence[idx]!=-1:
            current[idx]=evidence[idx]
            generate_marginal_terms_helper(current,n,idx+1,data,evidence)
        else:
            current[idx]=0
            generate_marginal_terms_helper(current,n,idx+1,data,evidence)
            current[idx]=1
            generate_marginal_terms_helper(current,n,idx+1,data,evidence)

# evidence has 0,1,-1; with -1 corresponding to missing
def generate_marginal_terms(n,evidence):
    data=[]
    current=[-1 for i in range(n)]
    generate_marginal_terms_helper(current,n,0,data, evidence)
    return torch.tensor(data)

if __name__ == "__main__":
    # print(generate_marginal_terms(5,[-1,1,1,0,-1]))
    print(sample_spanning_tree_approx(3,torch.tensor([[2,2,1],[2,2,1],[2,2,1]])))