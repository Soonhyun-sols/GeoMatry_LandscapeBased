import numpy as np
import ase.io
from scipy.optimize import minimize, OptimizeResult
from typing import Callable, Dict, Sequence, Optional
import torch
from torch import FloatTensor, IntTensor
from ..ff.graph import GRAPH_BUILDER_TYPE
from ..ff.baseFF import baseFF
from .calculator import get_ase_atoms
from .geomopt import GeometryFitting
from geomatry.ff.springs import SpringFF
from scipy.stats import chi2
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

def getMaxDirection(optimizer, params_0):
    Ra_star_0=optimizer.fitting._get_Ra_star(params_0,optimizer.graph_builder,optimizer.Ra_star,optimizer.Za,optimizer.fixed_atom_indices)
    direction=(Ra_star_0-optimizer.Ra_star).detach().numpy()
    return direction/np.linalg.norm(direction)

def drawGraphs(optimizer_Energy,params):
    n1=getMaxDirection(optimizer_Energy,params)
    optimizer_Energy.ff.reset_parameters(params['k'],params['r0'])
    Ra_temp=copy.deepcopy(optimizer_Energy.Ra_star)
    atoms=get_ase_atoms(Ra_temp, optimizer_Energy.Za)
    x=np.linspace(-2,5,100)
    z1=copy.deepcopy(x)
    for i in range(100):
        Ra_temp[:,:]+=x[i]*n1
        E0=optimizer_Energy.ff.get_E(Ra_temp,optimizer_Energy.Za,*optimizer_Energy.graph_builder(atoms))
        z1[i]=E0
        Ra_temp[:,:]-=x[i]*n1
    #z1[z1>30]=30
    plt.scatter(x,z1)
    plt.colorbar()


def truncated_chi2_rvs(df, low, high, size=1):
    cdf_low = chi2.cdf(low, df)
    cdf_high = chi2.cdf(high, df)
    
    # 2. Generate uniform random numbers in the range [cdf_low, cdf_high]
    uniform_variates = np.random.uniform(cdf_low, cdf_high, size)
    
    # 3. Apply the inverse CDF (PPF) to transform the uniform variates
    truncated_variates = chi2.ppf(uniform_variates, df)
    
    return truncated_variates

def callback(intermediate_result: OptimizeResult):
    print("this is a callback")

class SingleSystemOptimizer:
    def __init__(self, 
        Ra: FloatTensor, Za: IntTensor, graph_builder: GRAPH_BUILDER_TYPE,
        ff: baseFF, loss_function: Callable[[FloatTensor, FloatTensor], FloatTensor], 
        Ra_star: Optional[FloatTensor]=None,
        params_star: Optional[Dict[str, FloatTensor]]=None,
        fixed_atom_indices: Optional[Sequence[int]]=None,
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float64,
        fmax: float=1e-4
    ):
        self.ff = ff.to(device, dtype)
        self.Za = Za.to(device, dtype)
        self.graph_builder = graph_builder
        self.fixed_atom_indices = fixed_atom_indices
        self.fitting = GeometryFitting(self.ff, loss_function, fmax)
        self.device = device
        self.dtype = dtype
        if Ra_star is None:
            if params_star is None:
                raise ValueError("Either Ra_star or params_star must be provided")
            self.Ra_star = self.get_Ra_star(Ra, params_star)
            ase.io.write("Ra_star.xyz", get_ase_atoms(self.Ra_star, self.Za))
        else:
            self.Ra_star = Ra_star.to(device, dtype).requires_grad_(False)

    def get_Ra_star(self, Ra: FloatTensor, params_star: Dict[str, FloatTensor]) -> FloatTensor:
        self.fitting.reset_parameters(state_dict=params_star)
        Ra_star = self.fitting._get_Ra_star(params_star, self.graph_builder, Ra, self.Za, self.fixed_atom_indices)
        return Ra_star.detach().clone()

    def objective_function(self, params: np.ndarray) -> float:
        flattened_params = torch.from_numpy(params).to(self.device, self.dtype)
        loss, flattened_gradient = self.fitting.get_loss_and_gradient_flat(
            flattened_params, self.graph_builder, 
            self.Ra_star, self.Za, 
            self.fixed_atom_indices
        )
        print("loss:", loss.item())
        return loss.item(), flattened_gradient.detach().cpu().numpy()

    def optimize(self, params_0: Dict[str, FloatTensor], epoch, print_period, e_size, eN, sg) -> FloatTensor:
        for key in params_0.keys():
            params_0[key]=params_0[key].requires_grad_()
        self.ff.reset_parameters(params_0['k'],params_0['r0'])
        print(self.objective_function(self.fitting._flatten(self.ff.state_dict()).detach().numpy())[0])
        #alpha=torch.ones([3*self.Ra_star.shape[0]],device=self.device,requires_grad=True)
        alpha=torch.ones([1],device=self.device,requires_grad=True)
        optimizer = torch.optim.Adam(list(self.ff.parameters())+[alpha])
        drawGraphs(self,self.ff.state_dict())
        plt.show()
        for i in tqdm(range(epoch)):
            E0 = self.ff.get_E(self.Ra_star, self.Za,*self.graph_builder(get_ase_atoms(self.Ra_star, self.Za)))
            
            loss=0
            trueloss=0
            optimizer.zero_grad()
            N=self.Ra_star.shape[0]
            #deviations_magnitude=e_size*np.sqrt(truncated_chi2_rvs(3*N,0,3*N+sg*(6*N)**0.5,size=eN)).reshape(1,1,eN,1)
            #deviations_direction=np.random.normal(0,e_size,[N,eN,3]).reshape(1,N,eN,3)
            deviations_magnitude=np.abs(np.random.normal(0,e_size,[eN]).reshape(1,1,eN,1))
            deviations_direction=np.random.normal(0,e_size,[N,1,3]).reshape(1,N,1,3)
            deviations_direction[:,self.fixed_atom_indices,:,:]=0
            if (np.min(np.linalg.norm(np.linalg.norm(deviations_direction,axis=3),axis=1))<0.0000001): continue
            deviations_direction/=np.linalg.norm(np.linalg.norm(deviations_direction,axis=3,keepdims=True),axis=1,keepdims=True)
            deviations=deviations_direction*deviations_magnitude
            Ra_temp = self.Ra_star.clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
            Ra_temp += deviations
            Energies = torch.zeros([eN],dtype=torch.float64)
            for j in range(eN):
                Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Za,*self.graph_builder(get_ase_atoms(self.Ra_star, self.Za)))
            y=(Energies-E0).detach().numpy()
            x=deviations_magnitude.reshape(-1)**2
            alpha=1/(np.sum(x*y)/np.sum(x*x))/2/(e_size**2)
            if (alpha<0): alpha=1
            deviations = torch.tensor(deviations_magnitude,device=self.device,requires_grad=False).float().reshape(eN)
            Emins = torch.min(Energies,dim=0,keepdim=True)[0]
            Emins = Emins.detach()
            
            logprob = deviations**2/2/e_size/e_size
            #logalpha = torch.log(torch.sum(torch.sum(torch.exp(alpha.reshape(N,1,3))*(deviations_direction_torch**2),axis=2),axis=0))
            #exponentials=torch.exp(logprob-logalpha*(Energies-Emins).detach())
            exponentials=torch.exp(logprob-alpha*(Energies-Emins).detach())
            NormailzationConstant = torch.sum(exponentials,dim=0,keepdim=True)
            p=(exponentials/NormailzationConstant).detach()
            loss+=alpha*(torch.sum(torch.sum(Energies,axis=0)/Energies.shape[0]-torch.sum(p*Energies,axis=0)))
            #loss+=torch.sum(torch.sum(logalpha*Energies,axis=0)/Energies.shape[0]-torch.sum(p*logalpha*Energies,axis=0))
            trueloss+=-np.sum((-logprob+torch.log(p)).detach().cpu().numpy())
            
            '''
            loss += torch.sum((deviations.reshape(-1)**2/2/e_size/e_size - alpha * (Energies - E0))**2)
            trueloss += torch.sum((deviations.reshape(-1)**2/2/e_size/e_size - alpha * (Energies - E0))**2)
            '''
            deviations_magnitude=((1+np.random.rand(1,eN,1)*(3**3-1))**(1/3))*e_size*((3*N+sg*(6*N)**0.5)**0.5)
            maxerr=0
            deviations=deviations_direction*deviations_magnitude
            Ra_temp = self.Ra_star.clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
            Ra_temp += deviations
            Energies = torch.zeros([eN],dtype=torch.float64)
            for j in range(eN):
                Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Za,*self.graph_builder(get_ase_atoms(self.Ra_star, self.Za)))
                dE = -Energies[j] + E0
                maxerr=max(maxerr,torch.max(dE))
            
            alpha_neg = 1 / (maxerr+0.001)
            
            deviations = torch.tensor(deviations_magnitude,device=self.device,requires_grad=False).float().reshape(eN)
            for j in range(eN):
                exponentials=torch.exp(-alpha_neg*(Energies-E0).detach())
                p=(1/(1+exponentials)).detach()
                loss+=-alpha_neg*torch.sum((Energies-E0)*(1-p)*((deviations/e_size)),axis=0)/Energies.shape[0]
                trueloss+=-np.sum((torch.log(p)).detach().cpu().numpy())
            
            loss.backward()
            optimizer.step()
            if (i%print_period==0):
                print(i, alpha, trueloss)
                drawGraphs(self,self.ff.state_dict())
                plt.show()
                try:
                    self.objective_function(self.fitting._flatten(self.ff.state_dict()).detach().numpy())
                except:
                    print('Error finding minimum')
        params_0['k']=self.ff.k.detach()
        params_0['r0']=self.ff.r0.detach()
        print('final',params_0)
        return params_0