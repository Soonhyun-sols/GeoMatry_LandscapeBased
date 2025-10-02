import numpy as np
import ase.io
from scipy.optimize import minimize, OptimizeResult
from typing import Callable, Dict, Sequence, Optional, List
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

def getMaxDirection(optimizer, params_0, index):
    Ra_star_0=optimizer.fitting._get_Ra_star(params_0,optimizer.graph_builders[index],optimizer.Ra_stars[index],optimizer.Zas[index],optimizer.fixed_atom_indices)
    direction=(Ra_star_0-optimizer.Ra_stars[index]).detach().numpy()
    return direction/np.linalg.norm(direction)

def drawGraphs(optimizer_Energy,params,direction, index):
    if (np.isnan(direction[0,0])):
        direction=np.zeros(direction.shape)
        direction[-1,0]=1
    if ('k' in params.keys()):
        optimizer_Energy.ff.reset_parameters(params['k'],params['r0'])
    else:
        optimizer_Energy.ff.reset_parameters(params)
    Ra_temp=copy.deepcopy(optimizer_Energy.Ra_stars[index])
    atoms=get_ase_atoms(Ra_temp, optimizer_Energy.Zas[index])
    x=np.linspace(-0.1,0.1,30)
    z1=copy.deepcopy(x)
    for i in range(len(x)):
        Ra_temp[:,:]+=x[i]*direction
        E0=optimizer_Energy.ff.get_E(Ra_temp,optimizer_Energy.Zas[index],*optimizer_Energy.graph_builders[index](atoms))
        z1[i]=E0
        Ra_temp[:,:]-=x[i]*direction
    E0=optimizer_Energy.ff.get_E(Ra_temp,optimizer_Energy.Zas[index],*optimizer_Energy.graph_builders[index](atoms)).detach()
    plt.scatter(x,z1)
    #plt.scatter(x,E0+x**2)


def drawGraphs_Single(optimizer_Energy,params,direction):
    if (np.isnan(direction[0,0])):
        direction=np.zeros(direction.shape)
        direction[-1,0]=1
    if ('k' in params.keys()):
        optimizer_Energy.ff.reset_parameters(params['k'],params['r0'])
    else:
        optimizer_Energy.ff.reset_parameters(params)
    Ra_temp=copy.deepcopy(optimizer_Energy.Ra_star)
    atoms=get_ase_atoms(Ra_temp, optimizer_Energy.Za)
    x=np.linspace(-0.1,0.1,30)
    z1=copy.deepcopy(x)
    for i in range(len(x)):
        Ra_temp[:,:]+=x[i]*direction
        E0=optimizer_Energy.ff.get_E(Ra_temp,optimizer_Energy.Za,*optimizer_Energy.graph_builder(atoms))
        z1[i]=E0
        Ra_temp[:,:]-=x[i]*direction
    E0=optimizer_Energy.ff.get_E(Ra_temp,optimizer_Energy.Za,*optimizer_Energy.graph_builder(atoms)).detach()
    plt.scatter(x,z1)
    #plt.scatter(x,E0+x**2)

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
        for i in tqdm(range(epoch)):            
            loss=0
            trueloss=0
            optimizer.zero_grad()
            N=self.Ra_star.shape[0]
            #deviations_magnitude=e_size*np.sqrt(truncated_chi2_rvs(3*N,0,3*N+sg*(6*N)**0.5,size=eN)).reshape(1,1,eN,1)
            #deviations_direction=np.random.normal(0,e_size,[N,eN,3]).reshape(1,N,eN,3)
            deviations_magnitude=e_size*torch.randn([eN//2]).reshape(1,1,eN//2,1)
            deviations_magnitude=torch.cat([deviations_magnitude,-deviations_magnitude],dim=2)
            if (eN%2==1):
                deviations_magnitude=torch.cat([deviations_magnitude,np.zeros([1,1,1,1])],dim=2)
            
            
            '''
            deviations_direction=np.random.normal(0,e_size,[N,1,3]).reshape(1,N,1,3)
            deviations_direction[:,self.fixed_atom_indices,:,:]=0
            if (np.min(np.linalg.norm(np.linalg.norm(deviations_direction,axis=3),axis=1))<0.0000001): continue
            deviations_direction/=np.linalg.norm(np.linalg.norm(deviations_direction,axis=3,keepdims=True),axis=1,keepdims=True)
            '''
            
            Ra_star = self.Ra_star.clone().requires_grad_(True)
            E0 = self.ff.get_E(Ra_star, self.Za,*self.graph_builder(get_ase_atoms(self.Ra_star, self.Za)))
            Fa = self.ff.get_Fa(E0, Ra_star)
            if (torch.norm(Fa)==0): break
            Fa[self.fixed_atom_indices,:]=0
            deviations_direction = (Fa / torch.norm(Fa)).detach().reshape(1,N,1,3)
            deviations=deviations_direction*deviations_magnitude
            Ra_temp = self.Ra_star.clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
            Ra_temp += deviations
            Energies = torch.zeros([eN],dtype=torch.float64)
            for j in range(eN):
                Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Za,*self.graph_builder(get_ase_atoms(self.Ra_star, self.Za)))
            y=(Energies-E0)
            x=deviations_magnitude.reshape(-1)**2
            curvature = (torch.sum(x*y)/torch.sum(x*x))
            alpha = curvature
            if (alpha<5): alpha=5
            alpha=1/alpha/2
            deviations = torch.tensor(deviations_magnitude,device=self.device,requires_grad=False).float().reshape(eN)
            '''
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
            loss += torch.sum((deviations.reshape(-1)**2/2 - alpha * (Energies - E0))**2)/torch.sum(deviations_magnitude**2)/e_size/e_size
            trueloss += torch.sum((deviations.reshape(-1)**2/2 - alpha * (Energies - E0))**2)/torch.sum(deviations_magnitude**2)/e_size/e_size
            
            '''
            #deviations_magnitude=((1+np.random.rand(1,eN,1)*(3**3-1))**(1/3))*e_size*((3*N+sg*(6*N)**0.5)**0.5)
            deviations_magnitude=(1+np.random.rand(1,eN,1))*e_size*3
            maxerr=0
            deviations=deviations_direction*deviations_magnitude
            Ra_temp = self.Ra_star.clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
            Ra_temp += deviations
            Energies = torch.zeros([eN],dtype=torch.float64)
            for j in range(eN):
                Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Za,*self.graph_builder(get_ase_atoms(self.Ra_star, self.Za)))
                dE = -Energies[j] + E0
                maxerr=max(maxerr,torch.max(dE))
            '''
            '''
            alpha_neg = 1 / (maxerr+0.001)
            
            deviations = torch.tensor(deviations_magnitude,device=self.device,requires_grad=False).float().reshape(eN)
            for j in range(eN):
                exponentials=torch.exp(-alpha_neg*(Energies-E0).detach())
                p=(1/(1+exponentials)).detach()
                loss+=-alpha_neg*torch.sum((Energies-E0)*(1-p)*((deviations/e_size)**2),axis=0)/Energies.shape[0]
                trueloss+=-np.sum((torch.log(p)).detach().cpu().numpy())
            '''
            loss.backward()
            optimizer.step()
            if (i%print_period==0):
                print(i, curvature.item(), alpha, trueloss)
                drawGraphs_Single(self,self.ff.state_dict(),deviations_direction.reshape(N,3))
                plt.show()
                '''
                n2=getMaxDirection(self,self.ff.state_dict())
                drawGraphs(self,self.ff.state_dict(),n2)
                plt.show()
                '''
                
                #try:
                #    self.objective_function(self.fitting._flatten(self.ff.state_dict()).detach().numpy())
                #except:
                #    print('Error finding minimum')
                
        params_0=copy.deepcopy(self.ff.state_dict())
        print('final',params_0)
        return params_0
    
class MultipleSystemOptimizer:
    def __init__(self, 
        Ras: List[FloatTensor], Zas: List[IntTensor], graph_builders: GRAPH_BUILDER_TYPE,
        ff: baseFF, loss_function: Callable[[FloatTensor, FloatTensor], FloatTensor], 
        Ra_stars: Optional[List[FloatTensor]]=None,
        params_star: Optional[Dict[str, FloatTensor]]=None,
        fixed_atom_indices: Optional[Sequence[int]]=None,
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float64,
        fmax: float=1e-4, reoptimize=True
    ):
        self.ff = ff.to(device, dtype)
        self.Zas = Zas
        for i in range(len(self.Zas)):
            self.Zas[i] = self.Zas[i].to(device, dtype)
        self.graph_builders = graph_builders
        self.fixed_atom_indices = fixed_atom_indices
        self.fitting = GeometryFitting(self.ff, loss_function, fmax)
        self.device = device
        self.dtype = dtype
        if Ra_stars is None:
            if params_star is None:
                raise ValueError("Either Ra_star or params_star must be provided")
            if reoptimize:
                self.Ra_stars=[self.get_Ra_star(graph_builder, Ra, Za, params_star) for graph_builder, Ra, Za in zip(self.graph_builders, Ras, Zas)]
            else:
                self.Ra_stars=[Ra.to(device, dtype) for Ra in Ras]
            for i, (Ra_star, Za) in enumerate(zip(self.Ra_stars, Zas)):
                ase.io.write(f"Ra_star_{i}.xyz", get_ase_atoms(Ra_star, Za))
        else:
            self.Ra_stars = Ra_stars
            for i in range(len(self.Ra_stars)):
                self.Ra_stars[i] = self.Ra_stars[i].to(device, dtype).requires_grad_(False)

    def get_Ra_star(self, graph_builder, Ra: FloatTensor, Za: IntTensor, params_star: Dict[str, FloatTensor]) -> FloatTensor:
        self.fitting.reset_parameters(state_dict=params_star)
        Ra_star = self.fitting._get_Ra_star(params_star, graph_builder, Ra, Za, self.fixed_atom_indices)
        return Ra_star.detach().clone()

    def objective_function(self, params: np.ndarray, index: int) -> float:
        flattened_params = torch.from_numpy(params).to(self.device, self.dtype)
        loss, flattened_gradient = self.fitting.get_loss_and_gradient_flat(
            flattened_params, self.graph_builders[index], 
            self.Ra_stars[index], self.Zas[index], 
            self.fixed_atom_indices
        )
        print("loss:", loss.item())
        return loss.item(), flattened_gradient.detach().cpu().numpy()

    def optimize(self, params_0: Dict[str, FloatTensor], epoch, print_period, e_size, eN, sg) -> FloatTensor:
        for key in params_0.keys():
            params_0[key]=params_0[key].requires_grad_(True)
        if ('k' in params_0.keys()):
            self.ff.reset_parameters(params_0['k'],params_0['r0'])
        else:
            self.ff.reset_parameters(params_0)
        index=0
        #print(self.objective_function(self.fitting._flatten(self.ff.state_dict()).detach().numpy(),index)[0])
        optimizer = torch.optim.Adam(list(self.ff.parameters()))
        bar = tqdm(range(epoch))
        for i in bar:       
            optimizer.zero_grad()     
            loss=0
            trueloss=0
            '''           
            distances=torch.linspace(-1,1,10,dtype=torch.float32).reshape(-1)
            loss+=torch.sum((self.ff.nets['1_1'](distances+4)-(distances)**2)**2)
            loss.backward()
            optimizer.step()
            if (i%print_period==0):
                print(i, loss)
                plt.scatter(distances.detach().numpy(),self.ff.nets['1_1'](distances+4).detach().numpy())
                plt.scatter(distances.detach().numpy(),(distances.detach().numpy())**2)
                plt.show()

            #print(loss,self.ff.nets['1_1'](distances+3))
            continue
            '''
            for temp in range(10):
                index=np.random.randint(0,len(self.Ra_stars))
                N=self.Ra_stars[index].shape[0]
                deviations_magnitude=np.random.normal(0,e_size,[eN//2]).reshape(1,1,eN//2,1)
                deviations_magnitude=np.concatenate([deviations_magnitude,-deviations_magnitude],axis=2)
                #deviations_magnitude=np.linspace(-e_size*2,e_size*2,eN,dtype=np.float32).reshape(1,1,eN,1)
                if (eN%2==1):
                    deviations_magnitude=np.concatenate([deviations_magnitude,np.zeros([1,1,1,1])],axis=2,dtype=np.float32)
                Ra_star = self.Ra_stars[index].clone().requires_grad_(True)
                E0 = self.ff.get_E(Ra_star, self.Zas[index],*self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
                Fa = self.ff.get_Fa(E0, Ra_star)
                Fa[self.fixed_atom_indices,:]=0
                if (torch.norm(Fa)==0): break
                deviations_direction = (Fa / torch.norm(Fa)).detach().numpy().reshape(1,N,1,3)
                deviations = deviations_direction*deviations_magnitude
                Ra_temp = self.Ra_stars[index].clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
                Ra_temp += deviations
                Energies = torch.zeros([eN],dtype=torch.float64)
                distances=[]
                for j in range(eN):
                    distances.append(torch.sum((Ra_temp[0,0,j,:]-Ra_temp[0,1,j,:]).detach()**2)**0.5)
                    Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Zas[index],*self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
                deviations = torch.tensor(deviations_magnitude,device=self.device,requires_grad=False).float().reshape(eN)
                y=(Energies-E0)
                x=deviations.reshape(-1)**2
                
                curvature = (torch.sum(x*y)/torch.sum(x*x))
                alpha = curvature
                if (alpha<5): alpha=5
                alpha=1/(alpha)/2
                Threshold = 5
                nearMinimumLoss = torch.sum((deviations.reshape(-1)**2/2 - alpha * (Energies - E0))**2)/np.sum(deviations_magnitude**2)/e_size/e_size
                negativeCurvatureLoss = (Threshold>curvature)*(Threshold-curvature)**2
                #mixedLoss = (negativeCurvatureLoss + nearMinimumLoss*np.exp(curvature-Threshold/2))/(1+np.exp(curvature-Threshold/2))
                mixedLoss = nearMinimumLoss
                #mixedLoss = negativeCurvatureLoss*100
                loss     += mixedLoss
                trueloss += mixedLoss

                alpha_new=alpha/(e_size*e_size)
                #print(torch.norm(Fa)/curvature,curvature,mixedLoss)
                Emins = torch.min(Energies,dim=0,keepdim=True)[0]
                Emins = Emins.detach()
                logprob = deviations**2/2/e_size/e_size
                #logalpha = torch.log(torch.sum(torch.sum(torch.exp(alpha.reshape(N,1,3))*(deviations_direction_torch**2),axis=2),axis=0))
                #exponentials=torch.exp(logprob-logalpha*(Energies-Emins).detach())
                exponentials=torch.exp(logprob-alpha_new*(Energies-Emins).detach())
                NormailzationConstant = torch.sum(exponentials,dim=0,keepdim=True)
                p=(exponentials/NormailzationConstant).detach()
                #loss+=alpha_new*(torch.sum(torch.sum(Energies,axis=0)/Energies.shape[0]-torch.sum(p*Energies,axis=0)))
                #loss+=torch.sum(torch.sum(logalpha*Energies,axis=0)/Energies.shape[0]-torch.sum(p*logalpha*Energies,axis=0))
                #trueloss+=-np.sum((-logprob+torch.log(p)).detach().cpu().numpy())
                
                
                #deviations_magnitude=((1+np.random.rand(1,eN,1)*(3**3-1))**(1/3))*e_size*((3*N+sg*(6*N)**0.5)**0.5)
                deviations_magnitude=(1+np.random.rand(1,eN,1))*e_size*3
                maxerr=0
                deviations=deviations_direction*deviations_magnitude
                Ra_temp = self.Ra_stars[index].clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
                Ra_temp += deviations
                Energies = torch.zeros([eN],dtype=torch.float64)
                for j in range(eN):
                    Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Zas[index],*self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
                    dE = -Energies[j] + E0
                    maxerr=max(maxerr,torch.max(dE))
                epsilon=0.001
                alphaneg=1/(maxerr+epsilon)
                p=1/(1+torch.exp(alphaneg*(-Energies + E0))).detach()
                #loss += alphaneg*torch.sum((1-p)*(-Energies + E0))
                #trueloss += -torch.sum(torch.log(p))
            #loss=torch.sum((y-x)**2)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=f"{loss.item():.4f}")
            #print(self.ff.state_dict()['r0'][1,1],trueloss)
            if (i%print_period==0):
                print(i, 'curvature', curvature, nearMinimumLoss.item(), 'distance', (torch.norm(Fa)/curvature).item(), 'loss', trueloss.item())
                #print(i, 'curvature', curvature, negativeCurvatureLoss.item(), nearMinimumLoss.item(), 'distance', (torch.norm(Fa)/curvature).item(), 'loss', trueloss.item())
                print('trueloss',trueloss.item())
                print('loss', loss.item())
                print(Energies - E0)
                #print(self.ff.state_dict())
                drawGraphs(self,self.ff.state_dict(),deviations_direction.reshape(N,3),index)
                plt.show()
                '''
                n2=getMaxDirection(self,self.ff.state_dict())
                drawGraphs(self,self.ff.state_dict(),n2)
                plt.show()
                '''
                '''
                try:
                    self.objective_function(self.fitting._flatten(self.ff.state_dict()).detach().numpy(),index)
                except:
                    print('Error finding minimum')
                '''
        params_0=copy.deepcopy(self.ff.state_dict())
        return params_0