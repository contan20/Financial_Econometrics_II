# FINANCIAL ECONOMETRICS 2
# Student: Andrea Contenta
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.utils import resample

import statsmodels.api as sm
from statsmodels.multivariate.pca import PCA as PCA

#from scipy.spatial.distance import pdist,squareform
from scipy import stats

import scipy.optimize as opt
#from scipy.optimize import minimize
#from scipy.optimize import NonlinearConstraint
#from tqdm import tqdm

#import optim_functions as opt

### PART 1 ### 
# Through principal component analysis of the European equity returns, 
# extract the K main latent factors (principal components) of 
# the European equity market, i.e. the core factors that drive the returns 
# of European equities.

# LOADING
returns=np.array(pd.read_excel('/Users/andreacontenta/Library/Mobile Documents/com~apple~CloudDocs/M2 203 - drive/ECON2/Individual Project/EUROSTOXX50_DATA.xlsx', usecols='C:AX'))
my_index=np.array(pd.read_excel('/Users/andreacontenta/Library/Mobile Documents/com~apple~CloudDocs/M2 203 - drive/ECON2/Individual Project/EUROSTOXX50_DATA.xlsx', usecols='B'))
#NORMALIZATION OF THE RETURNS
returns_n=preprocessing.scale(returns)

# FACTORISATION N COMPONENTS
pca_out=PCA((returns_n))

# EIGENVALUES
pca_eval=pca_out.eigenvals

# RELATIVE EIGENVALUES
pca_eval_r=pca_eval/np.mean(pca_eval)

# EIGENVECTORS
pca_evec=pca_out.eigenvecs

# % OF EXPLAINED VARIANCE
pca_var=pca_eval/np.sum(pca_eval)

# CUMULATED EXPLAINED VARIANCE
pca_var_c=np.cumsum(pca_var)

# PRINCIPAL COMPONENTS
pca_comp=pca_out.factors
#little ajustment
pca_comp[:,0]=-1*pca_comp[:,0]

# CORRELATION OF PRINCIPAL COMPONENTS
corr=np.corrcoef(pca_comp,rowvar=0)

# SHARE OF VARIANCE EXPLAINED
npc=pca_comp.shape[1]
r_squared=np.zeros((npc,returns.shape[1]))
r2val=np.zeros((2,len(returns.T)))
for j in range(1,npc+1,1):
    for i in range(0,returns.shape[1],1):
        model = sm.OLS(returns[:,i],sm.add_constant(pca_comp[:,0:j]))
        results = model.fit()
        r_squared[j-1,i]=results.rsquared_adj
r_squared_avg=np.mean(r_squared,axis=1)

# set a target r-squared
r2_target=0.60
opt_factors=np.zeros(len(returns.T))
for i in range(len(pca_comp.T)):
    opt_factors[i]=[n for n,i in enumerate(r_squared[:,i]) if i>r2_target][0]
    

# BAI AND NG IC CRITERIA
# this is for the general number of criteria
pca_ic=pca_out.ic



# PLOTS
# Example with a specific asset and specific number of factors
n_asset=3
n_components=opt_factors[n_asset]+1 # optimal number of parameters
model = sm.OLS(returns[:,n_asset],sm.add_constant(pca_comp[:,0:int(n_components)]))
results = model.fit()

# I think this is wrong
# plt.suptitle('Asset returns and principal components via OLS')
# plt.plot(results.fittedvalues, label='fitted', linewidth='1')
# plt.plot(returns[:,n_asset], label='observed', linewidth='0.5')
# plt.xlabel('Time')
# plt.ylabel('Returns')
# plt.legend()
# plt.show()

plt.suptitle('R-squared and principal components')
plt.plot(r_squared_avg)
plt.xlabel('Number of factors')
plt.ylabel('R-squared')
plt.show()

fig,ax = plt.subplots()
plt.suptitle('Explained variance')
ax.bar(np.arange(len(pca_var)),pca_var, label='Marginal')
ax.set_xlabel('Number of factors')
ax.set_ylabel('Marginal explained variance')
ax2=ax.twinx()
ax2.plot(pca_var_c, color='red', label='Cumulative')
ax2.set_ylabel('Cumulative explained variance')
plt.legend(loc='lower right')
plt.show()

# Set the optimal number of PC (maximum of 20 PCs)
# build a matrix to compare the output of the three different criteria
optimal=[]
for i in range(3):
    optimal_temp=np.where(pca_ic[0:20,i] == min(pca_ic[0:20,i]))
    optimal=np.append(optimal, optimal_temp)
print('The number of optimal PCs that minimize Bai-Ng information criteria is', int(optimal[0]+1))


### PART 2 ###
# Estimate the linear model for each asset
# beta_factors contains alpha and betas
# store the results of the fitted returns


# Using stock-specific factors (ie, target R-squared)
# beta_factors=[]
# fitted_returns=[]
# for i in range(len(returns.T)):
#     model_ols = sm.OLS(returns[:,i],sm.add_constant(pca_comp[:,0:int(opt_factors[i])]))
#     results_ols = model_ols.fit()
#     beta_factors.append(results_ols.params)
#     fitted_returns.append(results_ols.fittedvalues)

# Global view of estimated factors
# my_beta_factors=pd.DataFrame(beta_factors)
# my_beta_factors.plot.bar(stacked=True)
# plt.show()

# Using information criteria
beta_factors_ic=np.zeros((int(optimal[0])+1,len(returns.T)))
fitted_returns_ic=np.zeros((len(returns),len(returns.T)))
for i in range(len(returns.T)):
    model_ols_ic = sm.OLS(returns[:,i],sm.add_constant(pca_comp[:,0:int(optimal[0])]))
    results_ols_ic = model_ols_ic.fit()
    beta_factors_ic[:,i]=results_ols_ic.params
    fitted_returns_ic[:,i]=results_ols_ic.fittedvalues

# Single-stock perspective
# target R-squared
n_stock=10
# fig,ax = plt.subplots()
# plt.suptitle('Observed and estimated stock returns - Target R-squared')
# plt.plot(fitted_returns[n_stock], c='red', label='fitted')
# plt.legend(loc='upper right')
# plt.xlabel('Time')
# plt.ylabel('Returns (in %)')
# ax2=ax.twinx()
# plt.plot(returns[:,n_stock], label='observed')
# plt.legend(loc='upper left')
# plt.show()

# Single-stock perspective
# Information criteria perspective
fig,ax = plt.subplots()
plt.suptitle('Observed and estimated stock returns - Bai-Ng criteria')
plt.plot(fitted_returns_ic[:,n_stock], c='red', label='fitted')
plt.legend(loc='upper right')
plt.xlabel('Time')
plt.ylabel('Returns (in %)')
ax2=ax.twinx()
plt.plot(returns[:,n_stock], label='observed')
plt.legend(loc='upper left')
plt.show()


### PART 3 ### 
# Compute the weights of the K equity portfolios designed to 
# replicate the K core equity factors

def port_minvol_ro(mean, cov, ro):
    def objective(W, R, C, ro):
        # calculate mean/variance of the portfolio
        varp=np.dot(np.dot(W.T,cov),W)
        #objective: min vol
        util=varp**0.5
        return util
    n=len(cov)
    # initial conditions: equal weights
    W=np.ones([n])/n
    # weights between 0%..100%: no shorts
    b_=[(0.,1.) for i in range(n)]   
    # No leverage: unitary constraint (sum weights = 100%)
    c_= ({'type':'eq', 'fun': lambda W: sum(W)-1. } , {'type':'eq', 'fun': lambda W: np.dot(W.T,mean)-ro })
    optimized=opt.minimize(objective,W,(mean,cov,ro),
                                      method='SLSQP',constraints=c_,bounds=b_, options={'maxiter': 100, 'ftol': 1e-08})
    return optimized.x

mean=np.mean(returns, axis=0)
cov=np.cov(returns,rowvar=0)
w=np.zeros((len(returns),len(returns.T), int(optimal[0]+1)))
port_ret=np.zeros((len(returns),len(returns.T), int(optimal[0]+1)))
for j in range(int(optimal[0]+1)):
    for i in range(len(returns)):
        print('Progress: '+str(j+1)+'.'+str(i+1))
        w[i,:,j]=port_minvol_ro(mean, cov, pca_comp[i,j]) # axis = 2
        port_ret[:,:,j]=w[:,:,j]*returns
    plt.suptitle('Factor mimicking portfolio - '+str(j+1)+'^ component')
    plt.plot(np.sum(port_ret[:,:,j], axis=1),label='Portfolio')
    plt.plot(pca_comp[:,j],label='Factor')
    plt.legend()
    plt.show()

### PART 4 ### 
# Estimate the alpha of these K portfolios against the market benchmark

port_ret_sum=np.sum(port_ret,axis=1)
beta_capm=np.zeros((2,len(port_ret_sum.T)))
for i in range(len(port_ret_sum.T)):
    model_capm=sm.OLS(port_ret_sum[:,i],sm.add_constant(my_index))
    results_capm=model_capm.fit()
    beta_capm[:,i]=results_capm.params

# Visualize alphas
plt.bar(np.arange(len(beta_capm[0,:])),beta_capm[0,:], width=0.5)
plt.xticks(np.arange(len(beta_capm[0,:])),('1','2','3','4'))
plt.xlabel('Portfolios')
plt.ylabel('Estimated alpha')
plt.suptitle('Estimated alpha against market benchmark')
plt.show()


### PART 5 ### 
# Assess the impact of errors in the estimation of the covariance 
# matrix Ω̂, on the estimated alphas of the K replicating portfolios. 
# Compute their confidence intervals at the 95% level.

def port_minvol(mean, cov):
    def objective(W, R, C):
        # calculate mean/variance of the portfolio
        varp=np.dot(np.dot(W.T,cov),W)
        #objective: min vol
        util=varp**0.5
        return util
    n=len(cov)
    # initial conditions: equal weights
    W=np.ones([n])/n                 
    # weights between 0%..100%: no shorts
    b_=[(0.,1.) for i in range(n)]   
    # No leverage: unitary constraint (sum weights = 100%)
    c_= ({'type':'eq', 'fun': lambda W: sum(W)-1. })
    optimized=opt.minimize(objective,W,(mean,cov),
                                      method='SLSQP',constraints=c_,bounds=b_,options={'maxiter': 100, 'ftol': 1e-08})
    return optimized.x

#from random import random
import time 

n_boot=50
beta_capm_boot=np.zeros((n_boot,2,int(optimal[0]+1)))
w_boot=np.zeros((n_boot,len(returns),len(returns.T), int(optimal[0]+1)))
port_ret_boot=np.zeros((n_boot,len(returns),len(returns.T), int(optimal[0]+1)))

beta_capm_boot=np.zeros((n_boot,2,int(optimal[0]+1)))

w_tot=[]
port_ret_boot_tot=[]
sim_ret=[]
mean=np.mean(returns,axis=0)
residuals=returns-mean
for i in range(n_boot):
    time.sleep(2)
    print('*** '+str(i+1)+'^ bootstrap series ***')
    # don't store those values as we directly compute what we need
    ret_boot=resample(residuals,replace=True)+mean
    mean_boot=np.mean(ret_boot,axis=0)
    cov_boot=np.cov(ret_boot,rowvar=0)
    # compute optimized bootstrapped FMP weights with target return
    for c in range(int(optimal[0]+1)):
        time.sleep(3)
        print(' '+str(c+1)+'^ FMP')
        for j in range(len(returns)):
            w_boot[i,j,:,c]=port_minvol_ro(mean_boot, cov_boot, pca_comp[j,c]) # axis = 2    
            port_ret_boot[i,:,:,c]=w_boot[i,:,:,c]*ret_boot
            #w_boot[i,j,:,c]=random()
            #port_ret_boot[i,:,:,c]=w_boot[i,:,:,c]*ret_boot 
    w_tot.append(w_boot[i,:,:,:])  
    port_ret_boot_tot.append(port_ret_boot[i,:,:,:])   

port_ret_sum_boot=np.sum(port_ret_boot_tot,axis=2) # dimensions n_boot x 190 x 4

for i in range(n_boot):
    for k in range(len(beta_capm_boot.T)):
        model_capm_boot=sm.OLS(port_ret_sum_boot[i,:,k],sm.add_constant(my_index))
        results_capm_boot=model_capm_boot.fit()
        beta_capm_boot[i,:,k]=results_capm_boot.params

plt.suptitle('Distribution of bootstrapped FMPs\' alphas')
plt.hist(beta_capm_boot[:,0,:])
plt.legend(['1^','2^','3^','4^'])
plt.xlabel('Estimated alpha')
plt.ylabel('Frequency')
plt.show()

plt.suptitle('Distribution of bootstrapped FMPs\' betas')
plt.hist(beta_capm_boot[:,1,:])
plt.legend(['1^','2^','3^','4^'])
plt.xlabel('Estimated beta')
plt.ylabel('Frequency')
plt.show()

plt.suptitle('Boxplot of bootstrapped FMPs\' alphas')
plt.boxplot(beta_capm_boot[:,0,:])
plt.xlabel('FMPs')
plt.ylabel('Value')
plt.show()

plt.suptitle('Boxplot of bootstrapped FMPs\' betas')
plt.boxplot(beta_capm_boot[:,1,:])
plt.xlabel('FMPs')
plt.ylabel('Value')
plt.show()

alpha_thresholds=np.quantile(beta_capm_boot[:,0,:],(0.025,0.975))
print('The 95% confidence interval of global bootstrapped alphas is ['+"%.4f" % round(((alpha_thresholds[0])), 4)+','+"%.4f" % round(((alpha_thresholds[1])), 4)+'].')

number=['first','second','third','fourth']
for i in range(4):
    alpha_thresholds_loop=np.quantile(beta_capm_boot[:,0,i],(0.025,0.975))
    print('The 95% confidence interval of the ' +number[i]+' bootstrapped alphas is ['+"%.4f" % round(((alpha_thresholds_loop[0])), 4)+','+"%.4f" % round(((alpha_thresholds_loop[1])), 4)+'].')


# Extra: visualisation of the estimation problem

# visualize the estimation error of the variance of a specific asset
# ie assuming normal distrib
# simulated risk return distribution for the first 10 assets

n_sim=1000
storage_stats=np.zeros((n_sim,len(returns.T),4))

for i in range(n_sim):
    sim=resample(returns,replace=True)
    storage_stats[i,:,0]=np.mean(sim, axis=0)
    storage_stats[i,:,1]=np.std(sim, axis=0)
    storage_stats[i,:,2]=stats.skew(sim,axis=0)
    storage_stats[i,:,3]=stats.kurtosis(sim, axis=0)
    #storage_cor[i,:,:]=np.corrcoef(sim, rowvar=0)

#original moments
mean=np.mean(returns,axis=1)
std=np.std(returns,axis=1)
skew=stats.skew(returns, axis=1)
kurt=stats.kurtosis(returns, axis=1)
#cor=np.corrcoef(returns,rowvar=0)

n_asset=9
for i in range(n_asset):
    plt.scatter(storage_stats[:,i,1], storage_stats[:,i,0], s=1)
    plt.suptitle('Simulated risk-return profile of the first '+str(n_asset+1)+' assets')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    

n_asset2=34
sim_stats=storage_stats[:,n_asset2,:]
error_mean=storage_stats[:,n_asset2,0]-mean[n_asset2]
error_std=storage_stats[:,n_asset2,1]-std[n_asset2]
error_skew=storage_stats[:,n_asset2,2]-skew[n_asset2]
error_kurt=storage_stats[:,n_asset2,3]-kurt[n_asset2]
    
# Distribution of the estimation error of the mean for the n_asset2 
fig, axs = plt.subplots(2, 2, figsize=(10,8))
axs[0,0].hist(error_mean, density=True, bins=50)
axs[0,0].set_title('Estimation error of the mean')
axs[0,1].set_title('Estimation error of the variance')
axs[0,1].hist(error_std, density=True, bins=50)
axs[1,0].set_title('Estimation error of the skewness')
axs[1,0].hist(error_skew, density=True, bins=50)
axs[1,1].set_title('Estimation error of the kurtosis')
axs[1,1].hist(error_kurt, density=True, bins=50)
#plt.suptitle('Distribution of estimation error for the '+str(n_asset2+1)+'-th asset')
plt.show()
