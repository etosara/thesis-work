import os
import pandas as pd
import censusgeocode as cg
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy as sp
import ipyparallel as ipp
import scipy.stats as ss

# national home-ownership rates
home_ownership = [.725,.419, .57, .57, .46,.57]

# inferred rental rates from home-ownership rates
rent_pop = 1 - np.array(home_ownership)

# dictionary of counts by class
ndict = {0: 63679, 1: 16959, 2: 1880, 3: 42, 4: 27822, 5: 685}

def naive(one,two = rent_pop):
    """Takes two lists of numbers, multplies them termwise, marginalizes,
    and returns the joint."""
    # instantiate variables
    temp = []
    output = []
    
    # multiply probabilities
    for i,j in zip(one,two):
        temp.append(i*j)
        
    # find the marginal and normalize
    total = sum(temp)
    
    for i in temp:
        output.append(i/total)
        
    return output
        
    
# complex variant of the preceding, where the multi-class
# is treated differently than the single class as it is the catch all.
def non_naive(one,two):
    """Not currently a fully functional operation.  Was intended to lump
    mixed predictions into multi-class, but will be rebuilt"""
    # instantiate variables
    output = []
    
    # multiply non-multi-class-probabilities
    for i,j in zip(one[:-1],two[:-1]):
        output.append(i*j)
        
    # find the multi-class remainder.
    output.append(1-sum(output))
    
    return output
    
# bootstrap for ACROSS TRACTS and EQUAL SAMPLING sampling OBS ONLY
def boot_strap_at_es():
    """This function bootstraps calculating the chi-square value
    from all tracts, assumes uniform sampling, and only samples
    the observations, not the expected.  Compares 'naive' against
    'loc_scale'."""
    import pandas as pd
    import numpy as np
    df = boot_nums.copy()
#     rename = dict(zip(df.columns,[0,1,2]))
#     df.rename(columns=rename,inplace = True)

    boot = []
    
    exp = np.zeros(6)
    
    for i,item in df.iterrows():
        exp += np.array(item["loc_scale"])

    for i in range(n):

        obs_temp = np.zeros(6)
        # generate bootstrap distribution
        for i,item in df.iterrows():

            # generate samples from the observed percentages, and sum
            # them over the given district
            obs_temp += np.random.multinomial(1,item["naive"])

        # sampled observed chi-square 
        chi_obs = np.sum((exp - obs_temp)**2 / exp)

        # find the difference in the expectation and the observation
        boot.append(chi_obs)

        # calculate the chi-square summed over all the districts,
        # and store these values in boot2
        
    return boot

def panel_plot(input_data,number,title,savename):
    
    name_dict = dict(zip([0,1,2,3,4,5],["Caucasian","African American","API","AIAN","Hispanic","Multi"]))

    plt.subplots(nrows = 2,ncols = 3,figsize = (18,12))

    for i in range(6):

        plt.subplot(2,3,i+1)
        data = np.array([x[i] for x in input_data])
        plt.hist(data,edgecolor = "w",density = True)
        plt.xlabel("Relative discrepancy in {} evictions".format(name_dict[i]))
        if i == 1:
            plt.title("Fig. {}\n{}".format(number,title))

        if np.mean(data) > 0:
            print("mu = ",np.mean(data),
                  ", sigma = ", np.std(data),
                  ", pval = ",ss.norm.cdf(0,np.mean(data),np.std(data)))
        else:
            print("mu = ",np.mean(data),
                  ", sigma = ", np.std(data),
                  ", pval = ",1-ss.norm.cdf(0,np.mean(data),np.std(data)))
        
    savename = samename.replace(" ","-")
    plt.savefig("f{}-{}.jpg".format(number,savename))
    plt.show()
    
def panel_plot(input_data,number,title,savename,num_dict = None):
    
    if num_dict == None:
        num_dict = {0: 63679, 1: 16959, 2: 1880, 3: 42, 4: 27822, 5: 685}

    name_dict = dict(zip([0,1,2,3,4,5],["Caucasian","African American","API","AIAN","Hispanic","Multi"]))

    plt.subplots(nrows = 2,ncols = 3,figsize = (18,12))

    for i in range(6):

        plt.subplot(2,3,i+1)
        data = np.array([x[i] for x in input_data])
        plt.hist(data,edgecolor = "w",density = True)
        plt.xlabel("Relative discrepancy in {} evictions".format(name_dict[i]))
        if i == 1:
            plt.title("Fig. {}\n{}".format(number,title))

        print("99percentiles = ",np.percentile(data,[.5,50,99.5]))
        print("relative rates = ",np.percentile(data,[.5,50,99.5])/num_dict[i],"\n")
        
    plt.savefig("f{}-{}.jpg".format(number,savename))
    plt.show()

def boot_strap_local(df,n = 1000, baseline = "loc_scale",model = "naive",classes = 6):
    """Slightly more robust variation of the fast boot strap.
    This variation won't crash if the last value
    in a probability profile is zero."""
    
    boot = []

    exp = np.zeros(classes)
    obs = np.zeros(classes)
    
    group = []
    
    for i, row in df.iterrows():

        exp += np.array(row[baseline])
        
        try:
            generate = np.random.multinomial(1,row[model],size = n)
            group.append(generate)
        except:
            attempt = np.random.multinomial(1,row[model][:-1],size = n)
            temp = np.zeros(shape = (n,classes))
            temp[:,:-1] = attempt
            group.append(temp)

    data = np.array(group)
    compress = np.sum(data,axis = 0) - np.array(exp)
    return compress

def load_stata_surnames():
    try:
        os.chdir("Thesis plus census/")
        census_name_data = pd.read_stata("census_surnames_lower.dta")
        os.chdir("..")
    except:
        print("failed")
        return None
    return census_name_data