import torch
import numpy as np
import scipy


# input dimensions: [sim, xyzt, ...]
# output dimensions: [sim, ...]
def dot4(v1,v2):
    return -v1[:,3]*v2[:,3] + torch.sum(v1[:,:3]*v2[:,:3], dim=1)

###################
# MINERBO CLOSURE #
###################
# compute inverse Langevin function for closure
# Levermore 1984 Equation 20
def function(Z,fluxfac):
    return (1./np.tanh(Z) - 1./Z) - fluxfac
def dfunctiondZ(Z,fluxfac):
    return 1./Z**2 - 1./np.sinh(Z)**2
def get_Z(fluxfac):
    badlocs = np.where(np.abs(fluxfac)<1e-4)
    nsims = len(fluxfac)
    initial_guess = 1
    Z = np.array([scipy.optimize.fsolve(function, 
                                        initial_guess,
                                        fprime=dfunctiondZ,
                                        args=(fluxfac[i]))[0]\
                   for i in range(nsims)])
    residual = np.max(np.abs(function(Z,fluxfac)) )
    # when near isotropic, use taylor expansion
    # f \approx Z/3 - Z^3/45 + O(Z^5)
    Z[badlocs] = 3.*fluxfac[badlocs]
    return Z, residual

# compute the value of the distribution at a given costheta
# n is the total number density
# all inputs have dimensions [sim, mu]
def distribution(n, Z, costheta):
    assert(np.all(Z >= 0))
    badlocs = np.where(Z<1e-4)
    prefactor = Z/np.sinh(Z)
    prefactor[badlocs] = 1. - Z[badlocs]**2/6 # taylor expansion
    return n/(2.*np.pi) * prefactor * np.exp(Z*costheta)

# generate an array of theta,phi pairs that uniformily cover the surface of a sphere
# based on DOI: 10.1080/10586458.2003.10504492 section 3.3 but specifying n_j=0 instead of n
# output has dimensions [3,ndirections]
def uniform_sphere(nphi_at_equator):
    assert(nphi_at_equator > 0)

    dtheta = np.pi * np.sqrt(3) / nphi_at_equator

    xyz = []
    theta = 0
    phi0 = 0
    while(theta < np.pi/2):
        nphi = nphi_at_equator if theta==0 else int(round(nphi_at_equator * np.cos(theta)))
        dphi = 2*np.pi/nphi
        if(nphi==1): theta = np.pi/2
        
        for iphi in range(nphi):
            phi = phi0 = iphi*dphi
            x = np.cos(theta) * np.cos(phi)
            y = np.cos(theta) * np.sin(phi)
            z = np.sin(theta)
            xyz.append(np.array([x,y,z]))
            if(theta>0): xyz.append(np.array([-x,-y,-z]))
        theta += dtheta
        phi0 += 0.5 * dphi # offset by half step so adjacent latitudes are not always aligned in longitude

    return np.array(xyz).transpose()


#########################################################################################################
############################     generate_stable_F4_zerofluxfac   #######################################
#########################################################################################################
# create list of simulations known to be stable when the flux factor is zero
# input has dimensions  [nsims  , xyzt, nu/nubar, flavor]
# output has dimensions [nsims*2, xyzt, nu/nubar, flavor]
# array of fluxes that are stable to the FFI
#Adapted from: https://github.com/srichers/Rhea/blob/main/model_training/ml_generate.py
def generate_stable_F4_zerofluxfac(parms):
    F4i = torch.zeros((parms["n_generate"], 4, 2, parms["NF"]), device=parms["device"])
    
    # keep all of the fluxes zero
    # densities randomly from 0 to 1
    F4i[:,3,:,:] = torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"])

    # normalize
    ntot = torch.sum(F4i[:,3,:,:], dim=(1,2))
    F4i = F4i / ntot[:,None,None,None]

    # average if necessary
    if parms["average_heavies_in_final_state"]:
        F4i[:,:,:,1:] = torch.mean(F4i[:,:,:,1:], dim=3, keepdims=True)
    
    #This is the output label for stable vs unstable
    #Is the point unstable?
    #Yes => unstable = 1
    #No => unstable = 0
    unstable = torch.zeros((parms["n_generate"]), device=parms["device"])
    unstable[:] = 0 #Since all the generated points are stable
    return F4i, unstable[:,None]



#########################################################################################################
########################         generate_stable_F4_oneflavor          ##################################
#########################################################################################################
def generate_stable_F4_oneflavor(parms):
    # choose the flux to be in a random direction
    costheta = 2*(torch.rand(parms["n_generate"], device=parms["device"]) - 0.5)
    phi = 2*torch.pi*torch.rand(parms["n_generate"], device=parms["device"])
    Frand = torch.zeros(parms["n_generate"], 3, device=parms["device"])
    Frand[:,0] = torch.sqrt(1-costheta**2) * torch.cos(phi)
    Frand[:,1] = torch.sqrt(1-costheta**2) * torch.sin(phi)
    Frand[:,2] = costheta

    # choose a random flux factor
    fluxfac = torch.rand(parms["n_generate"], device=parms["device"])

    # set Frand to be the flux factor times the density. Assume that the density is 1
    Frand = Frand * fluxfac[:,None]

    # set F4i with the new flux values
    F4i = torch.zeros((parms["n_generate"]*2*parms["NF"], 4, 2, parms["NF"]), device=parms["device"])
    for i in range(2):
        for j in range(parms["NF"]):
            # set start and end indices for current flavor and nu/nubar
            start = i*parms["n_generate"] + j*parms["n_generate"]*2
            end   = start + parms["n_generate"]
            F4i[start:end,0:3,i,j] = Frand
            F4i[start:end,  3,i,j] = 1
    
    # average if necessary
    if parms["average_heavies_in_final_state"]:
        F4i[:,:,:,1:] = torch.mean(F4i[:,:,:,1:], dim=3, keepdims=True)

    #This is the output label for stable vs unstable
    #Is the point unstable?
    #Yes => unstable = 1
    #No => unstable = 0
    unstable = torch.zeros((parms["n_generate"]*2*parms["NF"]), device=parms["device"])
    unstable[:] = 0 #Since all the generated points are stable

    return F4i, unstable[:,None]

#########################################################################################################
###################################  generate_random_F4   ###############################################
#########################################################################################################
def generate_random_F4(parms):
    assert(parms["n_generate"] >= 0)
    F4i = torch.zeros((parms["n_generate"], 4, 2, parms["NF"]), device=parms["device"])

    # choose a random number density
    Ndens = torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"])
    Ndens[torch.where(Ndens==0)] = 1
    F4i[:,3,:,:] = Ndens

    # choose the flux to be in a random direction
    costheta = 2*(torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"]) - 0.5)
    phi = 2*torch.pi*torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"])
    sintheta = torch.sqrt(1-costheta**2)
    F4i[:,0,:,:] = sintheta * torch.cos(phi)
    F4i[:,1,:,:] = sintheta * torch.sin(phi)
    F4i[:,2,:,:] = costheta
    F4i[:,3,:,:] = 1

    # choose a random flux factor
    fluxfac = torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"])*parms["generate_max_fluxfac"]
    fluxfac = fluxfac**parms["ME_stability_zero_weight"]

    # multiply the spatial flux by the flux factor times the density.
    F4i[:,0:3,:,:] = F4i[:,0:3,:,:] * fluxfac[:,None,:,:]
    
    # scale by the number density
    F4i = F4i * Ndens[:,None,:,:]

    # normalize so the total number density is 1
    ntot = torch.sum(F4i[:,3,:,:], dim=(1,2))
    F4i = F4i / ntot[:,None,None,None]

    # average if necessary
    if parms["average_heavies_in_final_state"]:
        F4i[:,:,:,1:] = torch.mean(F4i[:,:,:,1:], dim=3, keepdims=True)

    return F4i


#########################################################################################################
###################################     has_crossing      ###############################################
#########################################################################################################
# check whether a collection of number fluxes is stable according to the maximum entropy condition
# generate many directions along which to test whether the distributions cross
# F4i has dimensions [sim, xyzt, nu/nubar, flavor]
def has_crossing(F4i, parms):
    assert(parms["NF"]==3)
    nsims = F4i.shape[0]

    # evaluate the flux factor for each species
    Fmag = np.sqrt(np.sum(F4i[:,0:3,:,:]**2, axis=1))
    Fhat = F4i[:,0:3,:,:] / Fmag[:,None,:,:] # [sim, xyz, nu/nubar, flavor]
    fluxfac = Fmag / F4i[:,3,:,:]

    # avoid nans by setting fluxfac to zero when F4i is zero
    badlocs = np.where(Fmag/np.sum(F4i[:,3,:,:],axis=(1,2))[:,None,None] < 1e-6)
    if len(badlocs)>0:
        Fhat[:,0][badlocs] = 0
        Fhat[:,1][badlocs] = 0
        Fhat[:,2][badlocs] = 1
        fluxfac[badlocs] = 0
    assert(np.all(Fhat==Fhat))

    # avoid nans by setting flux factor of 1 within machine precision to 1
    assert(np.all(fluxfac<=1+1e-6))
    fluxfac = np.minimum(fluxfac, 1)
    assert(np.all(fluxfac>=0))

    # get Z for each species
    Z = np.zeros((nsims,2,parms["NF"]))
    for i in range(2):
        for j in range(parms["NF"]):
            Z[:,i,j], residual = get_Z(fluxfac[:,i,j])

    # generate a bunch of directions along which to test whether the distributions cross
    # result has dimensions [3,ndirections]
    xyz = uniform_sphere(parms["ME_stability_n_equatorial"])
    ndirections = xyz.shape[1]

    # Evaluate costheta relative to Fhat for each species
    costheta = np.zeros((nsims,2,parms["NF"],ndirections))
    for i in range(2):
        for j in range(parms["NF"]):
            costheta[:,i,j,:] = np.sum(xyz[None,:,:] * Fhat[:,:,i,j,None], axis=1)
    costheta[badlocs] = 0
    assert(np.all(np.abs(costheta)<1+1e-6))
    costheta[np.where(costheta>1)] =  1
    costheta[np.where(costheta<-1)] = -1

    # Evaluate the distribution for each species
    f = np.zeros((nsims,2,parms["NF"],ndirections))
    for i in range(2):
        for j in range(parms["NF"]):
            f[:,i,j,:] = distribution(F4i[:,3,i,j,None], Z[:,i,j,None], costheta[:,i,j,:])

    # lepton number is difference between neutrinos and antineutrinos
    # [nsims, NF, ndirections]
    lepton_number = f[:,0,:,:] - f[:,1,:,:]

    # calculate the G quantities
    NG = (parms["NF"]**2-parms["NF"])//2
    G = np.zeros((nsims,NG,ndirections))
    iG = 0
    for i in range(parms["NF"]):
        for j in range(i+1,parms["NF"]):
            G[:,iG,:] = lepton_number[:,i,:] - lepton_number[:,j,:]
            iG += 1
    assert(iG==NG)

    # check whether each G crosses zero
    crosses_zero = np.zeros((nsims,NG))
    for i in range(NG):
        minval = np.min(G[:,i,:], axis=1)
        maxval = np.max(G[:,i,:], axis=1)
        crosses_zero[:,i] = minval*maxval<0
    crosses_zero = np.any(crosses_zero, axis=1).astype(np.float32) # [nsims]

    # add extra unit dimension to compatibility with loss functions
    return crosses_zero[:,None]


#########################################################################################################
###########################################  X_from_F4    ###############################################
#########################################################################################################
#Given the a list of four fluxes, calculate the inputs to the neural network out of dot products of four fluxes with each other and the four velocity. The four-velocity is assumed to be timelike in an orthonormal tetrad.
#Args: F4 (torch.Tensor): Four-flux tensor. Indexed as [sim, xyzt, nu/nubar, flavor]
#Returns: torch.Tensor: Neural network input tensor. Indexed as [sim, iX]
def X_from_F4(parms, F4):
    index = 0
    nsims = F4.shape[0]
    NX = parms["NF"] * (1 + 2*parms["NF"])
    if parms["do_fdotu"]:
            NX += 2*parms["NF"]
    print("NX = ", NX)
    X = torch.zeros((nsims, NX), device=F4.device)
    F4_flat = F4.reshape((nsims, 4, 2*parms["NF"])) # [simulationIndex, xyzt, species]

    # calculate the total number density based on the t component of the four-vector
    # [sim]
    N = torch.sum(F4_flat[:,3,:], dim=1)

    # normalize F4 by the total number density
    # [sim, xyzt, 2*NF]
    F4_flat = F4_flat / N[:,None,None]

    # add the dot products of each species with each other species
    for a in range(2*parms["NF"]):
        for b in range(a,2*parms["NF"]):
            F1 = F4_flat[:,:,a]
            F2 = F4_flat[:,:,b]

            X[:,index] = dot4(F1,F2)
            index += 1

    # add the u dot products
    if parms["do_fdotu"]:
        u = torch.zeros((4), device=F4.device)
        u[3] = 1
        for a in range(2*parms["NF"]):
            X[:,index] = dot4(F4_flat[:,:,a], u[None,:])
            index += 1
    
    assert(index==NX)
    return X


################################################
############# Create a list of options #########
################################################
parms = {}

#TODO: Modify this to generate for data
#The number of points to generate
parms["n_generate"] = 50 #200000

#The number of flavors: should be 3
parms["NF"]= 3

#Use a GPU if available #

#parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"
#print("Using",parms["device"],"device")
#print(torch.cuda.get_device_name(0))
parms["device"] = "cpu"

parms["average_heavies_in_final_state"] = True
parms["do_fdotu"]= True
parms["generate_max_fluxfac"] = 0.95
parms["ME_stability_zero_weight"] = 10
parms["ME_stability_n_equatorial"] = 32


################################################
############# Run the test case ############### 
################################################
#Test cases for has_crossing function
# run test case
F4i = torch.zeros((1,4,2,3))
F4i[0,3,0,0] = 1
F4i[0,3,1,0] = 2
#print("should be false:", has_crossing(F4i.detach().numpy(), 3, 64))
print("should be false:", int(has_crossing(F4i.detach().numpy(), parms)))
assert(int(has_crossing(F4i.detach().numpy(), parms)) == 0)

F4i = torch.zeros((1,4,2,3))
F4i[0,3,0,0] = 1
F4i[0,3,1,0] = 1
F4i[0,0,0,0] =  0.5
F4i[0,0,1,0] = -0.5
#print("should be true:", has_crossing(F4i.detach().numpy(), 3, 64))
print("should be true:", int(has_crossing(F4i.detach().numpy(), parms)))
assert(int(has_crossing(F4i.detach().numpy(), parms)) == 1)



################################################
############# Generate the data ################
################################################
print("Generating data for stable_zerofluxfac..")
F4_zerofluxfac, unstable_zerofluxfac = generate_stable_F4_zerofluxfac(parms)
#print(F4_zerofluxfac.shape)
#print(F4_zerofluxfac[0,:,:,:])
#print(unstable_zerofluxfac.shape)
#print(unstable_zerofluxfac[0])

print("Generating data for stable_oneflavor..")
F4_oneflavor, unstable_oneflavor = generate_stable_F4_oneflavor(parms)
#print(F4_oneflavor.shape)
#print(F4_oneflavor[0,:,:,:])
#print(unstable_oneflavor.shape)
#print(unstable_oneflavor[0])

print("Generating random data..")
F4_random = generate_random_F4(parms)
#print(F4_random.shape)
#print(F4_random[0,:,:,:])
#If has_crossing returns true, unstable = 1 (i.e this point is unstable)
#If has_crossing returns false, unstable = 0 (i.e this point is stable)
unstable_random = has_crossing(F4_random.detach().numpy(), parms)
#print("unstable_random.shape", unstable_random.shape)
#print("unstable_random", unstable_random)



################################################
############# Convert from F4 to X #############
################################################
X_zerofluxfac = X_from_F4(parms, F4_zerofluxfac)
print("X_zerofluxfac.shape", X_zerofluxfac.shape)
print("X_zerofluxfac[0,:]", X_zerofluxfac[0,:])
print("unstable_zerofluxfac.shape", unstable_zerofluxfac.shape)
print("unstable_zerofluxfac[0]", unstable_zerofluxfac[0])

X_oneflavor = X_from_F4(parms, F4_oneflavor)
print("X_oneflavor.shape", X_oneflavor.shape)
print("X_oneflavor[0,:]", X_oneflavor[0,:])
print("unstable_oneflavor.shape", unstable_oneflavor.shape)
print("unstable_oneflavor[0]", unstable_oneflavor[0])

X_random = X_from_F4(parms, F4_random)
print("X_random.shape", X_random.shape)
print("X_random[0,:]", X_random[0,:])
print("unstable_random.shape", unstable_random.shape)
print("unstable_random[0]", unstable_random[0])



################################################
############# Save the data ####################
################################################
np.savez('train_data_stable_zerofluxfac.npz', X_zerofluxfac=X_zerofluxfac, unstable_zerofluxfac=unstable_zerofluxfac)
np.savez('train_data_stable_oneflavor.npz', X_oneflavor=X_oneflavor, unstable_oneflavor=unstable_oneflavor)
np.savez('train_data_random.npz', X_random=X_random, unstable_random=unstable_random)