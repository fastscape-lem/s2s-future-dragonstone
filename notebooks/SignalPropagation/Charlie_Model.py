import xsimlab as xs
import xarray as xr
import numpy as np
import numba as nb

#----------------

@xs.process
class UniformGrid1D:
    """Create 1D model grid with uniform spacing"""
    
    #grid parameters
    spacing = xs.variable(description="grid_spacing", static=True)
    length = xs.variable(description="grid total length", static=True)
    
    #x is an index variable, used for accessing the grid.
    x = xs.index(dims="x")
            
    #create the grid
    def initialize(self):
        self.x = np.arange(0, self.length, self.spacing)


#---------------

@xs.process
class ProfileZ:
    """Compute the evolution of the elevation (z) profile"""
    
    h_vars = xs.group("h_vars") #allows for multiple processes influencing; say diffusion and subsidence

    z = xs.variable(
        dims="x", intent="inout", description="surface elevation z", attrs={"units": "m"}
    )
    br = xs.variable(
        dims=[(), "x"], intent="in", description="bedrock_elevation", attrs={"units": "m"}
    )
    h = xs.variable(
        dims="x", intent="inout", description="sed_thickness", attrs={"units": "m"}
    )

    def run_step(self):
        #self._delta_br = sum((br for br in self.br_vars))
        self._delta_h = sum((h for h in self.h_vars))

    def finalize_step(self):
        #self.br += self._delta_br #update bedrock surface
        self.h += self._delta_h #update sediment thickness
        self.z = self.br + self.h #add sediment to bedrock to get topo elev.


#--------------

@nb.jit(nopython = True, cache = True)
def evolve_remaining_nodes(first_marine_node, z, slope, erosion, k_arr, s_crit, travel_dist, sea_level, deposition, qs, spacing, dh_dt, dh ,sed_porosity, dt,h):
    
    for i in range(first_marine_node + 1, len(z)): #iterate over each element of x ONLY IN THE MARINE
        
        if slope[i] <= 0: #this is the "regular," right-draining case
            deposition[i] = (qs[i-1] * (1 - np.minimum(1, np.power(slope[i] / s_crit, 2)))) / travel_dist
            erosion[i] = k_arr[i] * np.abs(slope[i])
            if z[i] > sea_level:
                deposition[i] = 0
        else: #this is the irregular, left-draining case
            deposition[i] = qs[i-1] / spacing#(self.qs[i-1] * (1 + np.minimum(1, np.power(self.slope[i] / self.s_crit, 2)))) / self.spacing#self.travel_dist #self.qs[i-1] / self.spacing
            erosion[i] = 0
            if z[i] > sea_level:
                deposition[i] = 0
        dh_dt[i] = (-erosion[i] + deposition[i]) / (1 - sed_porosity)

        dh[i] = dh_dt[i] * dt

        qs[i] = np.maximum(qs[i-1] + (erosion[i] - deposition[i]) * spacing, 0.)

        if -dh[i] > h[i]:
            dh[i] = -h[i]
            qs[i] = np.maximum(qs[i-1] + ((-dh[i] / dt) * (1 - sed_porosity)) * spacing, 0.)


#--------------------

@nb.jit(nopython = True, cache = True)      
def local_NewtonRaphson_scheme_for_compaction(nn,dh_compact,z0, h, sed_porosity, sed_porosity_depth_scale):
    #Newton-Raphson iteration  at every node
        for k in range(nn):
            fx = 1 #throwaway initial value to trigger "while" loop
            dfx = 1 #throwaway initial value to trigger "while" loop
            #check whether we're at the root
            if dh_compact[k] > 0 : #only apply compaction where deposition is happening
                while np.abs(fx / dfx) > 1e-6:
                    #calculate value of function at initial guess
                    fx = z0[k] - h[k] + sed_porosity * sed_porosity_depth_scale * (np.exp(-z0[k] / sed_porosity_depth_scale) - np.exp(-h[k] / sed_porosity_depth_scale)) - dh_compact[k]
        
                    #calculate derivative
                    dfx = 1 - sed_porosity * np.exp(-z0[k] / sed_porosity_depth_scale)
                    z0[k] = z0[k] - fx / dfx

            elif dh_compact[k] == 0: #no e or d
                z0[k] = h[k]
            else: #in the case where erosion happens, the sediment surface shouldn't rebound. 
                z0[k] = h[k] + dh_compact[k] / (1 - sed_porosity)


#--------------------

@xs.process
class ErosionDeposition:
    """Here's where the actual computation happenfirst_marine_nodecal diffusion

    """
    
    #get input parameters
    basin_width = xs.variable(description="basin width")
    sea_level = xs.variable(description="sea level")
    k_factor = xs.variable(description="k factor")
    k_depth_scale = xs.variable(description="k depth scale")
    travel_dist = xs.variable(description="marine_travel_distance")
    s_crit = xs.variable(description="marine_critical_slope")
    sed_porosity = xs.variable(description="marine_sediment_porosity")
    sed_porosity_depth_scale = xs.variable(description="marine_sediment_porosity_depth_scale")
    qs_in = xs.variable(description="sediment_flux_in")
    
    slope = xs.variable(
        dims="x", intent="out", description="topographic_slope", attrs={"units": "-"}
    )
    depth = xs.variable(
        dims="x", intent="out", description="depth", attrs={"units": "m"}
    )
    erosion = xs.variable(
        dims="x", intent="out", description="erosion", attrs={"units": "m/yr"}
    )
    deposition = xs.variable(
        dims="x", intent="out", description="deposition", attrs={"units": "m/yr"}
    )
    dh_dt = xs.variable(
        dims="x", intent="out", description="dh_dt", attrs={"units": "m/yr"}
    )
    qs = xs.variable(
        dims="x", intent="out", description="qs", attrs={"units": "m2/yr"}
    )
    #dbr = xs.variable(dims="x", intent="out", groups="br_vars")
    dh = xs.variable(dims="x", intent="out", groups="h_vars")
    
    spacing = xs.foreign(UniformGrid1D, "spacing")
    #x = xs.foreign(UniformGrid1D, "x")
    z = xs.foreign(ProfileZ, "z")
    br = xs.foreign(ProfileZ, "br")
    h = xs.foreign(ProfileZ, "h")
    
    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        k_factor_real = np.power(10, self.k_factor)
        self.erosion = np.repeat(0., len(self.z))
        self.deposition = np.repeat(0., len(self.z))
        self.dh_dt = np.repeat(0., len(self.z))
        self.qs = np.repeat(0., len(self.z))
        self.dh = np.repeat(0., len(self.z))
        
        #divide Qs_in by basin width to get qs_in
        qs_in = self.qs_in / self.basin_width
        
        #calculate topographic slope
        self.slope = np.append(np.diff(self.z) / self.spacing, 0)
        #calculate depth below water
        self.depth = np.maximum(self.sea_level - self.z, 0)
        
        #calculate k array
        k_arr = k_factor_real * np.exp(-self.depth / self.k_depth_scale)
        
        #impose hard basin floor
        k_arr[self.z <= (self.br + 0.001)] = 0
        
        #find first marine node
        marine_or_terrestrial = self.z <= self.sea_level #boolean: true is marine
        first_marine_node = np.argmax(marine_or_terrestrial) #finds the first true
        
        #evolve first marine node
        if self.slope[first_marine_node] <= 0: #this is the "regular," right-draining case
            self.erosion[first_marine_node] = k_arr[first_marine_node] * np.abs(self.slope[first_marine_node])
            self.deposition[first_marine_node] = (qs_in * (1 - np.minimum(1, np.power(self.slope[first_marine_node] / self.s_crit, 2)))) / self.travel_dist

        else:  #this is the irregular, left-draining case
            self.erosion[first_marine_node] = 0 #because slope = 0
            self.deposition[first_marine_node] = qs_in / self.spacing #because slope = 0 #self.qs_in / self.travel_dist
        self.dh_dt[first_marine_node] = (-self.erosion[first_marine_node] + self.deposition[first_marine_node]) / (1 - self.sed_porosity)
        self.dh[first_marine_node] = self.dh_dt[first_marine_node] * dt
        self.qs[first_marine_node] = np.maximum(qs_in + (self.erosion[first_marine_node] - self.deposition[first_marine_node]) * self.spacing, 0.)
        if -self.dh[first_marine_node] > self.h[first_marine_node]:
            self.dh[first_marine_node] = -self.h[first_marine_node]
            self.qs[first_marine_node] = np.maximum(qs_in + ((-self.dh[first_marine_node] / dt) * (1 - self.sed_porosity)) * self.spacing, 0.)
        
        
        
        #evolve remaining marine nodes
        evolve_remaining_nodes(first_marine_node, self.z, self.slope, self.erosion, k_arr, self.s_crit, 
                               self.travel_dist, self.sea_level, self.deposition, self.qs, self.spacing,
                               self.dh_dt, self.dh ,self.sed_porosity, dt, self.h)  
        
        #compact sediment
        #calculate the thickness after compaction, z0; dh is the thickness of new deposited solid sediment
        dh_compact = self.dh * (1 - self.sed_porosity)
    
        
        #compaction routine
        #def compaction(porosity, porosity_depth_scale, nn, dh, zi):
        nn = len(self.z)
        z0 = np.zeros(nn)
        #set initial guess for z0:
        z0[:] = self.h[:]
        
        local_NewtonRaphson_scheme_for_compaction(nn,dh_compact,z0, self.h, self.sed_porosity, self.sed_porosity_depth_scale)
  
        #here, have a chance to set the final dh by differencing new h (z0) and old h (h)
        self.dh[:] = z0[:] - self.h[:]


#--------------------

@xs.process
class InitBasinGeom:
    """
    Give initial basin elevation field as a function of x:
    z = exp(- (x - a) / b) + c
    """
    init_br = xs.variable(dims="x", description="shift parameter", static=True)
    
    x = xs.foreign(UniformGrid1D, "x")
    z = xs.foreign(ProfileZ, "z", intent="out")
    h = xs.foreign(ProfileZ, "h", intent="out")

    def initialize(self):
        self.h = np.zeros(len(self.x)) #initial sediment thickness is 0
        self.z = np.zeros(len(self.x)) + self.init_br 

#-------------------

marine = xs.Model(
    {
        "grid": UniformGrid1D,
        "profile": ProfileZ,
        "init": InitBasinGeom,
        "erode": ErosionDeposition
    }
)