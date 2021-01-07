import numpy as np

def find_largest_river (area, don, ndon):
    """To find the nodes defining the largest/longest river in a landscape
    generated by FastScape.

    This routine uses the area and donor information computed by FastScape
    to trace the geometry of the largest river in the landscape. For this
    it first finds the point with the largest area.
    It then finds recursively the one of its donors that has the largest
    drainage area. The recursion stops when a node has no donors.

    Parameters
    ----------
    area : float
        1D array containing the drainage area of each node of the landscape
        as computed by FastScape.
    don : int
        1D array containing the list of donors of each node. Has dimensions
        of [8,nnode] and for node ij, don[ij, :ndon[ij]] contains the list
        of ndon[ij] donors to ij.
    ndon : int
        1D array containing the number of donors (for each node) as
        computed by FastScape.

    Returns
    -------
    list
        1D list of nodes defining the largest/longest river

    """

    ij = np.argmax(area)
    largest_river=[]
    largest_river.append(ij)
    while ndon[ij]>0:
        ndonij = ndon[ij]
        donij = don[ij, :ndonij]
        ij = donij[np.argmax(area[donij])]
        largest_river.append(ij)

    return largest_river

def find_s_coordinate (river, x, y):
    """To find the s-coordinate along a river path.

    This routine uses the x- and y-corrdinates of nodes on a landscape
    to compute the s-coordinate along a river path.

    Parameters
    ----------
    river : int
        1D list containing the node numbers of a river/path in the 
        landscape
    x : float
        1D array containing x-coordinates of nodes on the landscape
        (length nx)
    y : float
        1D array containing y-coordinates of nodes on the landscape
        (length ny)

    Returns
    -------
    float
        s-coordinate along river path (same length as river)

    """
    
    s = np.zeros(len(river))
    Y, X = np.meshgrid(y,x)
    xx = X.flatten()
    yy = Y.flatten()
    dx0 = x[1]
    dy0 = y[1]
    for i in range(1,len(river)):
        dx = min(abs(xx[river[i]] - xx[river[i-1]]), dx0)
        dy = min(abs(yy[river[i]] - yy[river[i-1]]), dy0)
        ds = np.sqrt(dx**2 + dy**2)
        s[i] = s[i-1] + ds
        
    return s

def find_slopes (h, x, y, rec):
    """Compute slopes as defined in the FastScape algorithm.

    This function computes the slope between each node and its receiver.
    Note that care must be taken to deal with the boundary conditions or
    with the nodes that do not have a receiver.

    Parameters
    ----------
    h : float
        1D array containing landscape height
    x : float
        1D array containing x-coordinates of nodes on the landscape
        (length nx)
    y : float
        1D array containing y-coordinates of nodes on the landscape
        (length ny)
    rec : int
        1D array containing the list of receivers of each node. rec[ij]
        contains the list of receivers of ij.

    Returns
    -------
    float
        1D array containing the slope

    """
    
    s = np.zeros_like(h)
    Y, X = np.meshgrid(y,x)
    xx= X.flatten()
    yy = Y.flatten()
    dx0 = x[1]
    dy0 = y[1]
    for i in range(1,len(h)):
        dx = min(abs(xx[i] - xx[rec[i]]), dx0)
        dy = min(abs(yy[i] - yy[rec[i]]), dy0)
        s[i] = 0
        ds = dx**2 + dy**2
        if ds>0:
            ds = np.sqrt(ds)
            s[i] = (h[i] - h[rec[i]])/ds
        
    return s

def find_chi (river, s, area, mn):
    """Compute chi-parameter along a river path.

    This function computes the chi-parameter along a river path.
    Chi is defined as the integral (from base level) of 1/A^mn
    along the river path.

    Parameters
    ----------
    river : int
        List of nodes defining the river
    s : float
        s-coordinates of nodes along the river
    area : float
        1D array of area of all nodes on the landscape
    mn : float
        value of the product m*n, where m is the area exponent
        and n the slope exponent in the stream power law (SPL)

    Returns
    -------
    float
        chi coordinate/parameter along the river path

    """
    
    chi = np.zeros_like(s)
    for i in range(1,len(chi)):
        chi[i] = chi[i-1] + 2*(s[i] - s[i-1])*(area[river[i]] + area[river[i-1]])**(-mn)
    
    return chi