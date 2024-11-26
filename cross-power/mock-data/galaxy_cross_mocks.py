import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import sys
import os
from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.typing import List
from pathlib import Path
from pprint import pprint

parser = ArgumentParser()

parser.add_argument(
    "--meerpower-path",
    type=str,
    help="Path to meerpower repository."
)
parser.add_argument(
    "--survey",
    type=str,
    default="2021",
    help="One of either '2019' or '2021'."
)
parser.add_argument(
    "--filepath-HI",
    type=str,
    help="Path to an HI map file in FITS format."
)
parser.add_argument(
    "--gal-cat",
    type=str,
    default="gama",
    help="Galaxy catalog name.  Can be one of 'gama', 'wigglez', or 'cmass'."
)
parser.add_argument(
    "--filepath-g",
    type=str,
    help="Path to a galaxy catalog file in txt format (wigglez) or FITS "
         "format (gama or cmass)."
)
parser.add_argument(
    "--doMock",
    action="store_true",
    help="Use mock data for consistency checks."
)
parser.add_argument(
    "--mockindx",
    type=int,
    help="Mock file index.  If you pass '--doMock' and no value for "
         "'--mockindx' is passed, a random index will be used."
)
parser.add_argument(
    "--gamma",
    type=float,
    default=1.4,
    help="Resmoothing factor.  Defaults to 1.4."
)
parser.add_argument(
    "--Nmocks",
    type=int,
    help="Number of mock files to loop over for calculating uncertainties."
)
parser.add_argument(
    "--mockfilepath-HI",
    type=str,
    help="Path and base name for a set of numpy-readable, indexed mock HI "
         "files.  For example, if the mock data are stored in "
         "'/path/to/mocks_{index}.npy', you would pass "
         "'--mockfilepath-HI /path/to/mocks' and exclude the '_{index}.npy' "
         "suffix."
)
parser.add_argument(
    "--tukey-alpha",
    type=float,
    default=0.1,
    help="Tukey window shape parameter value."
)
parser.add_argument(
    "--grid-seed",
    type=int,
    default=834515,
    help="Random seed for the regridding step.  Fixes the random locations "
         "of the sampling particles within pixels."
)
parser.add_argument(
    "--out-dir",
    type=str,
    help="Path to a directory for output files."
)
parser.add_argument(
    "--config",
    action=ActionConfigFile
)

args = parser.parse_args()
print("Command Line Arguments:")
pprint(args.__dict__)
print()

sys.path.insert(1, (Path(args.meerpower_path) / 'meerpower').as_posix())
import Init

def RunPipeline(
    survey,
    filepath_HI,
    mockfilepath_HI,
    gal_cat,
    filepath_g,
    gamma=1.4,
    kcuts=[0.052, 0.031, 0.175, None],
    Nmocks=499,
    doMock=False,
    mockindx=None,
    out_dir="./",
    tukey_alpha=0.1,
    grid_seed=834515,
    conf_interval=95
):
    '''
    Use for looping over full pipeline with different choices of inputs for purposes
    of transfer function building. Input choices from below:
    # survey = '2019' or '2021'
    # filepath_HI = Path to a HI map file in FITS format
    # mockfilepath_HI = Path to a numpy-readable file containing mock HI data
    # gal_cat = 'wigglez', 'cmass', or 'gama'
    # filepath_g = Path to a galaxy catalog file in txt format (wigglez) or FITS format (gama or cmass).
    # gamma = float or None (resmoothing parameter)
    # kcuts = [kperpmin,kparamin,kperpmax,kparamax] or None (exclude areas of k-space from spherical average)]
    # Nmocks = Number of mock datasets to loop over
    # doMock = Use mock data for consistency checks
    # mockindx = Mock file index.  If None (default), choose a random index
    # out_dir = Path to the meerpower
    # tukey_alpha = Tukey window shape parameter
    # grid_seed = Random seed for gridding operations
    # conf_interval = Confidence interval to compute over Nmocks axis if Nmocks > 1
    '''
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)

    # Load data and run some pre-processing steps:
    if survey=='2019':
        numin,numax = 971,1023.2
    if survey=='2021':
        numin,numax = 971,1023.8 # default setting in Init.ReadIn()
    MKmap,w_HI,W_HI,counts_HI,dims,ra,dec,nu,wproj = Init.ReadIn(filepath_HI,numin=numin,numax=numax)
    if doMock==True:
        if mockindx is None:
            mockindx = np.random.randint(100)
        MKmap_mock = np.load(mockfilepath_HI + '_%s.npy'%mockindx)
    nx,ny,nz = np.shape(MKmap)

    # Initialise some fiducial cosmology and survey parameters:
    import cosmo
    nu_21cm = 1420.405751 #MHz
    zeff = (nu_21cm/np.median(nu)) - 1 # Effective redshift - defined as redshift of median frequency
    zmin = (nu_21cm/np.max(nu)) - 1 # Minimum redshift of band
    zmax = (nu_21cm/np.min(nu)) - 1 # Maximum redshift of band
    cosmo.SetCosmology(builtincosmo='Planck18',z=zeff,UseCLASS=True)
    Pmod = cosmo.GetModelPk(zeff,kmax=25,UseCLASS=True) # high-kmax needed for large k-modes in NGP alisasing correction
    f = cosmo.f(zeff)
    sig_v = 0
    b_HI = 1.5
    OmegaHIbHI = 0.85e-3 # MKxWiggleZ constraint
    OmegaHI = OmegaHIbHI/b_HI
    import HItools
    import telescope
    Tbar = HItools.Tbar(zeff,OmegaHI)
    r = 0.9 # cross-correlation coefficient
    D_dish = 13.5 # Dish-diameter [metres]
    theta_FWHM,R_beam = telescope.getbeampars(D_dish,np.median(nu))

    # Galaxy catalog parameters
    if gal_cat=='wigglez': # obtained from min/max of wigglez catalogue
        ramin_gal,ramax_gal = 152.906631, 172.099625
        decmin_gal,decmax_gal = -1.527391, 8.094599
    if gal_cat=='gama':
        ramin_gal,ramax_gal = 339,351
        decmin_gal,decmax_gal = -35,-30

    # Read-in overlapping galaxy survey:
    from astropy.io import fits
    if survey=='2019':
        if gal_cat=='wigglez':
            if doMock==False: # Read-in WiggleZ galaxies (provided by Laura):
                galcat = np.genfromtxt(filepath_g, skip_header=1)
                ra_g,dec_g,z_g = galcat[:,0],galcat[:,1],galcat[:,2]
            if doMock==True: ra_g,dec_g,z_g = np.load(mockfilepath_g + '_%s.npy'%mockindx)
            z_Lband = (z_g>zmin) & (z_g<zmax) # Cut redshift to MeerKAT IM range:
            ra_g,dec_g,z_g = ra_g[z_Lband],dec_g[z_Lband],z_g[z_Lband]
        if gal_cat=='cmass':
            if doMock==False: # Read-in BOSS-CMASS galaxies (in Yi-Chao's ilifu folder - also publically available from: https://data.sdss.org/sas/dr12/boss/lss):
                hdu = fits.open(filepath_g)
                ra_g,dec_g,z_g = hdu[1].data['RA'],hdu[1].data['DEC'],hdu[1].data['Z']
            if doMock==True: ra_g,dec_g,z_g = np.load(mockfilepath_g + '_%s.npy'%mockindx)
            ra_g,dec_g,z_g = Init.pre_process_2019Lband_CMASS_galaxies(ra_g,dec_g,z_g,ra,dec,zmin,zmax,W_HI)

    if survey=='2021':
        if doMock==False: # Read-in GAMA galaxies:
            hdu = fits.open(filepath_g)
            ra_g,dec_g,z_g = hdu[1].data['RA'],hdu[1].data['DEC'],hdu[1].data['Z']
        if doMock==True: ra_g,dec_g,z_g = np.load(mockfilepath_g + '_%s.npy'%mockindx)
        # Remove galaxies outside bulk GAMA footprint so they don't bias the simple binary selection function
        GAMAcutmask = (ra_g>ramin_gal) & (ra_g<ramax_gal) & (dec_g>decmin_gal) & (dec_g<decmax_gal) & (z_g>zmin) & (z_g<zmax)
        ra_g,dec_g,z_g = ra_g[GAMAcutmask],dec_g[GAMAcutmask],z_g[GAMAcutmask]

    print('Number of overlapping ', gal_cat,' galaxies: ', str(len(ra_g)), end='\n\n')

    # Assign galaxy bias:
    if gal_cat=='wigglez': b_g = np.sqrt(0.83) # for WiggleZ at z_eff=0.41 - from https://arxiv.org/pdf/1104.2948.pdf [pg.9 rhs second quantity]
    if gal_cat=='cmass':b_g = 1.85 # Mentioned in https://arxiv.org/pdf/1607.03155.pdf
    if gal_cat=='gama': b_g = 2.35 # tuned by eye in GAMA auto-corr
    b_g = 1.9  # Change made to match the value in the notebook ./galaxy_cross.ipynb

    grid_galaxies = True  # boolean to avoid re-gridding galaxies every iteration
    create_arrays = True  # boolean to avoid overwriting containers for each iteration
    for i_mock in range(Nmocks):
        MKmap_mock = np.load(mockfilepath_HI + f'_{i_mock}.npy')
        MKmap = MKmap_mock  # Replace map with mock data

        ### Remove incomplete LoS pixels from maps:
        MKmap,w_HI,W_HI,counts_HI = Init.FilterIncompleteLoS(MKmap,w_HI,W_HI,counts_HI)

        ### IM weights (averaging of counts along LoS so not to increase rank of the map for FG cleaning):
        w_HI = np.repeat(np.mean(counts_HI,2)[:, :, np.newaxis], nz, axis=2)

        ### Map resmoothing:
        MKmap_unsmoothed = np.copy(MKmap)
        if gamma is not None:
            w_HI_orig = np.copy(w_HI)
            MKmap,w_HI = telescope.weighted_reconvolve(MKmap,w_HI_orig,W_HI,ra,dec,nu,D_dish,gamma=gamma)
            if doMock==True: MKmap_mock,null = telescope.weighted_reconvolve(MKmap_mock,w_HI_orig,W_HI,ra,dec,nu,D_dish,gamma=gamma)

        ### Trim map edges:
        doTrim = True
        if doTrim==True:
            if survey=='2019':
                raminMK,ramaxMK = 149,190
                decminMK,decmaxMK = -5,20

            if survey=='2021':
                raminMK,ramaxMK = 334,357
                decminMK,decmaxMK = np.min(dec[np.mean(W_HI,2)>0]),np.max(dec[np.mean(W_HI,2)>0])
            ### Before trimming map, show contour of trimmed area:
            MKmap_untrim,W_HI_untrim = np.copy(MKmap),np.copy(W_HI)
            MKmap,w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,MKmap,w_HI,W_HI,counts_HI,ramin=raminMK,ramax=ramaxMK,decmin=decminMK,decmax=decmaxMK)

            if survey=='2019':
                cornercut_lim = 146 # set low to turn off
                cornercut = ra - dec < cornercut_lim
                MKmap[cornercut],w_HI[cornercut],W_HI[cornercut],counts_HI[cornercut] = 0,0,0,0

        # Spectral analysis for possible frequency channel flagging:
        #Also remove some corners/extreme temp values

        MKmap_flag,w_HI_flag,W_HI_flag = np.copy(MKmap),np.copy(w_HI),np.copy(W_HI)

        if survey=='2019':
            extreme_temp_LoS = np.zeros(np.shape(ra))
            extreme_temp_LoS[MKmap[:,:,0]>3530] = 1
            extreme_temp_LoS[MKmap[:,:,0]<3100] = 1
            MKmap_flag[extreme_temp_LoS==1] = 0
            w_HI_flag[extreme_temp_LoS==1] = 0
            W_HI_flag[extreme_temp_LoS==1] = 0

        import model
        nra,ndec = np.shape(ra)
        offsets = np.zeros((nra,ndec,len(nu)))
        for i in range(nra):
            for j in range(ndec):
                if W_HI_flag[i,j,0]==0: continue
                poly = model.FitPolynomial(nu,MKmap_flag[i,j,:],n=2)
                offsets[i,j,:] = np.abs((MKmap_flag[i,j,:] - poly)/MKmap_flag[i,j,:])
        offsets = 100*np.mean(offsets,axis=(0,1))

        if survey=='2019': offsetcut = 0.029 # Set to zero for no additional flagging
        #if survey=='2019': offsetcut = None # Set to None for no additional flagging
        if survey=='2021': offsetcut = None

        if offsetcut is None: flagindx = []
        else: flagindx = np.where(offsets>offsetcut)[0]

        flags = np.full(nz,False)
        flags[flagindx] = True

        MKmap_flag[:,:,flags] = 0
        w_HI_flag[:,:,flags] = 0
        W_HI_flag[:,:,flags] = 0

        # Foreground cleaning would normally be done here, but we are using mock HI
        # data which contain no foregrounds so no cleaning is required.
        MKmap,w_HI,W_HI = np.copy(MKmap_flag),np.copy(w_HI_flag),np.copy(W_HI_flag) # Propagate flagged maps for rest of analysis
        MKmap_clean = MKmap.copy()

        W_HI_untrim,w_HI_untrim = np.copy(W_HI),np.copy(w_HI)
        if gal_cat=='gama':
            w_HI,W_HI,counts_HI = Init.MapTrim(ra,dec,w_HI,W_HI,counts_HI,ramin=ramin_gal,ramax=ramax_gal,decmin=decmin_gal,decmax=decmax_gal)

        # Gridding maps and galaxies to Cartesian field:
        import grid # use this for going from (ra,dec,freq)->(x,y,z) Cartesian-comoving grid
        cell2vox_factor = 1.5 # increase for lower resolution FFT Cartesian grid
        Np = 5 # number of Monte-Carlo sampling particles per map voxel used in regridding
        window = 'ngp'
        compensate = True
        interlace = False
        nxmap,nymap,nzmap = np.shape(MKmap)
        ndim_rg = int(nxmap/cell2vox_factor),int(nymap/cell2vox_factor),int(nzmap/cell2vox_factor)
        nzcell2vox = int(nzmap/cell2vox_factor)
        if nzcell2vox % 2 != 0: nzcell2vox += 1 # Ensure z-dimension is even for FFT purposes
        ndim_rg = int(nxmap/cell2vox_factor),int(nymap/cell2vox_factor),nzcell2vox
        dims_rg,dims0_rg = grid.comoving_dims(ra,dec,nu,wproj,ndim_rg,W=W_HI_untrim,dobuffer=True) # dimensions of Cartesian grid for FFT
        lx,ly,lz,nx_rg,ny_rg,nz_rg = dims_rg

        # Regrid cleaned map, IM mask and weights to Cartesian field:
        ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=MKmap_clean,W=W_HI,Np=Np,seed=grid_seed)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
        MKmap_clean_rg,null,null = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window,compensate,interlace,verbose=False)
        ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=W_HI,W=W_HI,Np=Np,seed=grid_seed)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
        W_HI_rg,null,null = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)
        ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=w_HI,W=W_HI,Np=Np,seed=grid_seed)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),ramean_arr=ra,decmean_arr=dec,doTile=False)
        w_HI_rg = grid.mesh(xp,yp,zp,pixvals,dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]

        if grid_galaxies:
            # Grid galaxies straight to Cartesian field:
            xp,yp,zp = grid.SkyCoordtoCartesian(ra_g,dec_g,z_g,ramean_arr=ra,decmean_arr=dec,doTile=False)
            n_g_rg = grid.mesh(xp,yp,zp,dims=dims0_rg,window=window,compensate=compensate,interlace=interlace,verbose=False)[0]

        # Construct galaxy selection function:
        if survey=='2019':
            if gal_cat=='wigglez': # grid WiggleZ randoms straight to Cartesian field for survey selection:
                BuildSelFunc = False
                if BuildSelFunc==True:
                    nrand = 1000 # number of WiggleZ random catalogues to use in selection function (max is 1000)
                    W_g_rg = np.zeros(np.shape(n_g_rg))
                    for i in range(1,nrand):
                        plot.ProgressBar(i,nrand)
                        # galcat = np.genfromtxt( '/users/scunnington/MeerKAT/LauraShare/wigglez_reg11hrS_z0pt30_0pt50/reg11rand%s.dat' %'{:04d}'.format(i), skip_header=1)
                        galcat = np.genfromtxt(filepath_g.replace('data.dat', 'rand%s.dat' %'{:04d}'.format(i)), skip_header=1)
                        ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]
                        z_Lband = (z_g_rand>zmin) & (z_g_rand<zmax) # Cut redshift to MeerKAT IM range:
                        ra_g_rand,dec_g_rand,z_g_rand = ra_g_rand[z_Lband],dec_g_rand[z_Lband],z_g_rand[z_Lband]
                        xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
                        W_g_rg += grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]
                    W_g_rg /= nrand
                W_g_rg = np.load('/idia/projects/hi_im/meerpower/2019Lband/wigglez/data/wiggleZ_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)

            if gal_cat=='cmass':
            # Data obtained from untarrting DR12 file at: https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-DR12NGC-COMPSAM_V6C.tar.gz
                BuildSelFunc = False
                if BuildSelFunc==True:
                    nrand = 2048 # number of WiggleZ random catalogues to use in selection function (max is 1000)
                    W_g_rg = np.zeros(np.shape(n_g_rg))
                    for i in range(1,nrand):
                        plot.ProgressBar(i,nrand)
                        galcat = np.genfromtxt( '/idia/projects/hi_im/meerpower/2019Lband/cmass/sdss/Patchy-Mocks-DR12NGC-COMPSAM_V6C_%s.dat' %'{:04d}'.format(i+1), skip_header=1)
                        ra_g_rand,dec_g_rand,z_g_rand = galcat[:,0],galcat[:,1],galcat[:,2]
                        ra_g_rand,dec_g_rand,z_g_rand = Init.pre_process_2019Lband_CMASS_galaxies(ra_g_rand,dec_g_rand,z_g_rand,ra,dec,zmin,zmax,W_HI)
                        xp,yp,zp = grid.SkyCoordtoCartesian(ra_g_rand,dec_g_rand,z_g_rand,ramean_arr=ra,decmean_arr=dec,doTile=False)
                        W_g_rg += grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)[0]
                    W_g_rg /= nrand
                W_g_rg = np.load('/idia/projects/hi_im/meerpower/2019Lband/cmass/data/cmass_stackedrandoms_cell2voxfactor=%s.npy'%cell2vox_factor)

        if survey=='2021': # grid uncut pixels to obtain binary mask in comoving space in absence of GAMA mocks for survey selection:
            ra_p,dec_p,nu_p = grid.SkyPixelParticles(ra,dec,nu,wproj,Np=Np,seed=grid_seed)
            GAMAcutmask = (ra_p>ramin_gal) & (ra_p<ramax_gal) & (dec_p>decmin_gal) & (dec_p<decmax_gal)
            xp,yp,zp = grid.SkyCoordtoCartesian(ra_p[GAMAcutmask],dec_p[GAMAcutmask],HItools.Freq2Red(nu_p[GAMAcutmask]),ramean_arr=ra,decmean_arr=dec,doTile=False)
            null,W_g_rg,null = grid.mesh(xp,yp,zp,dims=dims0_rg,window='ngp',compensate=False,interlace=False,verbose=False)

        # Calculate FKP weigts:
        w_g_rg = np.copy(W_g_rg)

        MKmap_clean_rg_notaper,w_HI_rg_notaper,W_HI_rg_notaper = np.copy(MKmap_clean_rg),np.copy(w_HI_rg),np.copy(W_HI_rg)
        n_g_rg_notaper,w_g_rg_notaper,W_g_rg_notaper = np.copy(n_g_rg),np.copy(w_g_rg),np.copy(W_g_rg)

        # Footprint tapering/apodisation:
        ### Chose no taper:
        #taper_HI,taper_g = 1,1

        ### Chose to use Blackman window function along z direction as taper:
        blackman = np.reshape( np.tile(np.blackman(nz_rg), (nx_rg,ny_rg)) , (nx_rg,ny_rg,nz_rg) ) # Blackman function along every LoS
        tukey = np.reshape( np.tile(signal.windows.tukey(nz_rg, alpha=tukey_alpha), (nx_rg,ny_rg)) , (nx_rg,ny_rg,nz_rg) ) # Blackman function along every LoS


        #taper_HI = blackman
        #taper_g = blackman
        taper_HI = tukey
        #taper_g = tukey
        #taper_HI = 1
        taper_g = 1


        # Multiply tapering windows by all fields that undergo Fourier transforms:
        #MKmap_clean_rg,w_HI_rg,W_HI_rg = taper_HI*MKmap_clean_rg_notaper,taper_HI*w_HI_rg_notaper,taper_HI*W_HI_rg_notaper
        #n_g_rg,W_g_rg,w_g_rg = taper_g*n_g_rg_notaper,taper_g*W_g_rg_notaper,taper_g*w_g_rg_notaper
        w_HI_rg = taper_HI*w_HI_rg_notaper
        w_g_rg = taper_g*w_g_rg_notaper

        # Power spectrum measurement and modelling (without signal loss correction):
        import power
        nkbin = 16
        kmin,kmax = 0.07,0.3
        kbins = np.linspace(kmin,kmax,nkbin+1) # k-bin edges [using linear binning]

        sig_v = 200  # Changed to match the notebook ./galaxy_cross.ipynb
        dpix = 0.3
        d_c = cosmo.d_com(HItools.Freq2Red(np.min(nu)))
        s_pix = d_c * np.radians(dpix)
        s_para = np.mean( cosmo.d_com(HItools.Freq2Red(nu[:-1])) - cosmo.d_com(HItools.Freq2Red(nu[1:])) )

        if grid_galaxies:
            ### Galaxy Auto-power (can use to constrain bias and use for analytical errors):
            Pk_g,k,nmodes = power.Pk(n_g_rg,n_g_rg,dims_rg,kbins,corrtype='Galauto',w1=w_g_rg,w2=w_g_rg,W1=W_g_rg,W2=W_g_rg)
            grid_galaxies = False
        W_g_rg /= np.max(W_g_rg)
        Vfrac = np.sum(W_g_rg)/(nx_rg*ny_rg*nz_rg)
        nbar = np.sum(n_g_rg)/(lx*ly*lz*Vfrac) # Calculate number density inside survey footprint
        P_SN = np.ones(len(k))*1/nbar # approximate shot-noise for errors (already subtracted in Pk estimator)

        # Calculate power specs (to get k's for TF):
        Pk_gHI,k,nmodes = power.Pk(MKmap_clean_rg,n_g_rg,dims_rg,kbins,corrtype='Cross',w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts)
        if gamma is not None: 
            theta_FWHM_max,R_beam_max = telescope.getbeampars(D_dish,np.min(nu))
            R_beam_gam = R_beam_max * np.sqrt(gamma)
        else: R_beam_gam = np.copy(R_beam)
        pkmod,k = model.PkMod(Pmod,dims_rg,kbins,b_HI,b_g,f,sig_v,Tbar1=Tbar,Tbar2=1,r=r,R_beam1=R_beam_gam,R_beam2=0,w1=w_HI_rg,w2=w_g_rg,W1=W_HI_rg,W2=W_g_rg,kcuts=kcuts,s_pix1=s_pix,s_para1=s_para,interpkbins=True,MatterRSDs=True,gridinterp=True)[0:2]
        Pk_HI,k,nmodes = power.Pk(MKmap_clean_rg,MKmap_clean_rg,dims_rg,kbins,corrtype='HIauto',w1=w_HI_rg,w2=w_HI_rg,W1=W_HI_rg,W2=W_HI_rg)
        sig_err = 1/np.sqrt(2*nmodes) * np.sqrt( Pk_gHI**2 + Pk_HI*( Pk_g + P_SN ) ) # Error estimate

        if create_arrays:
            Pk_gHI_all = np.zeros((Nmocks, k.size), dtype=Pk_gHI.dtype)
            create_arrays = False

        Pk_gHI_all[i_mock] = Pk_gHI

    # Write results to disk
    out_path = out_dir / f'Pk_gHI_Nmocks{Nmocks}.npy'
    print('Writing power spectra to', out_path)
    np.save(out_path, Pk_gHI_all)

    out_path = out_dir / f'k.npy'
    print('Writing k values to', out_path)
    np.save(out_path, k)

    # Plot results and write to disk
    if Nmocks > 1:
        # Calculate confidence interval
        percentile = 50 + conf_interval/2
        Pk_gHI_lbound = np.percentile(Pk_gHI_all, 100-percentile, axis=0)
        Pk_gHI_ubound = np.percentile(Pk_gHI_all, percentile, axis=0)

    fig, ax = plt.subplots()
    if Nmocks == 1:
        ax.plot(k, Pk_gHI_all[0])
    else:
        yerr = [
            Pk_gHI_all.mean(axis=0) - Pk_gHI_lbound,
            Pk_gHI_ubound - Pk_gHI_all.mean(axis=0)
        ]
        ax.errorbar(
            k,
            Pk_gHI_all.mean(axis=0),
            yerr=yerr,
            ls='',
            capsize=3,
            marker='o',
            label=rf'Sample Mean $\pm$ {conf_interval:.0f}% conf.'
        )
        ax.legend()
    ax.grid()
    ax.set_ylabel(r'$P(k)_{\rm{g},\rm{HI}}$ [mK $h^{-1}$ Mpc]')
    ax.set_xlabel(r'$k$ $[h\ \rm{Mpc}^{-1}]$')
    ax.set_title(f'Nmocks = {Nmocks}')

    out_path = out_dir / f'Pk_gHI_Nmocks{Nmocks}.pdf'
    print('Writing figure to', out_path)
    fig.savefig(out_path)
    

# [kperpmin,kparamin,kperpmax,kparamax] (exclude areas of k-space from spherical average)
# Values chosen by MeerKLASS and left in place
kcuts = [0.052,0.031,0.175,None]

RunPipeline(
    args.survey,
    args.filepath_HI,
    args.mockfilepath_HI,
    args.gal_cat,
    args.filepath_g,
    gamma=args.gamma,
    kcuts=kcuts,
    doMock=args.doMock,
    mockindx=args.mockindx,
    out_dir=args.out_dir,
    tukey_alpha=args.tukey_alpha,
    grid_seed=args.grid_seed,
    Nmocks=args.Nmocks
)
