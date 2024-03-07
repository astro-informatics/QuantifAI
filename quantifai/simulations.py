"""
Installation instructions for mac M1/2/3

CONDA_SUBDIR=osx-64 conda create -n env_x86_py36 python=3.6
conda activate env_x86_py36


# Install CASA software with pip
# See https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Modular-Packages
# Be careful with the versions (the ones cited in teh docs are often incompatible)

# Install simms package
git clone https://github.com/ratt-ru/simms
cd simms
pip install .


# Commands used to generate the Meerkat datasets in the repository
simms -dir J2000,13h18m54.86s,-15d36m04.25s -os -T meerkat -dt 240 -st 1 -nc 10 -f0 1400MHz -df 10MHz -pl XX YY -n meerkat_simulation_1h.ms
simms -dir J2000,13h18m54.86s,-15d36m04.25s -os -T meerkat -dt 240 -st 2 -nc 10 -f0 1400MHz -df 10MHz -pl XX YY -n meerkat_simulation_2h.ms
simms -dir J2000,13h18m54.86s,-15d36m04.25s -os -T meerkat -dt 240 -st 4 -nc 10 -f0 1400MHz -df 10MHz -pl XX YY -n meerkat_simulation_4h.ms
simms -dir J2000,13h18m54.86s,-15d36m04.25s -os -T meerkat -dt 240 -st 8 -nc 10 -f0 1400MHz -df 10MHz -pl XX YY -n meerkat_simulation_8h.ms

"""


import numpy as np
import os
import matplotlib.pyplot as plt



def generate_random_empty_ms(
        msname='empty',
        dtime=240,
        synthesis_time=4,
        direction=None,
        f0=None,
    ):
    """creates empty meerkat measurement set

    Example of `direction`: `"13h18m54.86s,-15d36m04.25s"`
    Example of `f0`: `1400`
    
    """
    import simms
    from casatasks import exportuvfits
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u


    if direction is None:
        ra = np.random.rand()*360
        dec = -30 + np.random.rand()*80 - 40
        coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
        direction = coord.to_string('hmsdms',precision=2).replace(' ',',')
    if f0 is None:
        f0 = int(800 + 700*np.random.rand())

    simms_call = f"""simms -dir J2000,{direction} -os -T meerkat -dt {dtime} -st {synthesis_time} -nc 10 -f0 {f0}MHz -df 10MHz -pl XX YY -n {msname}.ms"""
    print(simms_call)
    os.system(simms_call)


    # Export UV data into a fits file
    exportuvfits(f"{msname}.ms", fitsfile=f"{msname}_uv.fits")

    # Load fits file
    uvfits = fits.open(f"{msname}_uv.fits")

    uu_data = uvfits[0].data['UU']
    vv_data = uvfits[0].data['VV']

    # Normalize UV data to [-pi, pi]
    uu_data = (uu_data - uu_data.min()) / (uu_data.max() - uu_data.min())
    uu_data = uu_data * 2 * np.pi - np.pi

    vv_data = (vv_data - vv_data.min()) / (vv_data.max() - vv_data.min())
    vv_data = vv_data * 2 * np.pi - np.pi

    # Save dict
    save_dict = {
        'uu' : uu_data,
        'vv' : vv_data,
        'simms_call' : simms_call,
    }
    np.save(f"{msname}_uv_only.npy", save_dict, allow_pickle=True)


    # Plot UV coverage
    plt.figure(figsize=(10,8), dpi=300)
    s = (np.arange(len(uu_data)) + 1) / (len(uu_data))
    plt.scatter(uu_data, vv_data, s=s, alpha=0.75)
    plt.xlabel(r"u", fontsize=16)
    plt.ylabel(r"v", fontsize=16)
    plt.savefig(f"{msname}_uv_coverage.jpg")
    plt.close()


