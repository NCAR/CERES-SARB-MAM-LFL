import os
import sys
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pprint import pprint

np.set_printoptions(threshold=np.inf)
xr.set_options(display_max_rows=128)


def refine_lut_points(x_coarse, n):

    x_fine = np.zeros(n * (len(x_coarse) - 1) + 1)

    for i in range(len(x_coarse) - 1):
        x_del = (x_coarse[i+1] - x_coarse[i]) / n
        for j in range(n):
            x_fine[i * n + j] = x_coarse[i] + j * x_del

    x_fine[-1] = x_coarse[-1]

    return x_fine


if __name__ == '__main__':

    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str,
        default=sys.stdout,
        help='log file (default stdout)')
    parser.add_argument('--debug', action='store_true',
        help='set logging level to debug')
    parser.add_argument('--datadir', type=str,
        default=os.path.join(os.getenv('HOME'), 'Data'),
        help='top-level data directory (default $HOME/Data)')
    parser.add_argument('--aerosol', type=str,
        default=os.path.join('aerosol.yaml'),
        help='yaml aerosol file')
    parser.add_argument('--scheme', type=str, default='MAM4')
    parser.add_argument('--mode', type=str, default='a1')
    parser.add_argument('--band', type=str, default='sw1')
    parser.add_argument('--refine', type=int, default=4)
    args = parser.parse_args()

    """
    Setup logging
    """
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(stream=args.logfile, level=logging_level)

    with open(args.aerosol, 'r') as f:
        aerosol_config = yaml.safe_load(f)

    logging.info('%s:mode:%s' % (args.scheme, args.mode))
    mode_info = aerosol_config[args.scheme][args.mode]
    mode_info['label'] = args.mode
    mode_info['band'] = args.band
    pprint(mode_info)
    filename_sarb = os.path.expandvars(mode_info['filename_sarb'])
    ds_table = xr.open_dataset(filename_sarb)
    logging.info(ds_table)
    band_idx = int(mode_info['band'][2:]) - 1
    # mode_idx = int(mode_info['label'][1]) - 1
    mode_idx = 0
    if 'sw' in mode_info['band']:
        n_real = ds_table['refindex_real_sw'].values[band_idx, :]
        n_imag = ds_table['refindex_im_sw'].values[band_idx, :]
        extp = ds_table['extpsw_mie'].values[band_idx, mode_idx, :, :, :]
        absp = ds_table['abspsw_mie'].values[band_idx, mode_idx, :, :, :]
        asmp = ds_table['asmpsw'].values[band_idx, mode_idx, :, :, :]
    if 'lw' in mode_info['band']:
        n_real = ds_table['refindex_real_lw'].values[band_idx, :]
        n_imag = ds_table['refindex_im_lw'].values[band_idx, :]
        extp = ds_table['extplw_mie'].values[band_idx, mode_idx, :, :, :]
        absp = ds_table['absplw_mie'].values[band_idx, mode_idx, :, :, :]
        asmp = ds_table['asmplw'].values[band_idx, mode_idx, :, :, :]
    radius = ds_table['particle_radius'].values

    n_real_fine = np.linspace(1.331, 1.999, 100)
    print(n_real_fine[0], n_real_fine[-1])
    n_imag_fine = np.linspace(0.0, 0.79, 100)
    print(n_imag_fine[0], n_imag_fine[-1])

    """
    n_real_fine = refine_lut_points(n_real, args.refine)
    n_imag_fine = refine_lut_points(n_imag, args.refine)

    print('n_real')
    for n in n_real_fine:
        if n in n_real:
            print('%4.2f %4.2f' % (n, n))
        else:
            print('     %4.2f' % n)
    print('\nn_imag')
    for n in n_imag_fine:
        if n in n_imag:
            print('%7.5f %7.5f' % (n, n))
        else:
            print('        %7.5f' % n)
    """

    n_real_imag_list = []
    for n_i in n_imag_fine:
        for n_r in n_real_fine:
            n_real_imag_list.append([n_i, n_r])

    idx_list = []
    for i in range(len(n_imag_fine)):
        for r in range(len(n_real_fine)):
            idx_list.append([i, r])

    ext_fine = np.zeros((len(n_real_fine), len(n_imag_fine), len(radius)), dtype=np.float32)
    abs_fine = np.zeros((len(n_real_fine), len(n_imag_fine), len(radius)), dtype=np.float32)
    asm_fine = np.zeros((len(n_real_fine), len(n_imag_fine), len(radius)), dtype=np.float32)

    for i in range(len(radius)):
        logging.info('interpolating:%.2f um', radius[i])
        ext_interpolator = RegularGridInterpolator((n_imag, n_real), extp[:,:,i])
        abs_interpolator = RegularGridInterpolator((n_imag, n_real), absp[:,:,i])
        asm_interpolator = RegularGridInterpolator((n_imag, n_real), asmp[:,:,i])
        ext_interp = ext_interpolator(n_real_imag_list)
        abs_interp = abs_interpolator(n_real_imag_list)
        asm_interp = asm_interpolator(n_real_imag_list)
        for j in range(len(ext_interp)):
            ext_fine[idx_list[j][1], idx_list[j][0], i] = ext_interp[j]
            abs_fine[idx_list[j][1], idx_list[j][0], i] = abs_interp[j]
            asm_fine[idx_list[j][1], idx_list[j][0], i] = asm_interp[j]

    da_n_real = xr.DataArray(n_real_fine, dims=['n_real'])
    da_n_imag = xr.DataArray(n_imag_fine, dims=['n_imag'])
    da_radius = xr.DataArray(radius, dims=['radius'])
    da_radius.attrs['units'] = 'micrometer'

    da_ext_fine = xr.DataArray(ext_fine, dims=['n_real', 'n_imag', 'radius'])
    # da_ext_fine.attrs['units'] = 'meter^2 kilogram^-1'
    da_ext_fine.attrs['units'] = 'micrometer^2'

    da_abs_fine = xr.DataArray(abs_fine, dims=['n_real', 'n_imag', 'radius'])
    # da_abs_fine.attrs['units'] = 'meter^2 kilogram^-1'
    da_abs_fine.attrs['units'] = 'micrometer^2'

    da_asm_fine = xr.DataArray(asm_fine, dims=['n_real', 'n_imag', 'radius'])

    ds_fine = xr.Dataset({'ext': da_ext_fine, 'abs': da_abs_fine, 'asm': da_asm_fine},
        coords={'n_real': da_n_real, 'n_imag': da_n_imag, 'radius': da_radius})

    # carry the mode geometry into the fine table so the production lookup can
    # assert config sigma_g == LUT sigma_g (guards against number/optics desync)
    for scalar in ('sigmag', 'dgnum', 'dgnumlo', 'dgnumhi'):
        if scalar in ds_table:
            ds_fine[scalar] = ((), float(ds_table[scalar]))
    ds_fine.attrs.update(ds_table.attrs)

    filename_fine = filename_sarb.replace('larc', mode_info['band'] + '_larc')
    logging.info(filename_fine)
    ds_fine.to_netcdf(filename_fine)

