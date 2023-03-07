import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

def get3Ddata(file):
    hc = pd.read_csv(file)
    hc = hc.groupby(['dev', 'alg','n','bs','q','lr']).agg([np.mean, np.std]).reset_index();
    hc['n-exp'] = np.log2(hc['n'])
    hc['nb'] = np.log2(hc['n'] / hc['bs'])
    hc['lr-ratio'] = np.round(np.log2(hc['lr'] / hc['n']))
    hc['mean_ns/q'] = hc['ns/q','mean']
    return hc

def heat_map(x, y, plane, df, vmax=100):
    col = ""
    for c in ['n-exp', 'nb', 'lr-ratio']:
        if c not in {x, y}:
            col = c
    df_nb = df[df[col] == plane]

    ax_labels = {'n-exp': "Array size (2^x)",
                 'nb' : "Number of blocks (2^x)",
                 'lr-ratio' : "Segment size fraction (2^x)"}
    ax_ticks = {'n-exp' : sorted(df_nb['n-exp'].unique()),
                'nb' : sorted(df_nb['nb'].unique()),
                'lr-ratio' : sorted(df_nb['lr-ratio'].unique())}

    pl = df_nb.pivot(values = 'mean_ns/q', index = y, columns = x)

    fig = plt.figure()
    fig.set_figwidth(11)
    fig.set_figheight(7)

    minval = df_nb['mean_ns/q'].min()
    maxval = df_nb['mean_ns/q'].max()
    print(f"{minval=}   {maxval=}")
    #plt.pcolor(ax_ticks[x], ax_ticks[y], pl, norm=matplotlib.colors.LogNorm(vmin=.5, vmax=vmax))
    plt.pcolor(ax_ticks[x], ax_ticks[y], pl, norm=matplotlib.colors.LogNorm(vmin=minval, vmax=maxval))
    plt.colorbar()
    plt.xlabel(ax_labels[x])
    plt.ylabel(ax_labels[y])
    # plt.yticks([i for i in range(5,26,2)])
    plt.title(f"RTX 3090 Ti, LR-frac vs #Blocks, n=2^{plane}")
    # plt.savefig(f"../plots/heat_map_3090Ti.pdf", dpi=300, facecolor="#ffffff", bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Run with arguments <csv_path> <n-exp>")
        exit()
    data_path = sys.argv[1]
    nexp = int(sys.argv[2])
    print(f"ARGS:\n\t{data_path=}\n\t{nexp=}")
    df_3d = get3Ddata(data_path)
    nmin = int(df_3d['n-exp'].min())
    nmax = int(df_3d['n-exp'].max())
    #print(f"{nmin=}  {nmax=}")
    if nexp > nmax or nexp < nmin:
        print(f"Error:\n\t'n-exp' must be between the file's range: {nmin}..{nmax}\n")
        exit()

    print("n:", sorted(df_3d['n-exp'].unique()))
    print("nb:", sorted(df_3d['nb'].unique()))
    print("lr:", sorted(df_3d['lr-ratio'].unique()))

    heat_map('nb', 'lr-ratio', nexp, df_3d, vmax=30)
