from python_src.first_passage import MoranFPT, Spatial, SpatInv
from manuscript_figs import fitness_spatial, spatial_vs_moran, fitness_spatial
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.lines import Line2D
import scipy as sp

plt.style.use('python_src/custom.mplstyle')


def moranME_FP(times, filename, model):
    s = [1.0, 1.01, 1.1]
    N = 100

    marker = 'X'
    cmap = matplotlib.cm.get_cmap('plasma')
    colors = [cmap(nbr) for nbr in np.linspace(0.0, 1.0, num=len(s))]
    linestyle = [':', '-', '--']

    # colour legend
    col_label = s
    col_lines = [Line2D([0], [0], color=col) for col in colors]
    cl = plt.legend(col_lines, col_label, loc='lower right')

    # Fig1A : P_absorbing
    fig1, ax1 = plt.subplots(figsize=(3.4, 2.5))
    ax1.set(title=r"",
            xlabel=r"Fractional abundance, $f$",
            ylabel=r"Probability fixation, $P_N(f)$")

    legend = ['moran (ME)', 'moran (FP)']
    custom_lines = [
        Line2D([0], [0], markerfacecolor='dimgray', linestyle='None',
               marker=marker, color='k', linewidth=1),
        Line2D([0], [0], color='dimgray', linestyle=linestyle[0])
        ]

    # Fig1B : MFPT function of x
    fig2, ax2 = plt.subplots(figsize=(3.4, 2.5))
    ax2.set_xlim([0.0, 1.0])
    ax2.set(title=r"",
            xlabel=r"Fractional abundance, $f$",
            ylabel=r"MFPT, $\tau(f)$")

    # Fig1C : MFPT as function of N
    fig3, ax3 = plt.subplots(figsize=(3.4, 2.5))
    ax3.set(title=r"",
            xlabel=r"Total population size, $N$",
            ylabel=r"MFPT, $\tau(f_{max})$")

    moran = []
    for i, fit in enumerate(s):
        # define models
        moran.append(model(fit, 1.0, N, times))

        # plots
        x = []
        ME_mfpt_moran = []
        ME_prob_moran = []

        # Fig 1 A
        for j in np.linspace(1, N - 1, 9):
            x.append(j / N)
            j = int(j)

            # ME approach
            ME_prob, ME_mfpt = moran[i].probability_mfpt(j)
            ME_prob_moran.append(ME_prob[1])
            ME_mfpt_moran.append(np.dot(ME_prob, ME_mfpt))

        X, FP_prob_moran = moran[i].FP_prob()
        ax1.plot(X, FP_prob_moran, linestyle=linestyle[0], color=colors[i],
                 zorder=i)
        ax1.scatter(x, ME_prob_moran, color=colors[i], marker=marker,
                    s=12, edgecolors='k', linewidth=1, zorder=i+.5)

        # Fig 1 B
        X, FP_mfpt_moran = moran[i].FP_mfpt()
        ax2.plot(X, FP_mfpt_moran, linestyle=linestyle[0], color=colors[i],
                 zorder=i)
        ax2.scatter(x, ME_mfpt_moran, color=colors[i], marker=marker,
                    s=12, edgecolors='k', linewidth=1, zorder=i+.5)

        # Fig 1 C
        # prob, mfpt = space[i].probability_mfpt(space[i].nmax)
        # ME_mfpt_space_N.append(np.dot(prob, mfpt))

    # Fig 1C: MFPT as a function of N at x_max
    """
    N_func = np.linspace(10, N[-1], 100)
    _, FP_mfpt_moran_N = moran[0].FP_mfpt(x=moran[0].xmax, N=N_func)
    _, FP_mfpt_space_N = space[0].FP_mfpt_x(x_find=space[0].xmax, N=N_func)

    ax3.plot(N_func, FP_mfpt_moran_N, color='k', linestyle=linestyle[2])
    ax3.plot(N_func, FP_mfpt_space_N, color='dimgray')

    ax3.scatter(N, ME_mfpt_space_N, label=r'ME Homog.', marker='o',
                color=colors[0: len(N)], s=12, edgecolors='k', linewidth=1,
                zorder=i+.5)
    """

    # figure limits and legends
    ax1.set_xlim([0.00, 1.0])
    ax1.set_ylim([0.00, 1.0])
    ax3.set_yscale('log')
    ax2.set_ylim(0.1, 100)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_ylim([1.0, 1000])
    cl = ax1.legend(col_lines, col_label, loc='lower right', title=r'$s$')
    ax1.legend(custom_lines, legend, title=r'Model', loc='upper right',
               framealpha=0.9)
    ax3.legend(custom_lines, legend, title=r'Model')
    ax1.add_artist(cl)
    # ax2.legend(det, [r'$\tau_{det}$'])
    # ax3.legend(custom_lines, legend)

    # save figures
    fig1.savefig(filename + '_prob.pdf')
    fig1.savefig(filename + '_prob.png')
    fig2.savefig(filename + '_mfpt.pdf')
    fig2.savefig(filename + '_mfpt.png')
    # fig3.savefig(filename + '_N.pdf')
    # fig3.savefig(filename + '_N.png')
    return s


def invasion_check(s, N, cmap_name, filename, model):
    marker = 'o'
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = [cmap(nbr) for nbr in np.linspace(0.0, 1.0, num=len(s))]
    linestyle = [':', '-', '--']

    # colour legend
    col_label = s
    col_lines = [Line2D([0], [0], color=col) for col in colors]

    # Fig2A : P_absorbing
    fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(3.4, 5.0))
    plt.subplots_adjust(wspace=0, hspace=0.0)
    # fig1, ax1 = plt.subplots(figsize=(3.4, 2.5))
    ax1.set(title=r"",
            # xlabel=r"Initial species 1 fraction, $f$",
            ylabel=r"Probability fixation, $P_N(f)$")

    # Fig2B : MFPT function of x
    # fig2, ax2 = plt.subplots(figsize=(3.4, 2.5))
    ax2.set_xlim([0.0, 1.0])
    ax2.set(title=r"",
            xlabel=r"Initial species 1 fraction, $f$",
            ylabel=r"MFPT, $\tau(f)$")

    space = []
    # ME_mfpt_space_N = []
    for i, fit in enumerate(s):
        # define models
        space.append(model(fit, 1.0, N, times))

        # plots
        x = []
        ME_mfpt_space = []
        ME_prob_space = []

        # Fig 1 A
        # print(np.linspace(1, N - 1, 9))
        for j in np.linspace(1, N - 1, 9):
            x.append(j / N)
            j = int(j)

            # ME approach
            ME_prob, ME_mfpt = space[i].probability_mfpt(0, space[i].K - j)
            ME_prob_space.append(ME_prob[0])
            ME_mfpt_space.append(np.dot(ME_prob, ME_mfpt))

        ax1.scatter(x, ME_prob_space, color=colors[i], marker=marker,
                    s=12, edgecolors='k', linewidth=1, zorder=i+.5)
        ax2.scatter(x, ME_mfpt_space, color=colors[i], marker=marker,
                    s=12, edgecolors='k', linewidth=1, zorder=i + .5)

    # deterministic approximation times
    x_det = np.linspace(0.2, 1.0, 100)
    S = 100
    t_det = - (np.log(np.abs((S - 1) * x_det**2 + 2 * x_det - 1)) - np.log(S))
    det = ax2.plot(x_det, t_det, linestyle=linestyle[2], color='r', zorder=i+1)
    x_det = np.linspace(0.0, 0.18, 100)
    t_det2 = - (sp.log((S - 1) * x_det**2 + 2 * x_det - 1) - sp.log(-1))
    ax2.plot(x_det, t_det2, linestyle=linestyle[2], color='r', zorder=i+1)

    # figure limits and legends
    # legend
    legend = ['spatial (ME)', 'spatial (FP)', 'moran (FP)']
    custom_lines = [
        Line2D([0], [0], markerfacecolor='dimgray', linestyle='None',
               marker=marker, color='k', linewidth=1),
        Line2D([0], [0], color='dimgray', linestyle=linestyle[1]),
        Line2D([0], [0], color='k', linestyle=linestyle[0])
        ]

    # limits
    ax1.set_xlim([0.00, 1.0])
    ax1.set_ylim([0.00, 1.0])
    cl = ax1.legend(col_lines, col_label, loc='lower right', title=r'$s$')
    ax1.legend(custom_lines, legend, title=r'Model', loc='upper right',
               framealpha=0.8)
    ax1.add_artist(cl)

    ax2.set_yscale('log')
    ax2.set_ylim(0.1, 100)
    ax2.legend(det, [r'$\tau_{\mathrm{det}}$'])

    # save figures
    fig1.savefig(filename + '_prob.pdf')
    fig1.savefig(filename + '_prob.png')
    return 0


def side_invasion(num, cmap_name_s, cmap_name_N, filename, model):
    s = np.logspace(0, 2, num * 2 + 1, dtype=int)
    N = np.logspace(1, 3, num * 2 + 1, dtype=int)
    marker = ['+', 'x']
    markeredge = None
    ms = 5
    cmap1 = matplotlib.cm.get_cmap(cmap_name_s)
    cmap2 = matplotlib.cm.get_cmap(cmap_name_N)
    color_s = [cmap1(nbr) for nbr in np.linspace(0.0, 1.0, num=num + 1)]
    color_N = [cmap2(nbr) for nbr in np.linspace(0.0, 1.0, num=num + 1)]
    linestyle = [':', '-', '--']

    # colour legend
    col_label_s = np.logspace(0, 2, num + 1, dtype=int)
    col_label_N = np.logspace(1, 3, num + 1, dtype=int)
    col_lines_s = [Line2D([0], [0], color=col) for col in color_s]
    col_lines_N = [Line2D([0], [0], color=col) for col in color_N]

    # Fig2A : P_absorbing
    # fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(3.4, 5.0))
    # plt.subplots_adjust(wspace=0, hspace=0.0)
    fig1, ax1 = plt.subplots(figsize=(3.4, 2.5))
    ax1.set(title=r"",
            xlabel=r"Fitness, s",
            ylabel=r"Invasion probability, $P_1(\frac{2}{N})$")

    # Fig2B : MFPT function of x
    fig2, ax2 = plt.subplots(figsize=(3.4, 2.5))
    ax2.set(title=r"",
            xlabel=r"Fitness, s",
            ylabel=r"Invasion MFPT, $\tau(\frac{2}{N})$")

    # Fig2B : MFPT function of x
    fig3, ax3 = plt.subplots(figsize=(3.4, 2.5))
    ax3.set(title=r"",
            xlabel=r"Fitness, s",
            ylabel=r"Invasion succeeds MFPT""\n"r"$\tau_1(\frac{2}{N})$")

    fig4, ax4 = plt.subplots(figsize=(3.4, 2.5))
    ax4.set(title=r"",
            xlabel=r"Population size, N",
            ylabel=r"Invasion probability, $P_1(\frac{2}{N})$")

    # Fig2B : MFPT function of x
    fig5, ax5 = plt.subplots(figsize=(3.4, 2.5))
    ax5.set(title=r"",
            xlabel=r"Population size, N",
            ylabel=r"Invasion MFPT, $\tau(\frac{2}{N})$")

    # Fig2B : MFPT function of x
    fig6, ax6 = plt.subplots(figsize=(3.4, 2.5))
    ax6.set(title=r"",
            xlabel=r"Population size, N",
            ylabel=r"Invasion succeeds MFPT""\n"r"$\tau_1(\frac{2}{N})$")

    # fitness
    for i, K in enumerate(np.logspace(1, 3, num + 1, dtype=int)):
        ME_mfpt_moran = []
        ME_prob_moran = []
        ME_mfpD_moran = []

        ME_mfpt_space = []
        ME_prob_space = []
        ME_mfpD_space = []
        for j, fit in enumerate(s):
            # define models
            moran = MoranFPT(fit, 1.0, K, times)
            space = model(fit, 1.0, K, times)

            # ME approach
            ME_prob, ME_mfpt = moran.probability_mfpt(2)
            ME_prob_moran.append(ME_prob[1])
            ME_mfpD_moran.append(ME_mfpt[1])
            ME_mfpt_moran.append(np.dot(ME_prob, ME_mfpt))
            ME_prob, ME_mfpt = space.probability_mfpt(2)
            ME_prob_space.append(ME_prob[1])
            ME_mfpD_space.append(ME_mfpt[1])
            ME_mfpt_space.append(np.dot(ME_prob, ME_mfpt))

        ax1.plot(s, ME_prob_moran, color=color_N[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax1.plot(s, ME_prob_space, color=color_N[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

        ax2.plot(s, ME_mfpt_moran, color=color_N[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax2.plot(s, ME_mfpt_space, color=color_N[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

        ax3.plot(s, ME_mfpD_moran, color=color_N[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax3.plot(s, ME_mfpD_space, color=color_N[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

    # population size
    for i, fit in enumerate(np.logspace(0, 2, num + 1)):
        ME_mfpt_moran = []
        ME_prob_moran = []
        ME_mfpD_moran = []

        ME_mfpt_space = []
        ME_prob_space = []
        ME_mfpD_space = []
        for j, K in enumerate(N):
            # define models
            moran = MoranFPT(fit, 1.0, K, times)
            space = model(fit, 1.0, K, times)

            # ME approach
            ME_prob, ME_mfpt = moran.probability_mfpt(2)
            ME_prob_moran.append(ME_prob[1])
            ME_mfpD_moran.append(ME_mfpt[1])
            ME_mfpt_moran.append(np.dot(ME_prob, ME_mfpt))
            ME_prob, ME_mfpt = space.probability_mfpt(2)
            ME_prob_space.append(ME_prob[1])
            ME_mfpD_space.append(ME_mfpt[1])
            ME_mfpt_space.append(np.dot(ME_prob, ME_mfpt))

        ax4.plot(N, ME_prob_moran, color=color_s[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax4.plot(N, ME_prob_space, color=color_s[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

        ax5.plot(N, ME_mfpt_moran, color=color_s[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax5.plot(N, ME_mfpt_space, color=color_s[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

        ax6.plot(N, ME_mfpD_moran, color=color_s[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax6.plot(N, ME_mfpD_space, color=color_s[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

    # figure limits and legends
    cl1 = ax1.legend(col_lines_N, col_label_N, loc='lower right', title=r'$N$')
    # ax1.legend(custom_lines, legend, title=r'Model', loc='upper right',
    #            framealpha=0.8)
    ax1.add_artist(cl1)
    ax1.set_xscale('log')
    ax1.set_ylim([0.00, 1.01])
    # ax1.set_yscale('log'); ax1.set_ylim([0.001, 1.01])

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    # ax2.set_ylim(0.1, 100)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    # ax3.set_ylim(0.1, 100)

    ax4.set_ylim([0.001, 1.01])
    cl2 = ax4.legend(col_lines_s, col_label_s, loc='lower right', title=r'$s$')
    # ax1.legend(custom_lines, legend, title=r'Model', loc='upper right',
    #            framealpha=0.8)
    ax4.add_artist(cl2)
    ax4.set_xscale('log')
    ax4.set_ylim([0.00, 1.01])
    # ax4.set_yscale('log'); ax4.set_ylim([0.001, 1.01])

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    # ax2.set_ylim(0.1, 100)
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    # ax3.set_ylim(0.1, 100)

    # save figures
    fig1.savefig(filename + '_prob_s.pdf')
    fig1.savefig(filename + '_prob_s.png')
    fig2.savefig(filename + '_mfpt_s.pdf')
    fig2.savefig(filename + '_mfpt_s.png')
    fig3.savefig(filename + '_mfpD_s.pdf')
    fig3.savefig(filename + '_mfpD_s.png')
    fig4.savefig(filename + '_prob_N.pdf')
    fig4.savefig(filename + '_prob_N.png')
    fig5.savefig(filename + '_mfpt_N.pdf')
    fig5.savefig(filename + '_mfpt_N.png')
    fig6.savefig(filename + '_mfpD_N.pdf')
    fig6.savefig(filename + '_mfpD_N.png')

    return 0


def average(num, cmap_name_s, cmap_name_N, filename, model):
    # Unclear what difference is between average and averages?
    s = np.logspace(0, 2, num * 2 + 1)
    N = np.logspace(1, 3, num * 2 + 1, dtype=int)
    marker = ['+', 'x']
    markeredge = None
    ms = 5
    cmap1 = matplotlib.cm.get_cmap(cmap_name_s)
    cmap2 = matplotlib.cm.get_cmap(cmap_name_N)
    color_s = [cmap1(nbr) for nbr in np.linspace(0.0, 1.0, num=num + 1)]
    color_N = [cmap2(nbr) for nbr in np.linspace(0.0, 1.0, num=num + 1)]
    linestyle = [':', '-', '--']

    # colour legend
    col_label_s = np.logspace(0, 2, num + 1, dtype=int)
    col_label_N = np.logspace(1, 3, num + 1, dtype=int)
    col_lines_s = [Line2D([0], [0], color=col) for col in color_s]
    col_lines_N = [Line2D([0], [0], color=col) for col in color_N]

    # Fig2A : P_absorbing
    # fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(3.4, 5.0))
    # plt.subplots_adjust(wspace=0, hspace=0.0)
    fig1, ax1 = plt.subplots(figsize=(3.4, 2.5))
    ax1.set(title=r"",
            xlabel=r"Fitness, s",
            ylabel=r"Average probability, $\langle P_1 \rangle$")

    # Fig2B : MFPT function of x
    fig2, ax2 = plt.subplots(figsize=(3.4, 2.5))
    ax2.set(title=r"",
            xlabel=r"Fitness, s",
            ylabel=r"Average MFPT, $\langle \tau \rangle$")

    # Fig2B : MFPT function of x
    fig3, ax3 = plt.subplots(figsize=(3.4, 2.5))
    ax3.set(title=r"",
            xlabel=r"Fitness, s",
            ylabel=r"Average MFPT at $f_F=1$""\n"r" $\langle \tau_1 \rangle$")

    fig4, ax4 = plt.subplots(figsize=(3.4, 2.5))
    ax4.set(title=r"",
            xlabel=r"Population size, N",
            ylabel=r"Average probability, $\langle P_1 \rangle$")

    # Fig2B : MFPT function of x
    fig5, ax5 = plt.subplots(figsize=(3.4, 2.5))
    ax5.set(title=r"",
            xlabel=r"Population size, N",
            ylabel=r"Average MFPT, $\langle \tau \rangle$")

    # Fig2B : MFPT function of x
    fig6, ax6 = plt.subplots(figsize=(3.4, 2.5))
    ax6.set(title=r"",
            xlabel=r"Population size, N",
            ylabel=r"Average MFPT at $f_F=1$""\n"r" $\langle \tau_1 \rangle$")

    # fitness
    for i, K in enumerate(np.logspace(1, 3, num + 1, dtype=int)):
        ME_mfpt_moran = []
        ME_prob_moran = []
        ME_mfpD_moran = []

        ME_mfpt_space = []
        ME_prob_space = []
        ME_mfpD_space = []
        for j, fit in enumerate(s):
            # define models
            moran = MoranFPT(fit, 1.0, K, times)
            space = model(fit, 1.0, K, times)

            # ME approach
            probM = 0
            mfptM = 0
            mfpDM = 0
            probS = 0
            mfptS = 0
            mfpDS = 0
            for n in np.arange(K + 1):
                ME_prob, ME_mfpt = moran.probability_mfpt(n)
                probM += ME_prob[1]
                mfpDM += ME_mfpt[1]
                mfptM += np.dot(ME_prob, ME_mfpt)
                ME_prob, ME_mfpt = space.probability_mfpt(n)
                probS += ME_prob[1]
                mfpDS += ME_mfpt[1]
                mfptS += np.dot(ME_prob, ME_mfpt)

            ME_prob_moran.append(probM / K)
            ME_mfpD_moran.append(mfptM / K)
            ME_mfpt_moran.append(mfpDM / K)
            ME_prob_space.append(probS / K)
            ME_mfpD_space.append(mfptS / K)
            ME_mfpt_space.append(mfpDS / K)

        ax1.plot(s, ME_prob_moran, color=color_N[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax1.plot(s, ME_prob_space, color=color_N[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

        ax2.plot(s, ME_mfpt_moran, color=color_N[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax2.plot(s, ME_mfpt_space, color=color_N[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

        ax3.plot(s, ME_mfpD_moran, color=color_N[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax3.plot(s, ME_mfpD_space, color=color_N[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

    # population size
    for i, fit in enumerate(np.logspace(0, 2, num + 1)):
        ME_mfpt_moran = []
        ME_prob_moran = []
        ME_mfpD_moran = []

        ME_mfpt_space = []
        ME_prob_space = []
        ME_mfpD_space = []
        for j, K in enumerate(N):
            # define models
            moran = MoranFPT(fit, 1.0, K, times)
            space = model(fit, 1.0, K, times)

            # ME approach
            probM = 0
            mfptM = 0
            mfpDM = 0
            probS = 0
            mfptS = 0
            mfpDS = 0
            for n in np.arange(K + 1):
                ME_prob, ME_mfpt = moran.probability_mfpt(n)
                probM += ME_prob[1]
                mfpDM += ME_mfpt[1]
                mfptM += np.dot(ME_prob, ME_mfpt)
                ME_prob, ME_mfpt = space.probability_mfpt(n)
                probS += ME_prob[1]
                mfpDS += ME_mfpt[1]
                mfptS += np.dot(ME_prob, ME_mfpt)

            ME_prob_moran.append(probM / K)
            ME_mfpD_moran.append(mfptM / K)
            ME_mfpt_moran.append(mfpDM / K)
            ME_prob_space.append(probS / K)
            ME_mfpD_space.append(mfptS / K)
            ME_mfpt_space.append(mfpDS / K)

        ax4.plot(N, ME_prob_moran, color=color_s[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax4.plot(N, ME_prob_space, color=color_s[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

        ax5.plot(N, ME_mfpt_moran, color=color_s[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax5.plot(N, ME_mfpt_space, color=color_s[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

        ax6.plot(N, ME_mfpD_moran, color=color_s[i], marker=marker[0],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[0], zorder=i + .5)
        ax6.plot(N, ME_mfpD_space, color=color_s[i], marker=marker[1],
                 markersize=ms + 1, markeredgecolor=markeredge, linewidth=1,
                 linestyle=linestyle[1], zorder=i + .5)

    # figure limits and legends
    cl1 = ax1.legend(col_lines_N, col_label_N, loc='best', title=r'$N$')
    # ax1.legend(custom_lines, legend, title=r'Model', loc='upper right',
    #            framealpha=0.8)
    ax1.add_artist(cl1)
    ax1.set_xscale('log')
    ax1.set_ylim([0.00, 1.01])
    # ax1.set_yscale('log'); ax1.set_ylim([0.001, 1.01])

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    # ax2.set_ylim(0.1, 100)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    # ax3.set_ylim(0.1, 100)

    cl2 = ax4.legend(col_lines_s, col_label_s, loc='lower right', title=r'$s$',
                     prop = { "size": 5 })
    # ax1.legend(custom_lines, legend, title=r'Model', loc='upper right',
    #            framealpha=0.8)
    ax4.add_artist(cl2)
    ax4.set_xscale('log')
    ax4.set_ylim([0.00, 1.01])
    # ax4.set_yscale('log'); ax4.set_ylim([0.001, 1.01])

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    # ax2.set_ylim(0.1, 100)
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    # ax3.set_ylim(0.1, 100)

    # save figures
    fig1.savefig(filename + '_prob_s.pdf')
    fig1.savefig(filename + '_prob_s.png')
    fig2.savefig(filename + '_mfpt_s.pdf')
    fig2.savefig(filename + '_mfpt_s.png')
    fig3.savefig(filename + '_mfpD_s.pdf')
    fig3.savefig(filename + '_mfpD_s.png')
    fig4.savefig(filename + '_prob_N.pdf')
    fig4.savefig(filename + '_prob_N.png')
    fig5.savefig(filename + '_mfpt_N.pdf')
    fig5.savefig(filename + '_mfpt_N.png')
    fig6.savefig(filename + '_mfpD_N.pdf')
    fig6.savefig(filename + '_mfpD_N.png')

    return 0


def moran_selection(filename, bot, mid, top, cmap_name):
    # compares discrete and fp solution of Moran model
    def discrete(x, s, N):
        return (1 - s ** (- x * N)) / (1 - s ** (- N))

    def continuo(x, s, N):
        a = 2 * (s - 1) / (s + 1)
        return (1 - np.exp(- a * x * N)) / (1 - np.exp(- a * N))

    fig1, ax1 = plt.subplots(figsize=(3.4, 2.5))
    ax1.set(title=r"",
            xlabel=r"Fitness, s",
            ylabel=r"y(s)")

    fig2, ax2 = plt.subplots(figsize=(3.4, 2.5))
    ax2.set(title=r"$s={}$".format(bot),
            xlabel=r"Fractional abundance, $f$",
            ylabel=r"Probability $P_1(f)$")

    fig3, ax3 = plt.subplots(figsize=(3.4, 2.5))
    ax3.set(title=r"$s={}$".format(mid),
            xlabel=r"Fractional abundance, $f$",
            ylabel=r"Probability $P_1(f)$")

    fig4, ax4 = plt.subplots(figsize=(3.4, 2.5))
    ax4.set(title=r"$s={}$".format(top),
            xlabel=r"Fractional abundance, $f$",
            ylabel=r"Probability $P_1(f)$")

    axs = [ax2, ax3, ax4]
    fit = [bot, mid, top]

    s = np.linspace(0.0, 10, 100)
    ax1.plot(s, np.exp(2 * (s - 1) / (s + 1)), color='k', linewidth=2,
             linestyle='-', label=r'$y = e^{2 (s - 1) / (s + 1)}$')
    ax1.plot(s, s, color='b', linewidth=2, linestyle=':', label=r'$y = s$')
    ax1.legend()

    x = np.linspace(0.0, 1.0, 100)
    n = np.linspace(0.0, 1.0, 10)
    K = [10, 100, 1000]
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = [cmap(nbr) for nbr in np.linspace(0.0, 0.8, num=len(K))]

    for i, ax in enumerate(axs):
        for j, k in enumerate(K):
            ax.scatter(n, discrete(n, fit[i], k), color=colors[j], linewidth=2,
                       marker='X', s=6, edgecolor=None)
            ax.plot(x, continuo(x, fit[i], k), color=colors[j], linewidth=2,
                    linestyle=':', label=r'$N={}$'.format(k))
        ax.legend()

    fig1.savefig(filename + '_comp.pdf')
    fig1.savefig(filename + '_comp.png')
    fig2.savefig(filename + '_bot_s.pdf')
    fig2.savefig(filename + '_bot_s.png')
    fig3.savefig(filename + '_mid_s.pdf')
    fig3.savefig(filename + '_mid_s.png')
    fig4.savefig(filename + '_top_s.pdf')
    fig4.savefig(filename + '_top_s.png')

    return 0


def det_time(s, cmap_name, filename, model):

    N = np.linspace(10, 1000, 100)
    K = 100  # a placeholder

    # colour legend
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = [cmap(nbr) for nbr in np.linspace(0.0, 0.8, num=len(s))]
    linestyle = [':', '-', '--']

    def x_det_o(s, N):
        a = np.sqrt(np.pi * N * np.sqrt(s)) / ((1 + np.sqrt(s)))
        squ = a * (np.sqrt(s) - 1) + 1 / (2 * np.sqrt(s) * a)
        divP = (s + 1.) + a * (s - 1.)
        divM = (s + 1.) - a * (s - 1.)
        return (1 + squ) / divP, (1 - squ) / divM

    def x_det(s, N):  # from Wolfram
        a = np.sqrt(np.pi * N * np.sqrt(s)) / ((1 + np.sqrt(s)))
        squ = np.sqrt(s * (a ** 2 - 1))
        divP = a * (s - 1.) + (s + 1)
        divM = a * (s - 1.) - (s + 1)
        return (1 + squ - a) / divP, (-1 + squ - a) / divM

    def time_det(s, N):
        x1, x2 = x_det(s, N)
        t1 = -(sp.log((s - 1) * x1 ** 2 + 2 * x1 - 1) - sp.log(s))
        t2 = -(sp.log((s - 1) * x2 ** 2 + 2 * x2 - 1) - sp.log(-1))
        return (t1 + t2) / 2

    # Fig3C : MFPT asymptotic function of N
    fig3, ax3 = plt.subplots(figsize=(3.4, 2.5))
    ax3.set(title=r"",
            xlabel=r"Total population size, $N$",
            ylabel=r"MFPT")

    # Fig3C : MFPT asymptotic function of N
    fig4, ax4 = plt.subplots(figsize=(3.4, 2.5))
    ax4.set(title=r"",
            xlabel=r"Total population size, $N$",
            ylabel=r"$\tau(f_{\mathrm{max}})-\tau_{\mathrm{det}}(x_t^{\pm})$")

    space = []

    # Fig 3C
    for i, fit in enumerate(s):
        space = model(fit, 1, K, 0)
        _, FP_mfpt_space_N = space.FP_mfpt_x(x_find=space.xmax, N=N)

        ax3.plot(N, FP_mfpt_space_N, color=colors[i], linestyle=linestyle[1],
                 zorder=i)

        tdet = time_det(fit, N)
        # tdif = ((-np.log(0.5) + np.log(0.5 + x1))/4 + x1 * np.arctan(x1))
        ax3.plot(N, tdet, color=colors[i], linestyle=linestyle[2],
                 zorder=i+0.5)
        ax4.plot(N, FP_mfpt_space_N - tdet, colors[i], color=colors[i])

    plt.subplots_adjust(wspace=0, hspace=0.0)

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    custom_label2 = [r'$\tau(f_{\mathrm{max}})$',
                     r'$\tau_{\mathrm{det}}(f_{t}^{\pm})$']
    custom_lines2 = [
        Line2D([0], [0], color='dimgray', linestyle=linestyle[1]),
        Line2D([0], [0], color='dimgray', linestyle=linestyle[2])
        ]
    ax3.legend(custom_lines2, custom_label2)
    ax3.set_xlim([np.min(N), np.max(N)])
    ax3.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())

    ax4.set_xscale('log')
    # ax4.set_yscale('log')
    custom_label2 = [r'$\tau(f_{\mathrm{max}})$',
                     r'$\tau_{\mathrm{det}}(f_{t}^{\pm})$']
    custom_lines2 = [
        Line2D([0], [0], color='dimgray', linestyle=linestyle[1]),
        Line2D([0], [0], color='dimgray', linestyle=linestyle[2])
        ]
    # ax3.legend(custom_lines2, custom_label2)
    ax4.set_xlim([np.min(N), np.max(N)])
    tdif = 0.4
    ax4.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax4.axhline(y=tdif, linewidth=1.2, color='black', linestyle=":")
    ax4.text(0.04, tdif-0.085, r"${}$".format(tdif),
             transform=ax4.get_yaxis_transform())

    fig3.savefig(filename + '_tdet.pdf')
    fig3.savefig(filename + '_tdet.png')
    fig4.savefig(filename + '_tdif.pdf')
    fig4.savefig(filename + '_tdif.png')

    return 0


def conditional_spatial_vs_moran(N, times, cmap_name, filename, model):
    r1 = 1.0
    r2 = 1.0
    # marker = 'o'
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = [cmap(nbr) for nbr in np.linspace(0.0, 0.8, num=len(N))]
    linestyle = [':', '-', '--']
    marker = {'right' : 'o', 'left' : '^'}

    # colour legend
    col_label = N
    col_lines = [Line2D([0], [0], color=col) for col in colors]
    
    marker_lines = [Line2D([0], [0], marker=marker['right'], color='k',
                        markerfacecolor='w', label="right species",
                        linestyle=''),
                Line2D([0], [0], marker=marker['left'], color='k',
                        markerfacecolor='w', label="left species",
                        linestyle='')
                ]

    # Fig1A : P_absorbing
    fig4, ax4 = plt.subplots(figsize=(3.4, 2.5))
    ax4.set(title=r"",
            xlabel=r"Fractional abundance, $f$",
            ylabel=r"Conditional fixation time")

    moran = []
    space = []
    for i, K in enumerate(N):
         # define models
        moran.append(MoranFPT(r1, r2, K, times))
        space.append(model(r1, r2, K, times))

        # plots
        x = []
        
        # Fig 1 D
        x = []
        ME_mfpt_left = []
        ME_mfpt_right = []
        ME_mfpt_l = []
        ME_mfpt_r = []
        for j in np.linspace(1, K - 1, 9):
            x.append(j / K)
            j = int(j)

            # ME approach
            ME_prob, ME_mfpt = moran[i].probability_mfpt(j)
            ME_mfpt_l.append(ME_mfpt[1])
            ME_mfpt_r.append(ME_mfpt[0])
            ME_prob, ME_mfpt = space[i].probability_mfpt(j)
            ME_mfpt_left.append(ME_mfpt[1])
            ME_mfpt_right.append(ME_mfpt[0])
        # ax4.scatter(x, ME_mfpt_l, color=colors[i], marker=marker['left'],
        #W            s=12, edgecolors='k', linewidth=0, zorder=i+.5)
        # ax4.scatter(x, ME_mfpt_r, color=colors[i], marker=marker['right'],
        #            s=12, edgecolors='k', linewidth=0, zorder=i+.5)
        ax4.scatter(x, ME_mfpt_left, color=colors[i], marker=marker['left'],
                    s=12, edgecolors='k', linewidth=0.5, zorder=i+.5,
                    label="left species")
        # ax4.scatter(x, ME_mfpt_right, color=colors[i], marker=marker['right'],
        #            s=12, edgecolors='k', linewidth=0.5, zorder=i+.5,
        #            label="right species")

    # ax4.legend(custom_lines, legend, title=r'Model')
    ax4.set_yscale('log')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.1, 10.])
    # ax4.legend()
    # ax4.set_xlim([np.min(N_func), np.max(N_func)])
    # ax4.legend(handles=marker_lines)
    legend1 = ax4.legend(col_lines, col_label, title=r"$N$", loc='best')
    ax4.add_artist(legend1)
    
        # save figures
    fig4.savefig(filename + '_condfix.pdf')
    fig4.savefig(filename + '_condfix.png')

    return 0


def conditional_fitness_spatial(s, N, times, cmap_name, filename, model):

    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = [cmap(nbr) for nbr in np.linspace(0.0, 0.8, num=len(s))]
    linestyle = [':', '-', '--']
    marker = {'right' : 'o', 'left' : '^'}

    # colour legend
    col_label = s  # ["N={}".format(nbr) for nbr in N]
    col_lines = [Line2D([0], [0], color=col) for col in colors]
    cl = plt.legend(col_lines, col_label, loc='lower right')
    
    marker_lines = [Line2D([0], [0], marker=marker['right'], color='k',
                           markerfacecolor='w', label="lower fitness species",
                           linestyle=''),
                    Line2D([0], [0], marker=marker['left'], color='k',
                           markerfacecolor='w', label="higher fitness species",
                           linestyle='')
                    ]

    # Fig1A : P_absorbing
    fig4, ax4 = plt.subplots(figsize=(3.4, 2.5))
    ax4.set(title=r"",
            xlabel=r"Fractional abundance, $f$",
            ylabel=r"Conditional fixation time")

    moran = []
    space = []
    for i, fit in enumerate(s):
         # define models
        moran.append(MoranFPT(fit, 1.0, N, times))
        space.append(model(fit, 1.0, N, times))

        # plots
        x = []
        
        # Fig 1 D
        x = []
        ME_mfpt_left = []
        ME_mfpt_right = []
        ME_mfpt_l = []
        ME_mfpt_r = []
        for j in np.linspace(1, N - 1, 9):
            x.append(j / N)
            j = int(j)

            # ME approach
            ME_prob, ME_mfpt = moran[i].probability_mfpt(j)
            ME_mfpt_l.append(ME_mfpt[1])
            ME_mfpt_r.append(ME_mfpt[0])
            ME_prob, ME_mfpt = space[i].probability_mfpt(j)
            ME_mfpt_left.append(ME_mfpt[1])
            ME_mfpt_right.append(ME_mfpt[0])
        # ax4.scatter(x, ME_mfpt_l, color=colors[i], marker=marker['left'],
        #W            s=12, edgecolors='k', linewidth=0, zorder=i+.5)
        # ax4.scatter(x, ME_mfpt_r, color=colors[i], marker=marker['right'],
        #            s=12, edgecolors='k', linewidth=0, zorder=i+.5)
        ax4.scatter(x, ME_mfpt_left, color=colors[i], marker=marker['left'],
                    s=12, edgecolors='k', linewidth=0.5, zorder=i+.5,
                    label="left species")
        ax4.scatter(x, ME_mfpt_right, color=colors[i], marker=marker['right'],
                    s=12, edgecolors='k', linewidth=0.5, zorder=i+.5,
                    label="right species")

    # ax4.legend(custom_lines, legend, title=r'Model')
    ax4.set_yscale('log')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.1, 10.])
    #ax4.legend()
    # ax4.set_xlim([np.min(N_func), np.max(N_func)])
    legend1 = ax4.legend(col_lines, col_label, title=r"$s$", loc=(0.725, 0.475))
    ax4.legend(handles=marker_lines, loc='lower center')
    ax4.add_artist(legend1)
    
    # save figures
    fig4.savefig(filename + '_condfix.pdf')
    fig4.savefig(filename + '_condfix.png')

    return s

def conditional_fitness_spatial_xmax(s, N, times, cmap_name, filename, model):

    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = [cmap(nbr) for nbr in np.linspace(0.0, 0.8, num=len(s))]
    linestyle = [':', '-', '--']
    marker = {'right' : 'o', 'left' : '^'}

    # colour legend
    col_label = s  # ["N={}".format(nbr) for nbr in N]
    col_lines = [Line2D([0], [0], color=col) for col in colors]
    cl = plt.legend(col_lines, col_label, loc='lower right')
    
    marker_lines = [Line2D([0], [0], marker=marker['right'], color='k',
                           markerfacecolor='w', label="right species",
                           linestyle=''),
                    Line2D([0], [0], marker=marker['left'], color='k',
                           markerfacecolor='w', label="left species",
                           linestyle='')
                    ]

    # Fig1A : P_absorbing
    fig4, ax4 = plt.subplots(figsize=(3.4, 2.5))
    ax4.set(title=r"",
            xlabel=r"Equiprobable takeover abundance, $f_{\mathrm{max}}$",
            ylabel=r"Conditional fixation time")

    moran = []
    space = []
    for i, fit in enumerate(s):
         # define models
        moran.append(MoranFPT(fit, 1.0, N, times))
        space.append(model(fit, 1.0, N, times))

        # plots
        x = []
        
        # Fig 1 D
        # ME approach
        ME_prob, ME_mfpt = moran[i].probability_mfpt(moran[i].nmax)
        ax4.scatter(moran[i].xmax, ME_mfpt[1], color=colors[i],
                    marker=marker['left'],
                    s=12, edgecolors='k', linewidth=0.5, zorder=i+.5,
                    label="left species")
        ME_prob, ME_mfpt = space[i].probability_mfpt(space[i].nmax)
        ax4.scatter(space[i].xmax, ME_mfpt[0], color=colors[i],
                    marker=marker['right'],
                    s=12, edgecolors='k', linewidth=0.5, zorder=i+.5,
                    label="right species")

    # ax4.legend(custom_lines, legend, title=r'Model')
    ax4.set_yscale('log')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.1, 10.])
    #ax4.legend()
    # ax4.set_xlim([np.min(N_func), np.max(N_func)])
    ax4.legend(handles=marker_lines)
    
        # save figures
    fig4.savefig(filename + '_cond_xmax.pdf')
    fig4.savefig(filename + '_cond_xmax.png')

    return s


if __name__ == '__main__':
    # mkdir
    dir = 'figures_suppl'
    Path(dir).mkdir(parents=True, exist_ok=True)

    # for certain distribution functions
    times = np.linspace(0.0, 100.0, 10001)

    # theory parameters
    model = Spatial
    model2 = SpatInv

    # colormaps
    cmap_name1 = 'plasma'
    cmap_name2 = 'viridis'

    # Figure S1 : ME vs FP moran model
    fname1 = dir + os.sep + 'moran'
    # moranME_FP(times, fname1, MoranFPT)

    # Figure SX : invasion model check
    s = [1., 10., 100.]
    K = 100
    fnameX = dir + os.sep + 'inv_check'
    # invasion_check(s, K, cmap_name1, fnameX, model2)

    # Comparing invasions
    fnameY = dir + os.sep + 'inv_compare'
    # side_invasion(4, cmap_name1, cmap_name2, fnameY, model)

    # Average quantities
    fnameZ = dir + os.sep + 'average'
    # average(4, cmap_name1, cmap_name2, fnameZ, model)

    # moran selection me vs fp analytical
    s0 = [1.1, 2, 10]
    fnameA = dir + os.sep + 'moran'
    # moran_selection(fnameA, s0[0], s0[1], s0[2], cmap_name2)
    
    # moran , me vs fp analytical
    N = [10, 100, 1000]
    fnameR = dir + os.sep + 'SvM-alt'
    # spatial_vs_moran(N, times, cmap_name2, fnameR, Spatial, True)

    # fitness results, me vs fp analytical
    s = [1., 10., 100.]
    K = 100
    fnameS = dir + os.sep + 'fit-alt'
    # fitness_spatial(s, K, times, cmap_name1, fnameS, Spatial, True)

    # Comparing tdet with exact
    fnameB = dir + os.sep + 'approx'
    # det_time(s, cmap_name1, fnameB, model)
    
    # Figure SY : fitness results, less fitness difference
    s = [1., 1.1, 2.0]
    K = 100
    fname2 = dir + os.sep + 'fit_small'
    # fitness_spatial(s, K, times, cmap_name1, fname2, Spatial)

    # Figure SZ : spatial vs moran
    N = [10, 100, 1000]
    fname1 = dir + os.sep + 'SvM'
    conditional_spatial_vs_moran(N, times, cmap_name2, fname1, Spatial)

    s = [1., 10., 100.]
    K = 100
    fname2 = dir + os.sep + 'fit'
    conditional_fitness_spatial(s, K, times, cmap_name1, fname2, Spatial)
    
    s = [1., 10., 100.]
    K = 100
    fname2 = dir + os.sep + 'fit'
    # conditional_fitness_spatial_xmax(s, K, times, cmap_name1, fname2, Spatial)
    
    
