#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:57:16 2023.

@author: Judith Angel

Create plots and gifs for bathymetry reconstruction.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dedalus.extras.plot_tools import quad_mesh, pad_limits
import importlib

folder = "2024_02_19_09_52_AM"
path = "ProblemRelatedFiles/" + folder

params = importlib.import_module("ProblemRelatedFiles." + folder + ".params")
pa = params.params()
if pa.data != "sim_everywhere":
    pos = pa.pos
save = True

with h5py.File(path+"/opt_data.hdf5", "r") as sol:

    f_vals = np.array(sol["f_vals"][:])
    f_err1s = np.array(sol["f_err1s"][:])
    f_err2s = np.array(sol["f_err2s"][:])
    f_regs = np.array(sol["f_regs"][:])
    f_b_exct = np.array(sol["f_b_exct"])
    alpha_js = np.array(sol["alpha_js"][:])
    v_norms = np.array(sol["v_norms"][:])
    b_exact = np.array(sol["b_exact"][:])
    bs = np.array(sol["bs"][:])
    vs = np.array(sol["vs"][:])
    saveall = sol.attrs.get("saveall")
    if saveall:
        qs = np.array(sol["qs"][:])
        ps = np.array(sol["ps"][:])
    else:
        q = np.array(sol["q"][:])
    obs = np.array(sol["obs"][:])
    x = np.array(sol["x"][:])
    t = np.array(sol["t"][:])
    j = sol.attrs.get("jmax")

b_errs = np.zeros(j)

for i in range(j):
    if np.linalg.norm(b_exact) > 1e-16:
        b_errs[i] = np.linalg.norm(bs[i]-b_exact)/np.linalg.norm(b_exact)
    else:
        b_errs[i] = np.linalg.norm(bs[i]-b_exact)

# Range for plots.
bmin = np.min(b_exact)
bmax = np.max(b_exact)
ymin = np.min(obs)
ymax = np.max(obs)
if saveall:
    p1min = np.min(ps[:, :, :, 0])
    p1max = np.max(ps[:, :, :, 0])
    p2min = np.min(ps[:, :, :, 1])
    p2max = np.max(ps[:, :, :, 1])

plt.figure()
plt.semilogy(range(j-1), f_vals[0:j-1], '-*', label=r"value at $b_j$")
plt.semilogy(range(j-1), f_err1s[0:j-1]+f_err2s[0:j-1], '--', label="mismatch")
plt.semilogy(range(j-1), f_regs[0:j-1], '--', label="regularisation")
plt.semilogy(range(j-1), f_b_exct*np.ones(j-1), 'r-', label="value at exact b")
plt.xlabel(r'Optimisation iteration $j$')
# plt.ylabel(r"$J(b_j)$")
plt.legend()
plt.title("Values of objective functional")
if save:
    plt.savefig(path + "/values_f.pdf", bbox_inches='tight')

plt.figure()
plt.semilogy(range(j), b_errs[0:j], '-*')
plt.xlabel('Optimisation iteration')
plt.ylabel(r"$||\cdot||_2 \ / \ ||b||_2$")
plt.title("Relative error to exact bathymetry")
if save:
    plt.savefig(path + "/errors_rel.pdf", bbox_inches='tight')

if j > 1:
    plt.figure()
    plt.semilogy(range(1, j+1), alpha_js[0:j], '-*')
    plt.xlabel('iteration')
    plt.ylabel(r'$\alpha$')
    plt.title("Chosen step size")
    if save:
        plt.savefig(path + "/stepsize.pdf", bbox_inches='tight')

    plt.figure()
    plt.semilogy(range(1, j+1), v_norms[0:j], '-*')
    plt.xlabel('Optimisation iteration')
    plt.ylabel(r"$||\cdot||$")
    plt.title("Norm of the gradient")
    if save:
        plt.savefig(path + "/norms.pdf", bbox_inches='tight')

#######################
# ----- Bathymetries ----- #

plt.figure()
plt.plot(x, b_exact, label="exact")
plt.plot(x, bs[0], label="computed")
if pa.data != "sim_everywhere":
    plt.plot(pos, np.zeros(len(pos)), "k*", label="sensor position")
plt.ylim([min(np.min(b_exact), np.min(bs)),
          max(np.max(b_exact), np.max(bs))])
plt.xlabel('x [m]')
plt.ylabel('b [m]')
plt.legend()
plt.title("Exact bathymetry and initial guess")
if save:
    plt.savefig(path + "/bathymetry_0.pdf", bbox_inches='tight')

if bs.shape[0] > 1:
    plt.figure()
    plt.plot(x, b_exact, label="exact")
    plt.plot(x, bs[1], label="computed")
    if pa.data != "sim_everywhere":
        plt.plot(pos, np.zeros(len(pos)), "k*", label="sensor position")
    plt.ylim([min(np.min(b_exact), np.min(bs)),
              max(np.max(b_exact), np.max(bs))])
    plt.xlabel('x [m]')
    plt.ylabel('b [m]')
    plt.legend()
    plt.title(r"Exact and computed bathymetry at $j=1$")
    if save:
        plt.savefig(path + "/bathymetry_1.pdf", bbox_inches='tight')


if bs.shape[0] > 2:
    plt.figure()
    plt.plot(x, b_exact, label="exact")
    plt.plot(x, bs[2], label="computed")
    if pa.data != "sim_everywhere":
        plt.plot(pos, np.zeros(len(pos)), "k*", label="sensor position")
    plt.ylim([min(np.min(b_exact), np.min(bs)),
              max(np.max(b_exact), np.max(bs))])
    plt.xlabel('x [m]')
    plt.ylabel('b [m]')
    plt.legend()
    plt.title(r"Exact and computed bathymetry at $j=2$")
    if save:
        plt.savefig(path + "/bathymetry_2.pdf", bbox_inches='tight')

fig1 = plt.figure()

plt.figure()
plt.plot(x, b_exact, label="exact")
plt.plot(x, bs[j], label="computed")
if pa.data != "sim_everywhere":
    plt.plot(pos, np.zeros(len(pos)), "k*", label="sensor position")
plt.ylim([min(np.min(b_exact), np.min(bs)),
          max(np.max(b_exact), np.max(bs))])
plt.xlabel('x [m]')
plt.ylabel('b [m]')
plt.legend()
plt.title(f"Exact and computed bathymetry at $j={{{j}}}$")
if save:
    plt.savefig(path + "/bathymetry_" + str(j-1) + ".pdf", bbox_inches='tight')

fig1 = plt.figure()


def update_plot1(i):

    plt.clf()
    plt.plot(x, b_exact, label="exact")
    plt.plot(x, bs[i], label="computed")
    if pa.data != "sim_everywhere":
        plt.plot(pos, np.zeros(len(pos)), "k*")
    plt.ylim([min(np.min(b_exact), np.min(bs)),
              max(np.max(b_exact), np.max(bs))])
    plt.xlabel('x [m]')
    plt.ylabel('b [m]')
    plt.legend()
    plt.title(f"Exact and computed bathymetry at j={i}")


anim1 = FuncAnimation(
    fig1, update_plot1, frames=np.arange(0, j, max(1, j//20)))
if save:
    anim1.save(path + "/bathymetries.gif", dpi=150, fps=2)

############################
# ----- Water height ----- #
if saveall:

    fig2 = plt.figure()

    def update_plot2(i):

        plt.clf()
        xmesh, ymesh = quad_mesh(x=x, y=t)
        plt.pcolormesh(xmesh, ymesh, qs[i, :, :, 0]+bs[i], cmap='plasma',
                       vmin=ymin, vmax=ymax, rasterized=True)
        plt.axis(pad_limits(xmesh, ymesh))
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f"Water height at j={i}")

    anim2 = FuncAnimation(fig2, update_plot2, frames=np.arange(
        0, j-1, max(1, (j-1)//20)))
    if save:
        anim2.save(path + "/states.gif", dpi=150, fps=2)

############################
# ----- Mismatch ----- #
if saveall:

    fig3 = plt.figure()
    mmin = np.min(qs[0, :, :, 0]+bs[0]-obs)
    mmax = np.max(qs[0, :, :, 0]+bs[0]-obs)

    def update_plot3(i):

        plt.clf()
        xmesh, ymesh = quad_mesh(x=x, y=t)
        plt.pcolormesh(xmesh, ymesh, qs[i, :, :, 0]+bs[i]-obs, cmap='plasma',
                       vmin=mmin, vmax=mmax, rasterized=True)
        plt.axis(pad_limits(xmesh, ymesh))
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f"Mismatch j={i}")

    anim3 = FuncAnimation(fig3, update_plot3, frames=np.arange(
        0, j-1, max(1, (j-1)//20)))
    if save:
        anim3.save(path + "/mismatch.gif", dpi=150, fps=2)

#################################################
# ----- Adjoint variable, first component ----- #
if saveall:

    fig4 = plt.figure()

    def update_plot4(i):

        plt.clf()
        xmesh, ymesh = quad_mesh(x=x, y=t)
        plt.pcolormesh(xmesh, ymesh, ps[i, :, :, 0], cmap='plasma', vmin=p1min,
                       vmax=p1max, rasterized=True)
        plt.axis(pad_limits(xmesh, ymesh))
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f"Adjoint, first component at j={i}")

    anim4 = FuncAnimation(fig4, update_plot4, frames=np.arange(
        0, j-1, max(1, (j-1)//20)))
    if save:
        anim4.save(path + "/adjoints1.gif", dpi=150, fps=2)

##################################################
# ----- Adjoint variable, second component ----- #
if saveall:

    fig5 = plt.figure()

    def update_plot5(i):

        plt.clf()
        xmesh, ymesh = quad_mesh(x=x, y=t)
        plt.pcolormesh(xmesh, ymesh, ps[i, :, :, 1], cmap='plasma', vmin=p2min,
                       vmax=p2max, rasterized=True)
        plt.axis(pad_limits(xmesh, ymesh))
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f"Adjoint, second component at j={i}")

    anim5 = FuncAnimation(fig5, update_plot5, frames=np.arange(
        0, j-1, max(1, (j-1)//20)))
    if save:
        anim5.save(path + "/adjoints2.gif", dpi=150, fps=2)

################################
# ----- Error in bathymetry ----- #
diff = np.zeros((j, b_exact.size))

for i in range(j):

    diff[i] = bs[i]-b_exact

fig6 = plt.figure()


def update_plot6(i):

    plt.clf()
    plt.plot(x, diff[i])
    plt.ylim([np.min(diff), np.max(diff)])
    plt.xlabel('x')
    plt.title(f"Difference bathymetry to exact bathymetry at j={i}")


anim6 = FuncAnimation(fig6, update_plot6, frames=np.arange(
    0, j, max(1, j//20)))
if save:
    anim6.save(path + "/errors.gif", dpi=150, fps=2)

########################
# ----- Gradient ----- #
fig7 = plt.figure()


def update_plot7(i):

    plt.clf()
    plt.plot(x, vs[i])
    plt.ylim([np.min(vs), np.max(vs)])
    plt.xlabel('x')
    plt.title(f"Gradient at j={i}")


anim7 = FuncAnimation(fig7, update_plot7, frames=np.arange(
    0, j-1, max(1, (j-1)//20)))
if save:
    anim7.save(path + "/gradients.gif", dpi=150, fps=2)

###########################
# ----- Observation ----- #
xmesh, ymesh = quad_mesh(
    x=x, y=t)
plt.figure()
plt.pcolormesh(xmesh, ymesh, obs, cmap='plasma', rasterized=True)
plt.axis(pad_limits(xmesh, ymesh))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title("Observation")
if save:
    plt.savefig(path + "/observation.png", bbox_inches='tight')

########################
# ----- Bathymetry and free surface elevation last iteration ----- #
if saveall:
    H = qs[j-1, :, :, 0] + bs[j-1]
else:
    H = q[:, :, 0] + bs[j-1]
Hmax = np.amax(H)
fig10 = plt.figure()
mask = obs == 0
obs[mask] = np.nan
start = np.argmin(abs(t-4))


def update_plot10(i):

    plt.clf()
    plt.plot(x, b_exact, "k", label="exact")
    plt.plot(x, bs[j-1], "tab:orange", label="computed")
    plt.plot(x, obs[i], "k*")
    plt.plot(x, H[i], "tab:orange")
    plt.ylim([0, Hmax+0.01])
    plt.xlabel('x [m]')
    plt.ylabel('H [m]')
    plt.legend()
    plt.title(f"Time t={round(t[i], 1)}")


anim10 = FuncAnimation(fig10, update_plot10, frames=np.arange(
    start, t.size-1, max(1, (t.size-1)//100)))
if save:
    anim10.save(path + "/Hb.gif", dpi=150, fps=2)

########################
# ----- Surface elevation and observation at sensor positions ----- #
if pa.data != "sim_everywhere":

    pos_n = len(pos)

    if pos_n > 1:
        posi = np.zeros((pos_n), dtype=int)
        fig, axs = plt.subplots(pos_n)

        for i in range(pos_n):

            posi[i] = np.argmin(abs(x-pos[i]))

        fig.tight_layout(pad=2.0)

        for i in range(pos_n):

            axs[i].plot(t[start:], H[start:, posi[i]], "y", label=r"$H$")
            axs[i].plot(t[start:], obs[start:, posi[i]], "k--", label=r"$H_{obs}$")
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel('H [m]')
            axs[i].set_title(f"Sensor {i+2} at {pos[i]}m")
            axs[i].set_box_aspect(1/8)
            if i == 0:
                axs[i].legend(fontsize="8", loc="upper left")

        plt.tight_layout()
        if save:
            plt.savefig(path + "/H_Hobs.pdf", bbox_inches='tight')
        plt.show()

        fig, axs = plt.subplots(pos_n)
        fig.tight_layout(pad=2.0)

        for i in range(pos_n):

            axs[i].plot(t[start:], H[start:, posi[i]]-obs[start:, posi[i]])
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel(r'$H - H_{obs} \ [m]$')
            axs[i].set_title(f"Sensor {i+2} at {pos[i]}m")

        plt.tight_layout()
        if save:
            plt.savefig(path + "/H-Hobs.pdf", bbox_inches='tight')
        plt.show()

    else:

        pos1 = np.argmin(abs(x-pos[0]))
        plt.figure()
        plt.plot(t[start:], H[start:, pos1], "y")
        plt.plot(t[start:], obs[start:, pos1], "k--")
        plt.xlabel('Time [s]')
        plt.ylabel('H [m]')
        plt.title(f"Sensor 2 at {pos[0]}m")
        if save:
            plt.savefig(path + "/H_Hobs.pdf", bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(
            t[start:],
            H[start:, pos1]-obs[start:, pos1])
        plt.xlabel('Time [s]')
        plt.ylabel(r'$H - H_{obs} \ [m]$')
        plt.title(f"Sensor 2 at {pos[0]}m")
        if save:
            plt.savefig(path + "/H-Hobs.pdf", bbox_inches='tight')
        plt.show()
