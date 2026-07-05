# patch16_covariance_fix.py
# fitter.py — correct the parameter-uncertainty SCALE factor.
#
# scipy.least_squares returns a Jacobian J of the (weighted) residuals we
# passed, so (J^T J)^-1 is effectively (J^T W J)^-1. The bug was the variance
# scale: the code used sqrt(result.cost/(m-p)) with result.cost = 0.5*sum(res^2),
# which UNDER-reports errors. The correct reduced-chi-square scale is
# s^2 = 2*result.cost/(m-p) = sum(res^2)/(m-p), applied inside the sqrt with the
# covariance diagonal. This now matches scipy.optimize.curve_fit's pcov exactly
# (verified: identical sigmas to 1e-6).
#
# NOTE: the Brug C_eff formula in eis_interpreter.py was reviewed for a possible
# Hirschorn (2010) upgrade but LEFT UNCHANGED: the surface-distribution form
# gives the wrong limit for R_s << R_ct (our carbon spectra) and choosing the
# right Hirschorn variant needs knowledge of the physical distribution type,
# which is ambiguous for mesoporous carbon. The simple Brug form, already
# labelled "approximate" with an n<0.80 dispersion warning, is the honest choice.
import shutil, sys

PATH = "eisforge/core/fitter.py"
s = open(PATH, encoding="utf-8").read()

OLD = '''                J   = result.jac
                cov = np.linalg.pinv(J.T @ J)
                diag = np.diag(cov)
                std_devs = np.sqrt(np.abs(diag)) * np.sqrt(result.cost / max(len(result.fun) - len(fitted), 1))
                for i in range(min(n, len(std_devs))):
                    if np.isfinite(std_devs[i]):
                        errors[param_names[i]] = float(std_devs[i])'''

NEW = '''                J   = result.jac
                cov = np.linalg.pinv(J.T @ J)          # (J^T W J)^-1: J already weighted
                diag = np.diag(cov)
                dof = max(len(result.fun) - len(fitted), 1)
                # reduced chi-square scale: result.cost = 0.5*sum(res^2)
                #   -> s^2 = 2*cost/dof = sum(res^2)/dof  (matches scipy.curve_fit)
                s_sq = 2.0 * result.cost / dof
                std_devs = np.sqrt(np.abs(diag) * s_sq)
                for i in range(min(n, len(std_devs))):
                    if np.isfinite(std_devs[i]):
                        errors[param_names[i]] = float(std_devs[i])'''

if NEW in s:
    print("Already applied.")
    sys.exit(0)
if OLD not in s:
    print("ERROR: OLD block not found. Aborting.")
    sys.exit(1)
shutil.copy(PATH, PATH + ".bak_patch16")
open(PATH, "w", encoding="utf-8").write(s.replace(OLD, NEW, 1))
print("Patched OK:", PATH)
