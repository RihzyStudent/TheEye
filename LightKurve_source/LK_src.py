import lightkurve as lk
import matplotlib.pyplot as plt
lc = lk.search_lightcurve("KIC 10797460", mission="Kepler").download()
# lc = lc.remove_nans().remove_outliers().flatten()
# print(lc.sap_flux)
#
# ax = lc.plot(label="KIC 10797460")
# plt.savefig("lightcurve.png", dpi=300, bbox_inches="tight")  # high-res PNG
# plt.close()
#
import numpy as np
import matplotlib.pyplot as plt

good = lc.QUALITY == 0
time = lc.time.value[good]
sap  = (lc.SAP_FLUX[good] / np.nanmedian(lc.SAP_FLUX[good])).value

plt.plot(time, sap, ".", markersize=2)
plt.xlabel("Time (BJD - 2454833)")
plt.ylabel("Normalized SAP flux")
plt.savefig("sap_normalized.png", dpi=300, bbox_inches="tight"); plt.close()

