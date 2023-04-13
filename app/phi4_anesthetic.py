import anesthetic as ns
import anesthetic.convert
import matplotlib.pyplot as plt

samples = ns.read_chains("cmake-build-debug/app/Phi4")
posterior = samples.posterior_points()

dist = ns.convert.to_getdist(samples)


#samples.gui()
plt.hist(abs(posterior['m']))
plt.show()