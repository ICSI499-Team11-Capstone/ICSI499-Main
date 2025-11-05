
from plotRun import genPlotForRun
filename = 'a0.005lds19b0.007g2.5d1h14'
genPlotForRun(runsPath="./runs/weighted/", run=filename + ".npz", graphsPath="./graphs/weighted", graph=filename + ".png")
