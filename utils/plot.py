def getErrorPlot(data, num_steps=100):
    x = list(range(0, data.shape[1], num_steps))
    y = np.empty(0)
    std = np.empty(0)
    for j in range(0, data.shape[1], num_steps):
        xj = data[:, j]
        y = np.append(y, np.median(xj))
        std = np.append(std, standarDeviation(xj))
    return x, y, std

def plotGraphs(graph):
    plt.plot(list(range(1,101)), np.average(graph, axis=0), label='Average')
    plt.plot(list(range(1,101)), np.median(graph, axis=0), label='Median')
    plt.plot(list(range(1,101)), np.min(graph, axis=0), label='Min')
    plt.plot(list(range(1,101)), np.max(graph, axis=0), label='Max')
    x, y, z = getErrorPlot(graph)
    plt.errorbar(x, y, z, marker='^', ls='None')
    plt.legend(loc='upper right', shadow=True, fontsize = "xx-small")
    plt.xlabel('iterations')
    plt.ylabel('fitness')