
##################################
# Training
##################################
for iStep in range(200000):
    #     if iStep % 10000 == 0:
    #         print(iStep)
    idx_I = (iStep // 2000) % N_IMG
    I_this = INP_lst[idx_I]  # + param['noise_mag'] * np.random.normal(0, 1, N_INP)
    x, U = update(x, U, I_this, param)

##################################
# Testing
##################################
N_STEP_TEST = 20000
times_lst = np.linspace(0, N_STEP_TEST - 1, N_STEP_TEST) * param['dt']
x_data = []
err = np.zeros((N_STEP_TEST, N_LAYER))
for iStep in range(N_STEP_TEST):
    idx_I = (iStep // 2000) % N_IMG
    I_this = INP_lst[idx_I] + param['noise_mag'] * np.random.normal(0, 1, N_INP)
    x = update(x, U, I_this, param, WITH_SP=False)
    x_data += [x]

    for iLayer in range(N_LAYER):
        Ieff = I_this if iLayer == 0 else x[iLayer - 1]
        err[iStep][iLayer] = np.linalg.norm(Ieff - U[iLayer].dot(x[iLayer])) / np.sqrt(len(Ieff))


##################################
# Plotting
##################################
def backprop_receptive(U, depth, iLayer, iX):
    return U[iLayer][:, iX] if depth == 0 else U[iLayer].dot(backprop_receptive(U, depth - 1, iLayer + 1, iX))


# Receptive fields
for iLayer in range(N_LAYER):
    fig, ax = plt.subplots(ncols=N_X[iLayer], figsize=(3 * N_X[iLayer], 3))
    fig.suptitle("Receptive Fields, Layer " + str(iLayer))
    for iX in range(N_X[iLayer]):
        ax[iX].imshow(backprop_receptive(U, iLayer, 0, iX).reshape((NPIX_ROW, NPIX_COL)))
        ax[iX].set_title("X" + str(iLayer) + str(iX))
    plt.show()

# Representation values
fig, ax = plt.subplots(ncols=N_LAYER, figsize=(5 * N_LAYER, 5))
fig.suptitle("Representation Values")
for iLayer in range(N_LAYER):
    xthis = np.array([x[iLayer] for x in x_data])
    ax_this = ax[iLayer] if N_LAYER > 1 else ax  # In matlab ax is not a list if it has only one element :(
    for iX in range(N_X[iLayer]):
        ax_this.plot(times_lst, xthis[:, iX])
        ax_this.set_title("Layer " + str(iLayer))
plt.show()

# Errors
plt.figure()
plt.title("Representation error")
for iLayer in range(N_LAYER):
    plt.plot(times_lst, err[:, iLayer], label="Layer " + str(iLayer))
plt.legend()
plt.show()