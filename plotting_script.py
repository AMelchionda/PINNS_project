import matplotlib.pyplot as plt
import numpy as np

#  LOADING DATA --------------------------------------------------------------------------------------------------------

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# # NB: SOLO Trial 4 e Trial 5 hanno, per Prediction_Vessel, i TEMPI messi nel modo giusto
load_from = 'noise5'

path_inputs = './Inputs/'
path_predictions = './Results/' + load_from + '/'
path_figures = './New Plots/'

N_f = 2000

# load measurement data(used in training)
input_vessel_1 = np.load(path_inputs + "input_1.npy").item()
output_vessel_2 = np.load(path_inputs + "output_2.npy").item()
output_vessel_3 = np.load(path_inputs + "output_3.npy").item()

t = input_vessel_1["Time"][:, None]  # i tempi dei input e test sono i medesimi (413 valori da 0s a 3.3s)

# load test data(used in plot)
test_vessel_1 = np.load(path_inputs + "test_1.npy").item()
test_vessel_2 = np.load(path_inputs + "test_2.npy").item()
test_vessel_3 = np.load(path_inputs + "test_3.npy").item()

velocity_measurements_vessel1 = input_vessel_1["Velocity"][:, None]
velocity_measurements_vessel2 = output_vessel_2["Velocity"][:, None]
velocity_measurements_vessel3 = output_vessel_3["Velocity"][:, None]

area_measurements_vessel1 = input_vessel_1["Area"][:, None]
area_measurements_vessel2 = output_vessel_2["Area"][:, None]
area_measurements_vessel3 = output_vessel_3["Area"][:, None]

velocity_test_vessel1 = test_vessel_1["Velocity"][:, None]
velocity_test_vessel2 = test_vessel_2["Velocity"][:, None]
velocity_test_vessel3 = test_vessel_3["Velocity"][:, None]

pressure_test_vessel1 = test_vessel_1["Pressure"][:, None]
pressure_test_vessel2 = test_vessel_2["Pressure"][:, None]
pressure_test_vessel3 = test_vessel_3["Pressure"][:, None]

# load predicted data from test
Prediction_Vessel1 = np.load(path_predictions + "Prediction_Test1.npy").item()
Prediction_Vessel2 = np.load(path_predictions + "Prediction_Test2.npy").item()
Prediction_Vessel3 = np.load(path_predictions + "Prediction_Test3.npy").item()
# PUOI usare Prediction_Vessel anzichè Prediction_Test ma SOLO con TRIAL 4 e TRIAL 5
# L'if qua sotto sarà utile solo con Prediction Vessel, altrimenti è inutile (ma non pericoloso)
# if( load_from == 'Trial 4' or load_from == 'Trial 5') :
# T = Prediction_Vessel1["Time"]
# else:
T = t


U_predicted_vessel1 = Prediction_Vessel1["Velocity"]
U_predicted_vessel2 = Prediction_Vessel2["Velocity"]
U_predicted_vessel3 = Prediction_Vessel3["Velocity"]

p_predicted_vessel1 = Prediction_Vessel1["Pressure"]
p_predicted_vessel2 = Prediction_Vessel2["Pressure"]
p_predicted_vessel3 = Prediction_Vessel3["Pressure"]

# riduco di 1 dim nel caso di trials 1,2 e 3
if( load_from == 'Sensitivity Analysis 2/7x20' or load_from == 'Trial 2' or load_from == 'Trial 3'):
    U_predicted_vessel1 = np.squeeze(U_predicted_vessel1, axis=2)
    U_predicted_vessel2 = np.squeeze(U_predicted_vessel2, axis=2)
    U_predicted_vessel3 = np.squeeze(U_predicted_vessel3, axis=2)

    p_predicted_vessel1 = np.squeeze(p_predicted_vessel1, axis=2)
    p_predicted_vessel2 = np.squeeze(p_predicted_vessel2, axis=2)
    p_predicted_vessel3 = np.squeeze(p_predicted_vessel3, axis=2)

# load interface prediction
Prediction_Interface1 = np.load(path_predictions + "Prediction_Interface1.npy").item()
Prediction_Interface2 = np.load(path_predictions + "Prediction_Interface2.npy").item()
Prediction_Interface3 = np.load(path_predictions + "Prediction_Interface3.npy").item()

U_predicted_interface1 = Prediction_Interface1["Velocity"]
U_predicted_interface2 = Prediction_Interface2["Velocity"]
U_predicted_interface3 = Prediction_Interface3["Velocity"]

A_predicted_interface1 = Prediction_Interface1["Area"]
A_predicted_interface2 = Prediction_Interface2["Area"]
A_predicted_interface3 = Prediction_Interface3["Area"]

p_predicted_interface1 = Prediction_Interface1["Pressure"]
p_predicted_interface2 = Prediction_Interface2["Pressure"]
p_predicted_interface3 = Prediction_Interface3["Pressure"]

tempo = Prediction_Interface1["Time"]

# riduco di 1 dim nel caso di trials 1,2 e 3
if load_from == 'Sensitivity Analysis 2/7x20' or load_from == 'Trial 2' or load_from == 'Trial 3':
    tempo = np.squeeze(tempo, axis=2)
    U_predicted_interface1 = np.squeeze(U_predicted_interface1, axis=2)
    U_predicted_interface2 = np.squeeze(U_predicted_interface2, axis=2)
    U_predicted_interface3 = np.squeeze(U_predicted_interface3, axis=2)

    A_predicted_interface1 = np.squeeze(A_predicted_interface1, axis=2)
    A_predicted_interface2 = np.squeeze(A_predicted_interface2, axis=2)
    A_predicted_interface3 = np.squeeze(A_predicted_interface3, axis=2)

    p_predicted_interface1 = np.squeeze(p_predicted_interface1, axis=2)
    p_predicted_interface2 = np.squeeze(p_predicted_interface2, axis=2)
    p_predicted_interface3 = np.squeeze(p_predicted_interface3, axis=2)

#  PLOTTING ------------------------------------------------------------------------------------------------------------

# Plot VELOCITY comparision between prediction and reference measurements in 3 vessels
fig1 = plt.figure(1, figsize=(22, 12), dpi=100, facecolor='w', frameon=False)

ax11 = fig1.add_subplot(131)
ax12 = fig1.add_subplot(132)
ax13 = fig1.add_subplot(133)

ax11.plot(T, U_predicted_vessel1, 'r-', linewidth=3.0, markersize=0.7, label='Predicted velocity Vessel1')
ax11.plot(t, velocity_test_vessel1, 'b--', linewidth=3.0, markersize=0.7, label='Reference velocity Vessel1')
ax11.legend(loc='upper right', frameon=False, fontsize='large')
ax11.set_ylim([-0.5, 2.9])


ax12.plot(T, U_predicted_vessel2, 'r-', linewidth=3.0, markersize=0.7, label='Predicted velocity Vessel2')
ax12.plot(t, velocity_test_vessel2, 'b--', linewidth=3.0, markersize=0.7, label='Reference velocity Vessel2')
ax12.legend(loc='upper right', frameon=False, fontsize='large')
ax12.set_ylim([-0.01, 3.7])

ax13.plot(T, U_predicted_vessel3, 'r-', linewidth=3.0, markersize=0.7, label='Predicted velocity Vessel3')
ax13.plot(t, velocity_test_vessel3, 'b--', linewidth=3.0, markersize=0.7,label='Reference velocity Vessel3')
ax13.legend(loc='upper right', frameon=False, fontsize='large')
ax13.set_ylim([-0.3, 2.32])

fig1.suptitle('Comparative velocity', size=30)
ax11.set_xlabel("t[s]", size=20)
ax11.set_ylabel("Velocity[m/s]", size=20)
ax12.set_xlabel("t[s]", size=20)
ax12.set_ylabel("Velocity[m/s]", size=20)
ax13.set_xlabel("t[s]", size=20)
ax13.set_ylabel("Velocity[m/s]", size=20)

fig1.savefig(path_figures + "Comparative_Velocity.png")  # Save figure



# Plot PRESSURE comparision between prediction and reference measurements in 3 vessels
fig2 = plt.figure(2, figsize=(22, 12), dpi=300, facecolor='w', frameon=False)

ax21 = fig2.add_subplot(131)
ax22 = fig2.add_subplot(132)
ax23 = fig2.add_subplot(133)

ax21.plot(T, p_predicted_vessel1, 'r-', linewidth=3.0, markersize=0.7, label='Predicted pressure Vessel1')
ax21.plot(t, pressure_test_vessel1, 'b--', linewidth=3.0, markersize=0.7, label='Reference pressure Vessel1')
ax21.legend(loc='upper right', frameon=False, fontsize='large')
ax21.set_ylim([0, 105000])

ax22.plot(T, p_predicted_vessel2, 'r-', linewidth=3.0, markersize=0.7, label='Predicted pressure Vessel2')
ax22.plot(t, pressure_test_vessel2, 'b--', linewidth=3.0, markersize=0.7, label='Reference pressure Vessel2')
ax22.legend(loc='upper right', frameon=False, fontsize='large')
ax22.set_ylim([0, 98000])

ax23.plot(T, p_predicted_vessel3, 'r-', linewidth=3.0, markersize=0.7, label='Predicted pressure Vessel3')
ax23.plot(t, pressure_test_vessel3, 'b--', linewidth=3.0, markersize=0.7, label='Reference pressure Vessel3')
ax23.legend(loc='upper right', frameon=False, fontsize='large')
ax23.set_ylim([0, 105000])

fig2.suptitle('Comparative pressure', size=30)
ax21.set_xlabel("t[s]", size=20)
ax21.set_ylabel("Pressure[Pa]", size=20)
ax22.set_xlabel("t[s]", size=20)
ax22.set_ylabel("Pressure[Pa]", size=20)
ax23.set_xlabel("t[s]", size=20)
ax23.set_ylabel("Pressure[Pa]", size=20)

fig2.savefig(path_figures + "Comparative_Pressure.png")  # Save figure

# plot conservation of MOMENTUM

# Compute flow in vessels
Q1 = A_predicted_interface1 * U_predicted_interface1
Q2 = A_predicted_interface2 * U_predicted_interface2
Q3 = A_predicted_interface3 * U_predicted_interface3

Q_in = Q1
Q_out = Q2 + Q3

# pp_predicted_interface1 = 69673881.97*(np.sqrt(A_predicted_interface1) - np.sqrt(1.35676200E-05))
# pp_predicted_interface2 = 541788704.42*(np.sqrt(A_predicted_interface2) - np.sqrt(1.81458400E-06))
# pp_predicted_interface3 = 69549997.97*(np.sqrt(A_predicted_interface3) - np.sqrt(1.35676200E-05))

# p1_diff = p_predicted_interface1 - pp_predicted_interface1
# fig3 = plt.figure(3, figsize=(22, 12), dpi=300, facecolor='w', frameon=False)
# ax31 = fig3.add_subplot(111)
# # ax32 = fig3.add_subplot(122)
# ax31.plot(tempo, p_predicted_interface1, 'r--', linewidth=3.0, markersize=0.7, label='p_predicted_interface1')
# ax31.plot(tempo, pp_predicted_interface1, 'b--', linewidth=3.0, markersize=0.7, label='pp_predicted_interface1')
# plt.show()
#
# # Compute the momentum in vessels
# p_1 = p_predicted_interface1 + (0.5 * U_predicted_interface1 ** 2)
# p_2 = p_predicted_interface2 + (0.5 * U_predicted_interface2 ** 2)
# p_3 = p_predicted_interface3 + (0.5 * U_predicted_interface3 ** 2)
#
# # Plot comparision of flow and momentum to check conservation
# fig3 = plt.figure(3, figsize=(22, 12), dpi=300, facecolor='w', frameon=False)
#
# ax31 = fig3.add_subplot(121)
# ax32 = fig3.add_subplot(122)
#
# ax31.plot(tempo, Q_in, 'r--', linewidth=3.0, markersize=0.7, label='Flow of vessel 1')
# ax31.plot(tempo, Q_out, 'b--', linewidth=3.0, markersize=0.7, label='Flow of vessel 2 + 3')
# ax31.legend(loc='upper right', frameon=False, fontsize='large')
#
# ax32.plot(tempo, p_1, 'r--', linewidth=3.0, markersize=0.7, label='Momentum in vessel 1')
# ax32.plot(tempo, p_2, 'b--', linewidth=3.0, markersize=0.7, label='Momentum in vessel 2')
# ax32.plot(tempo, p_3, 'k--', linewidth=3.0, markersize=0.7, label='Momentum in vessel 3')
# ax32.legend(loc='upper right', frameon=False, fontsize='large')
#
# fig3.suptitle('Conservation mass and momentum', size=30)
# ax31.set_xlabel("t[s]", size=20)
# ax31.set_ylabel("q(t)[m^3/s]", size=20)
# ax32.set_xlabel("t[s]", size=20)
# ax32.set_ylabel("p(t)[Pa]", size=20)
#
# # Save the plot
# fig3.savefig(path_figures + "Conservation_mass_and_momentum.png")





noise_data = np.load('./Model/noise5/' + 'Noise_data.npy').item()

# restore np.load for future normal usage
np.load = np_load_old

vel_meas_noise1 = noise_data['velocity_measurements_vessel1']
vel_meas_noise2 = noise_data['velocity_measurements_vessel2']
vel_meas_noise3 = noise_data['velocity_measurements_vessel3']




# Plot a comparative ov noised vs non-noised data
fig_noise = plt.figure(4, figsize=(22, 12), dpi=300, facecolor='w', frameon=False)
ax_noise1 = fig_noise.add_subplot(121)
ax_noise1.plot(t, velocity_measurements_vessel1, 'b-', linewidth=1, markersize=0.5,
               label='Velocity without noise 1')
ax_noise1.plot(t, vel_meas_noise1, 'r-', linewidth=1, markersize=0.5, label='Velocity with noise 1')

ax_noise2 = fig_noise.add_subplot(122)
ax_noise2.plot(T, p_predicted_vessel1, 'r-', linewidth=3.0, markersize=0.7, label='Predicted pressure Vessel1')
ax_noise2.plot(t, pressure_test_vessel1, 'b--', linewidth=3.0, markersize=0.7, label='Reference pressure Vessel1')
ax_noise2.legend(loc='upper right', frameon=False, fontsize='large')
ax_noise2.set_ylim([0, 105000])

fig_noise.suptitle('Input with white noise error at ' + '5' + '%')
ax_noise1.set_xlabel("t in s")
ax_noise1.set_ylabel("Velocity [Vessel 1] in m/s")
ax_noise2.set_xlabel("t in s")
ax_noise2.set_ylabel("Velocity [Vessel 2] in m/s")


# Save the plot
fig_noise.savefig(path_figures + "noise_comparision.png")
