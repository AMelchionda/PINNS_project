import matplotlib.pyplot as plt
import numpy as np

from Y_shaped_pinns_INVERSE_PB import OneDBioPINN

if __name__ == "__main__":

    # READ INPUT FILES -------------------------------------------------------------------------------------------------

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    N_f = 2000
    
    input_vessel_1 = np.load("./Inputs/input_1.npy").item()
    output_vessel_2 = np.load("./Inputs/output_2.npy").item()
    output_vessel_3 = np.load("./Inputs/output_3.npy").item()

    t = input_vessel_1["Time"][:, None]
    
    test_vessel_1 = np.load("./Inputs/test_1.npy").item()
    test_vessel_2 = np.load("./Inputs/test_2.npy").item()
    test_vessel_3 = np.load("./Inputs/test_3.npy").item()


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

    # ADD WHITE NOISE TO VELOCITY MEASUREMENTS (PARTICULAR CASE)
    # samples1 = np.random.normal(0, 0.1, size=velocity_measurements_vessel1.shape[0])
    # vel_meas_noise1 = np.zeros((velocity_measurements_vessel1.shape[0], 1))
    # for i in range(velocity_measurements_vessel1.shape[0]):
    #    vel_meas_noise1[i] = velocity_measurements_vessel1[i]*(1 + samples1[i])
    # # fig1 = plt.figure(1)
    # # ax = fig1.add_subplot(111)
    # # ax.plot(t, velocity_measurements_vessel1)
    # # ax.plot(t, vel_meas_noise1)
    # # plt.show()
    #
    # samples2 = np.random.normal(0, 0.1, size=velocity_measurements_vessel2.shape[0])
    # vel_meas_noise2 = np.zeros((velocity_measurements_vessel2.shape[0], 1))
    # for i in range(velocity_measurements_vessel2.shape[0]):
    #     vel_meas_noise2[i] = velocity_measurements_vessel2[i] * (1 + samples2[i])
    #
    # samples3 = np.random.normal(0, 0.1, size=velocity_measurements_vessel3.shape[0])
    # vel_meas_noise3 = np.zeros((velocity_measurements_vessel3.shape[0], 1))
    # for i in range(velocity_measurements_vessel3.shape[0]):
    #     vel_meas_noise3[i] = velocity_measurements_vessel3[i] * (1 + samples3[i])



#     amp_vel1 = np.amax(velocity_measurements_vessel1) - np.mean(velocity_measurements_vessel1)
#     samples1 = np.random.normal(0, np.sqrt(0.01*amp_vel1), size=velocity_measurements_vessel1.shape[0])
#     velocity_measurementss_vessel1 = np.ones((velocity_measurements_vessel1.shape[0], 1))
#     for i in range(velocity_measurements_vessel1.shape[0]):
#         velocity_measurementss_vessel1[i] = velocity_measurements_vessel1[i] + samples1[i]
#
#     fig1 = plt.figure(1)
#     ax = fig1.add_subplot(111)
#     ax.plot(t, velocity_measurements_vessel1)
#     ax.plot(t, velocity_measurementss_vessel1)
#     plt.show()
#
#
#     #######
#
#     x_watts = velocity_measurements_vessel1 ** 2
#     # Set a target SNR
#     target_snr_db = 20
#     # Calculate signal power and convert to dB
#     sig_avg_watts = np.mean(x_watts)
#     sig_avg_db = 10 * np.log10(sig_avg_watts)
#     # Calculate noise according to [2] then convert to watts
#     noise_avg_db = sig_avg_db - target_snr_db
#     noise_avg_watts = 10 ** (noise_avg_db / 10)
#     # Generate an sample of white noise
#     mean_noise = 0
#     noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
#     x_wattsNEW = np.ones((x_watts.shape[0], 1))
#     for i in range(x_watts.shape[0]):
#         x_wattsNEW[i] = velocity_measurements_vessel1[i]+ noise_volts[i]
#
#     fig2 = plt.figure(1)
#     ax = fig2.add_subplot(111)
#     ax.plot(t, x_watts)
#     ax.plot(t, 10* np.log10(x_wattsNEW**2))
#     plt.show()
#
#
# #######
#
#     max_vel1 = np.amax(velocity_measurements_vessel1)
#     samples1 = np.random.normal(0, np.sqrt(0.01 * max_vel1), size=velocity_measurements_vessel1.shape[0])
#     velocity_measurements_vessel1 = velocity_measurements_vessel1 + samples1
#
#     max_vel1 = np.amax(velocity_measurements_vessel1)
#     samples1 = np.random.normal(0, np.sqrt(0.01 * max_vel1), size=velocity_measurements_vessel1.shape[0])
#     velocity_measurements_vessel1 = velocity_measurements_vessel1 + samples1


    # restore np.load for future normal usage
    np.load = np_load_old

    # INITIALIZE VARIABLES FOR TRAINING --------------------------------------------------------------------------------
    
    N_u = t.shape[0]

    # Structure of the Neural Network
    # layers = [2, 100, 100, 100, 100, 100, 100, 100, 3]
    layers = [2, 50, 50, 50, 50, 50, 50, 50, 3]

    lower_bound_t = t.min(0)
    upper_bound_t = t.max(0)
    
    lower_bound_vessel_1 = 0.0   
    upper_bound_vessel_1 = 0.1703
    
    lower_bound_vessel_2 = 0.1703
    upper_bound_vessel_2 = 0.1773
    
    lower_bound_vessel_3 = 0.1703
    upper_bound_vessel_3 = 0.1770
    
    bif_points = 0.1703
    
    X_initial_vessel1 = np.linspace(lower_bound_vessel_1, upper_bound_vessel_1, N_u)[:, None]
    X_initial_vessel2 = np.linspace(lower_bound_vessel_2, upper_bound_vessel_2, N_u)[:, None]
    X_initial_vessel3 = np.linspace(lower_bound_vessel_3, upper_bound_vessel_3, N_u)[:, None]
    
    T_initial = lower_bound_t*np.ones((N_u))[:, None]
    
    X_boundary_vessel1 = lower_bound_vessel_1*np.ones((N_u))[:, None]
    X_boundary_vessel2 = upper_bound_vessel_2*np.ones((N_u))[:, None]
    X_boundary_vessel3 = upper_bound_vessel_3*np.ones((N_u))[:, None]

    T_boundary = t
        
    X_measurement_vessel1 = np.vstack((X_initial_vessel1, X_boundary_vessel1))
    X_measurement_vessel2 = np.vstack((X_initial_vessel2, X_boundary_vessel2))    
    X_measurement_vessel3 = np.vstack((X_initial_vessel3, X_boundary_vessel3))    
    
    T_measurement = np.vstack((T_initial, T_boundary))

    # Collocation points
    X_residual_vessel1 = lower_bound_vessel_1 + (upper_bound_vessel_1-lower_bound_vessel_1) * \
                         np.random.random((N_f))[:,None]
    X_residual_vessel2 = lower_bound_vessel_2 + (upper_bound_vessel_2-lower_bound_vessel_2) * \
                         np.random.random((N_f))[:,None]
    X_residual_vessel3 = lower_bound_vessel_3 + (upper_bound_vessel_3-lower_bound_vessel_3) * \
                         np.random.random((N_f))[:,None]
    
    T_residual = lower_bound_t + (upper_bound_t-lower_bound_t)*np.random.random((N_f))[:,None]  #tempi random per collocations pts.
   
    A_initial_vessel1 = 1.35676200E-05*np.ones((N_u,1))
    A_initial_vessel2 = 1.81458400E-06*np.ones((N_u,1))
    A_initial_vessel3 = 1.35676200E-05*np.ones((N_u,1))

    U_initial_vessel1 = 0.*np.ones((N_u,1))
    U_initial_vessel2 = 0.*np.ones((N_u,1))
    U_initial_vessel3 = 0.*np.ones((N_u,1))
        
    A_training_vessel1 = np.vstack((A_initial_vessel1,area_measurements_vessel1))
    U_training_vessel1 = np.vstack((U_initial_vessel1,velocity_measurements_vessel1))

    A_training_vessel2 = np.vstack((A_initial_vessel2,area_measurements_vessel2))
    U_training_vessel2 = np.vstack((U_initial_vessel2,velocity_measurements_vessel2))
    
    A_training_vessel3 = np.vstack((A_initial_vessel3,area_measurements_vessel3))
    U_training_vessel3 = np.vstack((U_initial_vessel3,velocity_measurements_vessel3))

    # INITIALIZE THE NEURAL NETWORK ------------------------------------------------------------------------------------
              
    model = OneDBioPINN(X_measurement_vessel1, 
                               X_measurement_vessel2,
                               X_measurement_vessel3,
                               A_training_vessel1,  U_training_vessel1,
                               A_training_vessel2,  U_training_vessel2,
                               A_training_vessel3,  U_training_vessel3,
                               X_residual_vessel1, 
                               X_residual_vessel2, 
                               X_residual_vessel3, 
                               T_residual, T_measurement, layers, bif_points)

    # TRAIN THE NEURAL NETWORK -----------------------------------------------------------------------------------------

    # Train the neural network
    model.train(92000, 1e-3)
    model.train(40000, 1e-4)

    beta_1_value = 10 ** model.sess.run(model.beta1V)
    beta_2_value = 10 ** model.sess.run(model.beta2V)
    beta_3_value = 10 ** model.sess.run(model.beta3V)

    # Specify where to save the files (folder name)
    path = 'inv2/'

    # Save the trained neural network
    # model.save_NN(path, 1)

    # LOAD THE NEURAL NETWORK ------------------------------------------------------------------------------------------
    # Doesn't work properly... can't restore Xmean, Xstd, Tmean and Tstd....
    # ... To be implemented

    # model.load_NN('final_model')

    # EVALUATE, SAVE, AND PLOT PREDICTIONS -----------------------------------------------------------------------------

    # Set folder for saving plots
    path_figures = './Figures/' + path

    # Order T_Residuals
    T_residual.sort(axis=0)

    # Evaluate prediction at test points
    X_test_vessel1 = 0.1*np.ones((T_residual.shape[0], 1))
    X_test_vessel2 = 0.176*np.ones((T_residual.shape[0], 1))
    X_test_vessel3 = 0.174*np.ones((T_residual.shape[0], 1))

    A_predicted_vessel1, U_predicted_vessel1, p_predicted_vessel1 = model.predict_vessel1(X_test_vessel1, T_residual,
                                                                                          path + 'Prediction_Vessel1')
    A_predicted_vessel2, U_predicted_vessel2, p_predicted_vessel2 = model.predict_vessel2(X_test_vessel2, T_residual,
                                                                                          path + 'Prediction_Vessel2')
    A_predicted_vessel3, U_predicted_vessel3, p_predicted_vessel3 = model.predict_vessel3(X_test_vessel3, T_residual,
                                                                                          path + 'Prediction_Vessel3')

    # Plot VELOCITY comparision between prediction and reference measurements in 3 vessels
    fig1 = plt.figure(1, figsize=(22, 22), dpi=300, facecolor='w', frameon=False)

    ax11 = fig1.add_subplot(131)
    ax12 = fig1.add_subplot(132)
    ax13 = fig1.add_subplot(133)

    ax11.plot(t, velocity_test_vessel1, 'b-', linewidth=1, markersize=0.5, label='Reference velocity Vessel1')
    ax11.plot(T_residual, U_predicted_vessel1, 'r--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel1')

    ax12.plot(t, velocity_test_vessel2, 'b-', linewidth=1, markersize=0.5, label='Reference velocity Vessel2')
    ax12.plot(T_residual, U_predicted_vessel2, 'r--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel2')

    ax13.plot(t, velocity_test_vessel3, 'b-', linewidth=1, markersize=0.5, label='Reference velocity Vessel3')
    ax13.plot(T_residual, U_predicted_vessel3, 'r--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel3')

    fig1.suptitle('Comparative velocity')
    ax11.set_xlabel("t in s")
    ax11.set_ylabel("Velocity in m/s")
    ax12.set_xlabel("t in s")
    ax12.set_ylabel("Velocity in m/s")
    ax13.set_xlabel("t in s")
    ax13.set_ylabel("Velocity in m/s")

    fig1.savefig(path_figures + "Comparative_Velocity.png")  # Save figure

    # Plot PRESSURE comparision between prediction and reference measurements in 3 vessels
    fig2 = plt.figure(2, figsize=(22, 22), dpi=300, facecolor='w', frameon = False)

    ax21 = fig2.add_subplot(131)
    ax22 = fig2.add_subplot(132)
    ax23 = fig2.add_subplot(133)

    ax21.plot(t, pressure_test_vessel1, 'b-', linewidth=1, markersize=0.5, label='Reference pressure Vessel1')
    ax21.plot(T_residual, p_predicted_vessel1, 'r--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')

    ax22.plot(t, pressure_test_vessel2, 'b-', linewidth=1, markersize=0.5, label='Reference pressure Vessel2')
    ax22.plot(T_residual, p_predicted_vessel2, 'r--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel2')

    ax23.plot(t, pressure_test_vessel3, 'b-', linewidth=1, markersize=0.5, label='Reference pressure Vessel3')
    ax23.plot(T_residual, p_predicted_vessel3, 'r--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel3')

    fig2.suptitle('Comparative pressure')
    ax21.set_xlabel("t in s")
    ax21.set_ylabel("Pressure in Pa")
    ax22.set_xlabel("t in s")
    ax22.set_ylabel("Pressure in Pa")
    ax23.set_xlabel("t in s")
    ax23.set_ylabel("Pressure in Pa")

    fig2.savefig(path_figures + "Comparative_Pressure.png")  # Save figure

    # Evaluate the solution at interface points
    X_test_bif = bif_points * np.ones((T_residual.shape[0], 1))

    A_predicted_interface1, U_predicted_interface1, p_predicted_interface1 = \
        model.predict_vessel1(X_test_bif, T_residual, path + 'Prediction_Interface1')
    A_predicted_interface2, U_predicted_interface2, p_predicted_interface2 = \
        model.predict_vessel2(X_test_bif, T_residual, path + 'Prediction_Interface2')
    A_predicted_interface3, U_predicted_interface3, p_predicted_interface3 = \
        model.predict_vessel3(X_test_bif, T_residual, path + 'Prediction_Interface3')

    # Compute flow in vessels
    Q1 = A_predicted_interface1 * U_predicted_interface1
    Q2 = A_predicted_interface2 * U_predicted_interface2
    Q3 = A_predicted_interface3 * U_predicted_interface3

    Q_in = Q1
    Q_out = Q2 + Q3

    # Compute the momentum in vessels
    p_1 = p_predicted_interface1 + (0.5 * U_predicted_interface1 ** 2)
    p_2 = p_predicted_interface2 + (0.5 * U_predicted_interface2 ** 2)
    p_3 = p_predicted_interface3 + (0.5 * U_predicted_interface3 ** 2)

    # Plot comparision of flow and momentum to check conservation
    fig3 = plt.figure()

    ax31 = fig3.add_subplot(121)
    ax32 = fig3.add_subplot(122)

    ax31.plot(T_residual, Q_in, 'r-', label='Flow of vessel 1')
    ax31.plot(T_residual, Q_out, 'b--', label='Flow of vessel 2 + 3')

    ax32.plot(T_residual, p_1, 'r-', label='Momentum in vessel 1')
    ax32.plot(T_residual, p_2, 'b--', label='Momentum in vessel 2')
    ax32.plot(T_residual, p_3, 'g--', label='Momentum in vessel 3')

    ax31.set_xlabel("t in s")
    ax31.set_ylabel("q(t)[m^3/s]")
    ax32.set_xlabel("t in s")
    ax32.set_ylabel("p(t)[Pa]")

    # Save the plot
    fig3.savefig(path_figures + "Conservation_mass_and_momentum.png")

    # Save and print the solution at test points
    X_new_test_vessel1 = 0.1 * np.ones((t.shape[0], 1))
    X_new_test_vessel2 = 0.176 * np.ones((t.shape[0], 1))
    X_new_test_vessel3 = 0.174 * np.ones((t.shape[0], 1))

    A_predicted_test1, U_predicted_test1, p_predicted_test1 = model.predict_vessel1(X_new_test_vessel1, t,
                                                                                    path + 'Prediction_Test1')
    A_predicted_test2, U_predicted_test2, p_predicted_test2 = model.predict_vessel2(X_new_test_vessel2, t,
                                                                                    path + 'Prediction_Test2')
    A_predicted_test3, U_predicted_test3, p_predicted_test3 = model.predict_vessel3(X_new_test_vessel3, t,
                                                                                    path + 'Prediction_Test3')

    # Compute L2 errors for table
    error_p1 = np.linalg.norm(pressure_test_vessel1 - p_predicted_test1, 2) / np.linalg.norm(pressure_test_vessel1, 2)
    error_p2 = np.linalg.norm(pressure_test_vessel2 - p_predicted_test2, 2) / np.linalg.norm(pressure_test_vessel2, 2)
    error_p3 = np.linalg.norm(pressure_test_vessel3 - p_predicted_test3, 2) / np.linalg.norm(pressure_test_vessel3, 2)

    print('L2 error in vessel 1:', error_p1)
    print('L2 error in vessel 2:', error_p2)
    print('L2 error in vessel 3:', error_p3)
    print('Estimate of beta1:', beta_1_value)
    print('Estimate of beta2:', beta_2_value)
    print('Estimate of beta3:', beta_3_value)
