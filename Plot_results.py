# fig = plt.figure()
# fig.canvas.draw()
# plt.subplot(2, 2, 1)
# plt.title('Best Realized Patter', fontsize=10)
# plt.xlabel('Track Length', fontsize=8)
# plt.ylabel('Track Width', fontsize=8)
# my_axis = plt.gca()
# my_axis.set_xlim(0, HeliWorld.track_width)
# my_axis.set_ylim(0, HeliWorld.track_height)
# im1 = plt.imshow(HeliWorld.track,
#                  cmap=plt.cm.jet,
#                  interpolation='nearest',
#                  vmin=-1,
#                  vmax=8)
# plt.colorbar(im1, fraction=0.01, pad=0.01)
#
# plt.subplot(2, 2, 2)
# plt.title('Final Q Matrix', fontsize=10)
# plt.xlabel('Track Length', fontsize=8)
# plt.ylabel('Track Width', fontsize=8)
# my_axis = plt.gca()
# my_axis.set_xlim(0, HeliWorld.track_width)
# my_axis.set_ylim(0, HeliWorld.track_height)
# a = np.zeros(shape=(HeliWorld.track_height,
#                     HeliWorld.track_width))
# im2 = plt.imshow(a)
# plt.colorbar(im2, fraction=0.01, pad=0.01)
#
# plt.subplot(2, 2, 3)
# plt.title('Completion Time Chart', fontsize=10)
# plt.xlabel('Trial Numbers', fontsize=8)
# plt.ylabel('Seconds Per Trial', fontsize=8)
# my_axis = plt.gca()
# my_axis.set_xlim(0, settings['trials'])
#
# plt.subplot(2, 2, 4)
# plt.title('Learning Chart', fontsize=10)
# plt.xlabel('Trial Numbers', fontsize=8)
# plt.ylabel('End Location', fontsize=8)
# my_axis = plt.gca()
# my_axis.set_xlim(0, settings['trials'])
# my_axis.set_ylim(0, 1)
#
# colors = ['black', 'green', 'red', 'cyan', 'magenta',
#           'yellow', 'blue', 'white', 0.3, 0.55, 0.85]
#
#
#         plt.subplot(2, 2, 4)
#                 # Final Location Plot
#                 plt.scatter(HeliWorld.trials,
#                             Helicopter1.final_location[-1][0] /
#                             float(HeliWorld.track_width),
#                             s=np.pi * (1 * 1) ** 2,
#                             c=colors[value_iter],
#                             alpha=0.5)
#                 plt.legend()
#                 # Completion Time Chart
#                 plt.subplot(2, 2, 3)
#                 plt.scatter(value[0],
#                             value[1],
#                             s=np.pi * (1 * 1) ** 2,
#                             c=colors[value_iter],
#                             alpha=0.5)
#
# fig.suptitle('Time for Trial Completion: {} - Current State: {} - Current Location: {}\n\
#                              Trials Completed: {} with Total Time {:.3f} seconds\n\
#                              Agent Model: {} \n\
#                              Agent Parameters: alpha {} - epsilon {:.4f} - gamma {} - Number of Actions: {}\n\
#                              World Paramers: Length of Track: {} - Width of Track: {} - CASE: {}'.format(
#                     HeliWorld.trials,
#                     Helicopter1.current_state,
#                     Helicopter1.current_location,
#                     value[0],
#                     time() - st + 1e-9,
#                     Utils.titles[settings['model'] - 1],
#                     settings['alpha'],
#                     settings['epsilon'],
#                     settings['gamma'],
#                     settings['nb_actions'],
#                     HeliWorld.track_width,
#                     HeliWorld.track_height,
#                     case),
#                     fontsize=10,
#                     horizontalalignment='center',
#                     verticalalignment='top')
#
#                 # Plotting Real-time plot
#                 plt.subplot(2, 2, 1)
#                 plt.imshow(HeliWorld.track,
#                            cmap=plt.cm.jet,
#                            interpolation='nearest',
#                            vmin=-1,
#                            vmax=8)
#                 plt.scatter(Helicopter1.current_location[0],
#                             Helicopter1.current_location[1],
#                             s=np.pi * (1 * 1) ** 2,
#                             c=colors[value_iter])
#
#                 plt.subplot(2, 2, 2)
#                 plt.imshow(a)
#                 plt.pause(1e-10)
