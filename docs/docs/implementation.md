#### Track Generation:

For this project we developed a couple of scripts to create tracks, the tracks comprise of some obstacles that can either protrude from the ceiling of the track or upwards from the floor. To implement this, we first generate a random number of tuples which contain four attributes; width and height of the obstacles, starting location in the window and if it is on the ceiling or floor of the window. The set of tuples that are generated can then be made into windows, where a window will contain one obstacle.

To create the track, a random number of windows are then selected, trimmed and concatenated to make a track. At the window generation phase, the window is initially created with zeros and ones where ones are the obstacle. We then fill the zeros in with a function (```x**2 + 2 * y**2```) to generate a continuous value, which are then be binned into seven distinct bins which correspond to different actions of the wind.



