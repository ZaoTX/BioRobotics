import re
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import circle_fit
import sklearn.linear_model


def parse_body_position(data_three_vals) -> np.array:
    expr = r'[A-Za-z0-9]+\(.+=(-?[0-9\.]+),.+=(-?[0-9\.]+),.+=(-?[0-9\.]+)\)'
    positions = []
    for i, bp in enumerate(data_three_vals):
        matches = re.search(expr, bp)
        if matches is not None:
            positions.append(np.array([float(matches.group(1)),
                float(matches.group(2)),
                float(matches.group(3))]))
        else:
            positions.append(np.array([float('nan'), float('nan'), float('nan')]))
    return np.array(positions)


# Euler length of some path computed as the sum of pairwise Eulerian distances.
def euler_length(points: np.array) -> float:
    return np.sqrt(np.power(points[1:] - points[:-1], 2).sum(axis=1)).sum()


def path_direction(points: np.array) -> float:
    def unit_vector(v):
        return v / np.linalg.norm(v)

    def angle_between(v1, v2):
        return np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

    angle = 0.0
    for v1, v2 in itertools.pairwise(points[1:] - points[:-1]):
        angle += angle_between(v1, v2)
    return angle


stuff = np.array([
    0.0, 0.0,
    1.0, 1.0,
    2.0, 1.0,
    3.0, 0.0,
    2.0, -1.0,
    1.0, -1.0,
    0.0, 0.0,
    ]).reshape((-1, 2,))
# stuff[:, 1] *= -1

print(path_direction(stuff))


if __name__ == '__meeain__':
    bad_straight_measurements = {
        # Have crashed?
        24, 33,
        # Have nans
        26,
    }
    straight = pd.read_csv('data/tracks_straight.csv')
    groups = straight.groupby(by=['track', 'left_speed', 'right_speed'])

    # list containing [lspeed, rspeed, velo, angvelo]
    measurements = []
    measurements2 = []

    for g_idx, g_data in groups:
        track, left_speed, right_speed = g_idx[0], g_idx[1], g_idx[2]

        # skip bad tracks
        # TODO: handle bad tracks properly
        if track in bad_straight_measurements:
            continue
        gsrn = g_data[['frame', 'timestamp', 'position', 'rotation']].to_numpy()

        # Make sure the frames are incrementing by one (ie. in order)
        assert (gsrn[:-1, 0] == gsrn[1:, 0] - 1).all()
        # Make sure time is running forward
        assert (gsrn[:-1, 1] < gsrn[1:, 1]).all()

        # RT6DBodyPosition(x=-1064.4171142578125, y=-43.048301696777344, z=94.53657531738281)
        # RT6DBodyEuler(a1=-0.555755615234375, a2=0.3585425913333893, a3=4.479945182800293)
        positions = parse_body_position(gsrn[:, 2])
        rotations = parse_body_position(gsrn[:, 3])

        # skip tracks with nans
        # TODO: fix
        if np.isnan(positions).any():
            continue

        assert positions.shape == rotations.shape

        xc, yc, r, sigma = circle_fit.taubinSVD(positions[:, :2])

        tracktime = gsrn[-1, 1] - gsrn[0, 1]
        tracktime = tracktime / 10**6
        track_length = euler_length(positions[:, [0, 1]])
        track_angle = path_direction(positions[:, [0, 1]])
        velocity = track_length / tracktime
        angular_velocity = track_length / (2.0 * np.pi * r) / tracktime

        fig, axes = plt.subplots(1, 1)
        fig.suptitle(f'track={track}, left_speed={left_speed}, right_speed={right_speed}\n'
                     + f'velo={velocity:.3f}, angvelo={angular_velocity:.3f}\n'
                     + f'length={track_length:.3f}, angle={track_angle:.3f}')
        axes.scatter(positions[:, 0], positions[:, 1], s=0.1)
        axes.scatter(positions[0, 0], positions[0, 1])
        axes.scatter([xc], [yc])
        axes.add_patch(plt.Circle((xc, yc), r, color='r', fill=False))
        axes.axis('equal')
        fig.show()

        print(f'track={track}, vl={left_speed}, vr={right_speed}, v={velocity}, va={angular_velocity}')
        print(f'tracktime={tracktime}, length={track_length}, angle={track_angle}')
        measurements.append(np.array([left_speed, right_speed, velocity, angular_velocity]))

    measurements = np.array(measurements)
    print(measurements)
    plt.scatter(measurements[:, 0], measurements[:, 1], c=measurements[:, 2])
    plt.show()
    plt.scatter(measurements[:, 0], measurements[:, 1], c=measurements[:, 3])
    plt.show()

    model = sklearn.linear_model.LinearRegression().fit(measurements[:, [2, 3]], measurements[:, [0, 1]])
    print(model.intercept_)
    print(model.coef_)
    print(np.array([400.0, 0.0]) * model.coef_)
    print(np.array([400.0, 0.0]) @ model.coef_)

