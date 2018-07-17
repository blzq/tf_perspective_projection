#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf


def project(points_world_space, cams_pos, cams_angles, focal_lens):
    """
    Given a set of 3D points (batch, 3) and camera parameters (batch, N),
    return points projected onto 2D screen in shape (batch, 2).
    https://en.wikipedia.org/wiki/3D_projection#Perspective_projection
    https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/
    cams_angles should be in axis-angle representation.
    Note that the camera before rotation points in +z in world coords - to
    flip this change ones to -ones. Scaling of z to [0, 1] using near and far
    clip planes not implemented. No clipping of -z (behind camera) values.
    """
    batch_size = tf.shape(points_world_space)[0]
    rot_mats = rodrigues_batch(cams_angles)

    # Points to camera space
    points_cam_space = tf.matmul(
        rot_mats, (points_world_space - cams_pos)[:, :, tf.newaxis])
    points_cam_space = tf.squeeze(points_cam_space)

    zeros = tf.zeros([batch_size])
    ones = tf.ones([batch_size])

    points_cam_homog = tf.concat(
        [points_cam_space, ones[:, tf.newaxis]], axis=1)[:, :, tf.newaxis]

    # Projection matrix.
    # pyformat: disable
    proj_mat_1 = tf.stack([focal_lens, zeros     ,  zeros, zeros], axis=1)
    proj_mat_2 = tf.stack([zeros     , focal_lens,  zeros, zeros], axis=1)
    proj_mat_3 = tf.stack([zeros     , zeros     ,  ones , zeros], axis=1)
    proj_mat_4 = tf.stack([zeros     , zeros     ,  ones , zeros], axis=1)
    # pyformat: enable
    proj_mat = tf.stack([proj_mat_1, proj_mat_2, proj_mat_3, proj_mat_4],
                        axis=1)
    points_2d_homog = tf.matmul(proj_mat, points_cam_homog)
    points_2d_homog = tf.squeeze(points_2d_homog)

    # Homogeneous divide
    points_2d = points_2d_homog / points_2d_homog[:, 3, tf.newaxis]

    return points_2d[:, :2]


def rodrigues_batch(rvecs):
    """
    Convert a batch of axis-angle rotations in rotation vector form shaped
    (batch, 3) to a batch of rotation matrices shaped (batch, 3, 3).
    See
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    batch_size = tf.shape(rvecs)[0]
    tf.assert_equal(tf.shape(rvecs)[1], 3)

    thetas = tf.norm(rvecs, axis=1, keepdims=True)
    is_zero = tf.equal(tf.squeeze(thetas), 0.0)
    u = rvecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = tf.zeros([batch_size])  # for broadcasting
    Ks_1 = tf.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = tf.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = tf.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = tf.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    Rs = tf.eye(3, batch_shape=[batch_size]) + \
         tf.sin(thetas)[..., tf.newaxis] * Ks + \
         (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)

    # Avoid returning NaNs where division by zero happened
    return tf.where(is_zero,
                    tf.eye(3, batch_shape=[batch_size]), Rs)


# For testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from math import pi

    tf.enable_eager_execution()
    tf.executing_eagerly()

    points_3d = tf.constant([[0, 0, 0], [1, 1, 5]], dtype=tf.float32)
    cam_pos = tf.constant([[0, 0, -2], [0, 0, -2]], dtype=tf.float32)
    cam_angles = tf.constant([[0, pi/6, 0], [0, pi/6, 0]], dtype=tf.float32)
    focal_lens = tf.constant([0.1, 0.1], dtype=tf.float32)

    points = project(points_3d, cam_pos, cam_angles, focal_lens)
    print(points)

    plt.scatter(points[:, 0], points[:, 1])
    plt.show()
