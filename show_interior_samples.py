#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import ctypes
import deep_sdf.data
import sys

import OpenGL.GL as gl
import pypangolin as pango
import numpy as np

if __name__ == "__main__":

    if not len(sys.argv) == 2:
        print("Usage: show_interior_samples.py <npz_file>")
        sys.exit(1)

    npz_filename = sys.argv[1]

    data = deep_sdf.data.read_sdf_samples_into_ram(npz_filename)

    xyz_neg = data[1][:, 0:3].numpy().astype(ctypes.c_float)
    sdf_neg = data[1][:, 3].numpy().astype(ctypes.c_float)
    rgb_neg = np.zeros_like(xyz_neg).astype(ctypes.c_float)

    xyz_pos = data[0][:, 0:3].numpy().astype(ctypes.c_float)
    sdf_pos = data[0][:, 3].numpy().astype(ctypes.c_float)
    rgb_pos = np.zeros_like(xyz_pos).astype(ctypes.c_float)

    s = 10.
    rgb_neg[:,0] = np.minimum(s*sdf_neg / sdf_neg.min(), np.ones_like(sdf_neg))
    rgb_neg[:,1] = 1. - rgb_neg[:,0]
    
    rgb_pos[:,2] = np.minimum( s*sdf_pos / sdf_pos.max(), np.ones_like(sdf_pos))
    rgb_pos[:,1] = 1. - rgb_pos[:,2]

    win = pango.CreateWindowAndBind("Interior Samples | " + npz_filename, 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    pm = pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000)
    mv = pango.ModelViewLookAt(-0, 0.5, -3, 0, 0, 0, pango.AxisY)
    s_cam = pango.OpenGlRenderState(pm, mv)

    handler = pango.Handler3D(s_cam)
    d_cam = (
        pango.CreateDisplay()
        .SetBounds(
            pango.Attach(0),
            pango.Attach(1),
            pango.Attach(0),
            pango.Attach(1),
            -640.0 / 480.0,
        )
        .SetHandler(handler)
    )

    pango.CreatePanel("ui").SetBounds(
        pango.Attach(0), pango.Attach(1), pango.Attach(0), pango.Attach(0)
    )

    while not pango.ShouldQuit():

        gl.glClear(gl.GL_COLOR_BUFFER_BIT + gl.GL_DEPTH_BUFFER_BIT)
        d_cam.Activate(s_cam)


        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(
            3, gl.GL_FLOAT, 0, xyz_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        gl.glColorPointer(
            3, gl.GL_FLOAT, 0, rgb_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        gl.glDrawArrays(gl.GL_POINTS, 0, xyz_neg.shape[0])

        gl.glVertexPointer(
            3, gl.GL_FLOAT, 0, xyz_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        gl.glColorPointer(
            3, gl.GL_FLOAT, 0, rgb_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        gl.glDrawArrays(gl.GL_POINTS, 0, xyz_pos.shape[0])
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        pango.FinishFrame()
