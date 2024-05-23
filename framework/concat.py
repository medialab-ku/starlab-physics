import sys

import igl
import os
import numpy as np

def concat_mesh(model_dir, model_names, translations, scales):
    vtx_id_offsets = []
    num_models = len(model_names)
    accum = 0
    v_list = []
    f_list = []

    for i in range(num_models):
        vtx_id_offsets.append(accum)
        model_path = os.path.join(model_dir, model_names[i])
        v, _, n, f, _, _ = igl.read_obj(model_path)
        v_list.append(v)
        f_list.append(f)
        accum += len(v)

    for i in range(num_models):
        v = v_list[i]
        f = f_list[i]

        scale = lambda x, sc: sc * x
        v = scale(v, scales[i])

        translate = lambda x, trans: x + trans
        v_list[i] = np.apply_along_axis(lambda row: translate(row, translations[i]), 1, v)

        add_offset = lambda vid, offset: vid + offset
        f_list[i] = add_offset(f, vtx_id_offsets[i])

    v_concat = np.concatenate(v_list, axis=0)
    f_concat = np.concatenate(f_list, axis=0)

    model_directory_cat = "../models/"
    model_name_concat = "concat.obj"

    model_path_concat = os.path.join(model_directory_cat, model_name_concat)
    igl.write_obj(model_path_concat, v_concat, f_concat)

