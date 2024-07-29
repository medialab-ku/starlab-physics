import sys

import igl
import os
import numpy as np
from pathlib import Path

# def rot2mat(rot):
#
#     return np.matrix([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
def concat_mesh(concat_model_name, model_dir, model_names, translations, scales):
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

    vtx_id_offsets.append(accum)

    for i in range(num_models):
        v = v_list[i]
        f = f_list[i]

        scale = lambda x, sc: sc * x
        v = scale(v, scales[i])

        # for j in range(vtx_id_offsets[i + 1], vtx_id_offsets[i]):
        #     offset = vtx_id_offsets[i]
        #     xi = v_list[j + offset]
        #     xi_4d = np.array([xi, 1])
        #     rot = rotations[i]

        translate = lambda x, trans: x + trans
        v_list[i] = np.apply_along_axis(lambda row: translate(row, translations[i]), 1, v)

        add_offset = lambda vid, offset: vid + offset
        f_list[i] = add_offset(f, vtx_id_offsets[i])

    # if num_models > 1:
    v_concat = np.concatenate(v_list, axis=0)
    f_concat = np.concatenate(f_list, axis=0)

    # model_directory_cat = "../models/"
    # model_name_concat = "concat.obj"
    model_path = Path(__file__).resolve().parent.parent.parent / "models"
    # print(model_path)
    model_path_concat = str(model_path / concat_model_name)
    # model_path_concat = os.path.join(model_path, concat_model_name)
    igl.write_obj(model_path_concat, v_concat, f_concat)

    return vtx_id_offsets
    # else:
    #     v_concat = v_list[0]
    #     f_concat = f_list[0]
    #
    #     model_directory_cat = "../models/"
    #     model_name_concat = "concat.obj"
    #
    #     model_path_concat = os.path.join(model_directory_cat, model_name_concat)
    #     igl.write_obj(model_path_concat, v_concat, f_concat)

