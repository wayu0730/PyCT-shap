#!/usr/bin/env python3

import os
import libct.explore
import json
from libct.utils import get_module_from_rootdir_and_modpath, get_function_from_module_and_funcname
from utils.pyct_attack_exp import get_save_dir_from_save_exp

PYCT_ROOT = './'
MODEL_ROOT = os.path.join(PYCT_ROOT, 'model')


def run(model_name, in_dict, con_dict, norm, solve_order_stack, save_exp=None,
        max_iter=0, single_timeout=900, timeout=900, total_timeout=900, verbose=1,
        limit_change_range=None, only_first_forward=False):

    model_path = os.path.join(MODEL_ROOT, f"{model_name}.h5")
    modpath = os.path.join(PYCT_ROOT, f"dnn_predict_common.py")
    func = "predict"
    funcname = t if (t:=func) else modpath.split('.')[-1]
    save_dir = None
    smtdir = None


    dump_projstats = False
    file_as_total = False
    formula = None
    include_exception = False
    lib = None
    logfile = None
    root = os.path.dirname(__file__)
    safety = 0

    # verbose = 1 # 5:all, 3:>=DEBUG. 2:including SMT, 1: >=INFO
    # norm = True


    statsdir = None
    if dump_projstats:
        statsdir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "project_statistics",
            os.path.abspath(root).split('/')[-1], modpath, funcname)


    module = get_module_from_rootdir_and_modpath(root, modpath)
    func_init_model = get_function_from_module_and_funcname(module, "init_model")
    execute = get_function_from_module_and_funcname(module, funcname)
    func_init_model(model_path)

    ##############################################################################
    # This section creates an explorer instance and starts our analysis procedure!    
    if save_exp is not None:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"                    
        save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=only_first_forward)
        
        if save_exp.get('save_smt', False):        
            smtdir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=only_first_forward)        
    
    engine = libct.explore.ExplorationEngine(solver='cvc4', timeout=timeout, safety=safety,
                                            store=formula, verbose=verbose, logfile=logfile,
                                            statsdir=statsdir, smtdir=smtdir,
                                            save_dir=save_dir, input_name=save_exp['input_name'],
                                            module_=module, execute_=execute,
                                            only_first_forward=only_first_forward)


    result = engine.explore(
        modpath, in_dict, concolic_dict=con_dict, root=root, funcname=func, max_iterations=max_iter,
        single_timeout=single_timeout, total_timeout=total_timeout, deadcode=set(),
        include_exception=include_exception, lib=lib,
        file_as_total=file_as_total, norm=norm, solve_order_stack=solve_order_stack,
        limit_change_range=limit_change_range, 
    )
        
            
    return result

